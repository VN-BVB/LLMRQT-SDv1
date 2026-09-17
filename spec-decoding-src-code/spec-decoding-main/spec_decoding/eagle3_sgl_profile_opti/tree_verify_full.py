# SPDX-License-Identifier: Apache-2.0
"""
全层 Target 树形 Verify

对 draft 树上 L 个节点 一次 跑完整 PreTrainedModel forward（带已有
past_key_values 前缀），在 每一层 attention 里对新 token 段的 L×L
子块施加 树掩码（兄弟分支不可互看），再取 logits[0, i, :] 做叶子路径
greedy 接受

实现要点
--------
- 通过 tree_verify_attention_context 临时 patch decoder 的
  _update_causal_mask / _prepare_decoder_attention_mask，在因果 mask 的
  右下角 [-L:, -L:] 叠树约束
- verify 步 use_cache=False，避免把整棵 draft 树写进 KV；接受段仍由
  runner 用 append 重放对齐 cache。
- verify_attn_backend=eager（默认回退）：_attn_implementation=eager + 树 mask patch。
- verify_attn_backend=flash_attn（topk==1）：官方 Flash Attention，无树 mask。
- verify_attn_backend=triton_tree（topk>1）：SGLang extend_attention_fwd + custom_mask。
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_allow_mask_4d,
    build_tree_cross_attn_bias_with_prefix,
    layout_tree_position_ids,
)
from .tree_verify_attn import (
    build_triton_tree_verify_metadata,
    resolve_verify_attn_backend,
    tree_verify_attention_context_flash,
    tree_verify_triton_attention_context,
)
from .tree_verify_parallel import (
    path_node_indices_root_to_leaf,
    verify_leaf_path_against_parallel_logits,
)


def _decoder_stack(target_m: PreTrainedModel) -> nn.Module:
    if hasattr(target_m, "model") and isinstance(getattr(target_m, "model"), nn.Module):
        return target_m.model
    raise TypeError(
        "full_model_tree verify expects a HF CausalLM with .model (e.g. Llama/Qwen)"
    )


def _apply_tree_block_to_causal(
    causal: Optional[torch.Tensor],
    tree_mask_block: torch.Tensor,
    *,
    seq_len: int,
    min_dtype: torch.dtype,
) -> torch.Tensor:
    """Merge tree allow/block into the draft×draft corner of a 4D causal mask."""
    tl = tree_mask_block.size(-1)
    if causal is None:
        causal = torch.zeros(
            1,
            1,
            seq_len,
            seq_len,
            device=tree_mask_block.device,
            dtype=min_dtype,
        )
    block = tree_mask_block.to(device=causal.device, dtype=causal.dtype)
    causal[:, :, -tl:, -tl:][block == 0] = torch.finfo(causal.dtype).min
    return causal


@contextmanager
def tree_verify_attention_context(
    decoder: nn.Module,
    tree_mask_block: torch.Tensor,
):
    """
    patch decoder的_update_causal_mask / _prepare_decoder_attention_mask，在因果 mask 的 [-L:, -L:] 子块上叠 树掩码
    Install tree_mask_block on decoder for the duration of one tree verify forward.
    """
    config = getattr(decoder, "config", None)
    saved_impl: Optional[str] = None
    if config is not None and hasattr(config, "_attn_implementation"):
        saved_impl = config._attn_implementation
        config._attn_implementation = "eager"

    decoder._llmqrt_tree_mask_block = tree_mask_block
    originals: List[Tuple[str, Callable[..., Any]]] = []

    if hasattr(decoder, "_update_causal_mask"):
        orig = decoder._update_causal_mask

        def _patched_update(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Any,
            output_attentions: bool,
            **kwargs: Any,
        ):
            causal = orig(
                attention_mask,
                input_tensor,
                cache_position,
                past_key_values,
                output_attentions,
                **kwargs,
            )
            tm = getattr(self, "_llmqrt_tree_mask_block", None)
            if tm is not None:
                min_dtype = torch.finfo(input_tensor.dtype).min
                causal = _apply_tree_block_to_causal(
                    causal,
                    tm,
                    seq_len=input_tensor.shape[1],
                    min_dtype=min_dtype,
                )
            return causal

        decoder._update_causal_mask = types.MethodType(_patched_update, decoder)
        originals.append(("_update_causal_mask", orig))

    if hasattr(decoder, "_prepare_decoder_attention_mask"):
        orig_prep = decoder._prepare_decoder_attention_mask

        def _patched_prepare(
            self,
            attention_mask: torch.Tensor,
            input_shape: torch.Size,
            inputs_embeds: torch.Tensor,
            past_key_values_length: int,
        ):
            combined = orig_prep(attention_mask, input_shape, inputs_embeds, past_key_values_length)
            tm = getattr(self, "_llmqrt_tree_mask_block", None)
            if tm is not None and combined is not None:
                tl = tm.size(-1)
                combined[:, :, -tl:, -tl:][
                    tm.to(device=combined.device, dtype=combined.dtype) == 0
                ] = combined.min()
            return combined

        decoder._prepare_decoder_attention_mask = types.MethodType(_patched_prepare, decoder)
        originals.append(("_prepare_decoder_attention_mask", orig_prep))

    try:
        yield
    finally:
        for name, fn in reversed(originals):
            setattr(decoder, name, fn)
        if hasattr(decoder, "_llmqrt_tree_mask_block"):
            del decoder._llmqrt_tree_mask_block
        if config is not None and saved_impl is not None:
            config._attn_implementation = saved_impl

# 根据topk，选择不同的verify attn backend，可以是flash attn(topk=1)，triton attn(topk>1)，eager(回退)
@torch.inference_mode()
def full_model_tree_verify_logits(
    target_m: PreTrainedModel,
    layout: TreeVerifyLayout,
    past_key_values: Any,
    past_len: int,
    *,
    verify_attn_backend: Optional[str] = None,
    topk: int = 1,
) -> torch.Tensor:
    """
    One full-target forward over L draft nodes with tree-masked self-attention.

    Returns
    -------
    logits : [1, L, vocab]
    """
    device = next(target_m.parameters()).device
    l_n = len(layout)
    if l_n == 0:
        v = getattr(target_m.config, "vocab_size", 0)
        return torch.zeros(1, 0, v, device=device)

    draft_ids = torch.tensor([layout.token_ids], device=device, dtype=torch.long)
    # position ids表示某个token id在树的深度，同一层的token id则position id一样
    position_ids = layout_tree_position_ids(layout, past_len).unsqueeze(0).to(device)
    tree_block = build_tree_allow_mask_4d(layout, device=device)
    decoder = _decoder_stack(target_m)
    backend = resolve_verify_attn_backend(
        verify_attn_backend, topk=topk, layout_len=l_n
    )
    fwd_kw = dict(
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=False,
        attention_mask=torch.ones(1, l_n, device=device, dtype=torch.long),
    )

    if backend == "eager":
        with tree_verify_attention_context(decoder, tree_block):
            out = target_m(draft_ids, **fwd_kw)
    elif backend == "flash_attn":
        with tree_verify_attention_context_flash(decoder):
            out = target_m(draft_ids, **fwd_kw)
    elif backend == "triton_tree":
        # 构建metadata，包括custommask
        meta = build_triton_tree_verify_metadata(layout, past_len, device=device)
        # patch attn为tree attention，使用extend attention fwd，然后在target fwd
        with tree_verify_triton_attention_context(decoder, meta):
            out = target_m(draft_ids, **fwd_kw)
    else:
        raise RuntimeError(f"unknown verify_attn_backend: {backend}")
    return out.logits


def select_append_from_full_tree_logits(
    logits: torch.Tensor,
    parent: List[int],
    token_ids: List[int],
    leaves: List[int],
) -> Tuple[List[int], int, int]:
    """
    Pick best leaf by greedy match length.

    Returns (append_token_list, best_match_len, best_leaf_index).
    """
    best_m = -1
    best_li = 10**9
    best_next = 0
    best_leaf = 0
    # 对draft tree里的每个draft path对应得draft token ids都去和verify logits比对，看谁match len更长，选最长的那个path，一样长得话选logits大得那个
    for li, leaf in enumerate(leaves):
        mlen, nxt = verify_leaf_path_against_parallel_logits(
            logits, parent, token_ids, leaf
        )
        if mlen > best_m or (mlen == best_m and li < best_li):
            best_m, best_li, best_next, best_leaf = mlen, li, nxt, leaf

    path = path_node_indices_root_to_leaf(parent, int(best_leaf))
    path_toks = [token_ids[i] for i in path]
    if best_m == len(path_toks):
        append_list = path_toks + [best_next]
    else:
        append_list = path_toks[:best_m] + [best_next]
    return append_list, best_m, best_leaf


# -----------------------------------------------------------------------------
# 复用前缀 KV cache 的extendverify（不重算前缀，只前向 L 个树节点）。
# HF sdpa + 4D 树 cross mask，或 triton extend kernel；前缀 KV 来自 runner 的 past，
# 在 cache 上临时 extend 后 crop 还原（不污染 past）。
# -----------------------------------------------------------------------------
def _extend_then_crop(target_m, tree_tokens, past_key_values, *, position_ids, attention_mask):
    """在 runner 的 past 上临时 extend（追加 L 个树节点 KV）跑一次 forward，取完 logits 后 crop 还原。

    比深拷贝整份 cache 便宜得多（只多存/截掉 L 个位置的 KV）；KV-replay 用全长 mask 后 past 不被污染。
    """
    before = int(past_key_values.get_seq_length())
    out = target_m(
        tree_tokens,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
        attention_mask=attention_mask,
    )
    logits0 = out.logits[0]
    past_key_values.crop(before)  # 去掉本次 verify 追加的树 KV，还原成 out_ids 的 KV
    return logits0


def _select_append_root_child(
    node_logits: torch.Tensor,
    recovery_logits: torch.Tensor,
    parent: List[int],
    token_ids: List[int],
    leaves: List[int],
) -> Tuple[List[int], int, int, List[int]]:
    """基于node logits对应的token和每条叶子路径上的token串行比对，作出接受与否的判断，选 match 最长的一条。"""
    # 张量化贪心接受：一次性 argmax + 单次 D2H 回读，替代逐 (leaf, k) 的 .item()/.argmax 标量同步。
    # 语义不变：root 用 recovery_logits 的贪心预测，其余节点用其父 node_logits 的贪心预测。
    node_pred = node_logits.argmax(dim=-1)             # [L]，节点对其子的贪心预测
    root_pred = recovery_logits.argmax(dim=-1).view(1)  # [1]，root 的贪心预测
    preds = torch.cat([root_pred, node_pred]).tolist()  # 单次 D2H 同步（整段一起读回）
    root_pred_int = preds[0]
    node_pred_list = preds[1:]

    best_m = -1
    best_li = 10**9
    best_next = 0
    best_leaf = 0
    # 对每条叶子路径串行作接受与否的判断，选 match 最长的一条（平局取 leaf 下标更小者）
    for li, leaf in enumerate(leaves):
        idxs = path_node_indices_root_to_leaf(parent, leaf)  # 根→叶 的所有节点下标
        toks = [token_ids[i] for i in idxs]                  # draft 在该路径上提案的 token 序列
        m = 0       # 已连续接受的长度
        corr = None # 首个不匹配处 target 的贪心修正 token
        for k in range(len(idxs)):
            # parent→child 接受：第 k 个 draft token 应由其父的 logits 预测
            # k=0 无父节点 → 用 recovery_logits（前缀末位对下一 token的预测）
            # k>0 → 用 parent=idxs[k-1] 处 node_logits 的 argmax
            pred = root_pred_int if k == 0 else node_pred_list[idxs[k - 1]]
            if pred == toks[k]:
                m += 1
            else:
                corr = pred  # draft 猜错：本轮只采纳到 m，下一位用 target 修正
                break
        # 路径全中：bonus = 叶节点 logits 的 argmax（再多接受 1 个 token）
        # 路径中途失败：下一位 = corr
        if m == len(idxs):
            nxt = node_pred_list[idxs[-1]]
        else:
            nxt = int(corr)
        if m > best_m or (m == best_m and li < best_li):
            best_m, best_li, best_next, best_leaf = m, li, nxt, leaf
    path_idx = path_node_indices_root_to_leaf(parent, int(best_leaf))
    path_toks = [token_ids[i] for i in path_idx]
    if best_m == len(path_toks):
        append_list = path_toks + [best_next]
    else:
        append_list = path_toks[:best_m] + [best_next]
    return append_list, best_m, int(best_leaf), path_toks


@torch.inference_mode()
def full_tree_verify_extend(
    target_m: PreTrainedModel,
    layout: TreeVerifyLayout,
    past_key_values: Any,
    past_len: int,
    recovery_logits: torch.Tensor,
    *,
    parent: List[int],
    token_ids: List[int],
    leaves: List[int],
    device: torch.device,
) -> Tuple[List[int], int, int, List[int]]:
    """复用前缀 KV 的 extend 树掩码 verify（HF sdpa + 4D 树 cross mask）：只前向 L 个树节点。

    在 past 上临时 extend 后 crop 还原（不污染 runner 的 past）；root 接受用 recovery_logits。
    """
    L = len(layout)
    dtype = next(target_m.parameters()).dtype
    tree_tokens = torch.tensor([layout.token_ids], device=device, dtype=torch.long)
    cross = build_tree_cross_attn_bias_with_prefix(
        layout, past_len, device=device, dtype=dtype
    ).view(1, 1, L, past_len + L)
    pos = layout_tree_position_ids(layout, past_len).unsqueeze(0).to(device)
    # 把use_cache=False=>True，使得target_m前向verify拿到各个logits
    # 之前用use_cache=False的原因: 怕被拒绝tokens kv污染kv cache，所以牺牲性能换取这个干净
    # extend_then_crop优化了这一点，使得不用重算prefix kv，只是计算后把拒绝tokens的kv截断掉就好
    node_logits = _extend_then_crop(
        target_m, tree_tokens, past_key_values, position_ids=pos, attention_mask=cross
    )  # [L, V]
    return _select_append_root_child(node_logits, recovery_logits, parent, token_ids, leaves)


@torch.inference_mode()
def full_tree_verify_triton(
    target_m: PreTrainedModel,
    layout: TreeVerifyLayout,
    past_key_values: Any,
    past_len: int,
    recovery_logits: torch.Tensor,
    *,
    parent: List[int],
    token_ids: List[int],
    leaves: List[int],
    device: torch.device,
) -> Tuple[List[int], int, int, List[int]]:
    """extend verify 的 triton 版：复用前缀 KV，用 extend_attention_fwd + 树 custom_mask。

    数值与 full_tree_verify_extend 等价；同样在 cache 上临时 extend 后 crop、root 接受用 recovery_logits。
    """
    L = len(layout)
    tree_tokens = torch.tensor([layout.token_ids], device=device, dtype=torch.long)
    pos = layout_tree_position_ids(layout, past_len).unsqueeze(0).to(device)
    meta = build_triton_tree_verify_metadata(layout, past_len, device=device)
    decoder = _decoder_stack(target_m)
    with tree_verify_triton_attention_context(decoder, meta):
        # 把use_cache=False=>True，使得有着triton extend attn patch的target_m前向verify拿到各个logits
        # 之前用use_cache=False的原因: 怕被拒绝tokens kv污染kv cache，所以牺牲性能换取这个干净
        # extend_then_crop优化了这一点，使得不用重算prefix kv，只是计算后把拒绝tokens的kv截断掉就好
        node_logits = _extend_then_crop(
            target_m, tree_tokens, past_key_values, position_ids=pos,
            attention_mask=torch.ones(1, L, device=device, dtype=torch.long),
        )  # [L, V]
    return _select_append_root_child(node_logits, recovery_logits, parent, token_ids, leaves)
