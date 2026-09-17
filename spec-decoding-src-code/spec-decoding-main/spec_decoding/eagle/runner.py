# SPDX-License-Identifier: Apache-2.0
"""
外层调度：把hidden 条件 draft + 树 + verify串成一步 / 多步生成

完整一轮投机在概念上永远是：

  ① Target 跑当前前缀（可带 KV cache）→ 取 末 token 的 hidden  
  ② Draft 在该 hidden 条件下 topk 展开成树  
  ③ Verify：要么 树注意力 一次算清（需自接模型 + tree_verify_mask），
    要么如本文件 逐叶子路径 用 HF causal forward 与 greedy 对齐（实现简单）

本模块与 eagle_speculative 无互相 import。

--------------------------------------------------------------------
当前实现细节（读代码时对照）
--------------------------------------------------------------------

verify_mode=reference_paths：对每个 叶子 路径调用 _verify_path_clean，
选 接受长度最长 的路径（同长取更前叶子）；再拼 append 并用 target
重放一次以对齐 past_key_values（与链式投机里接受 + bonus同精神）。

verify_mode=full_model_tree：tree_verify_full 对 整段 target 一次
forward，各层 attention 注入树掩码（对齐官方 EAGLE verify）。

generate() 支持 full_model_tree、reference_paths。
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .config import Eagle1Config
from .hidden import forward_target_last_hidden
from .tree_draft import (
    DraftTopkFn,
    TreeDraftResult,
    expand_draft_tree,
    tree_draft_to_verify_layout,
)
from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
    build_tree_sdpa_attn_bias,
)
from .tree_verify_attn import triton_extend_available
from .tree_verify_full import (
    full_model_tree_verify_logits,
    full_tree_verify_extend,
    full_tree_verify_triton,
    select_append_from_full_tree_logits,
)


def _unwrap(m: Union[PreTrainedModel, nn.Module]) -> PreTrainedModel:
    if isinstance(m, PreTrainedModel):
        return m
    inner = getattr(m, "model", None)
    if inner is not None and isinstance(inner, PreTrainedModel):
        return inner
    raise TypeError("target must be HF PreTrainedModel or BaseModelForCausalLM")


def _leaf_indices(parent: List[int]) -> List[int]:
    """返回树里所有叶子节点的下标（没有任何子节点的节点）。"""
    n = len(parent)
    has_ch = [False] * n
    for i in range(n):
        p = parent[i]
        if p >= 0:
            has_ch[p] = True  # 标记 p 有孩子
    return [i for i in range(n) if not has_ch[i]]


def _path_tokens_to_leaf(parent: List[int], tok: List[int], leaf: int) -> List[int]:
    """从某叶子沿 parent 指针回溯到根，返回 根→叶 顺序的 token 列表。"""
    out: List[int] = []
    k = leaf
    while k >= 0:
        out.append(tok[k])
        k = parent[k]
    out.reverse()  # 回溯是叶→根，反转成根→叶
    return out


# -----------------------------------------------------------------------------
# Verify（参考实现）：用 一次干净整段前向（无共享增量 KV）核对一条 draft 路径。
# 位置对齐：logits[P-1+j] 是 path[j] 所在位置的 target greedy 预测：
#   - path[0] 的真值 = logits[P-1]（吃完真实前缀后的预测，而非吃完 path[0] 之后）；
#   - 全中时 bonus = logits[P-1+len(path)]。
# 用 use_cache=False 避免 HF DynamicCache 原地修改（把被拒绝的续写留进缓存）。
# -----------------------------------------------------------------------------
@torch.inference_mode()
def _verify_path_clean(
    target_m: PreTrainedModel,
    full_prefix_ids: torch.LongTensor,
    path_tokens: List[int],
    device: torch.device,
) -> Tuple[int, int]:
    """对 prefix + path 做一次干净前向，返回 (match_len, next_token)（lossless）。"""
    if not path_tokens:
        out = target_m(
            full_prefix_ids,
            attention_mask=torch.ones_like(full_prefix_ids, device=device),
            use_cache=False,
        )
        return 0, int(out.logits[0, -1, :].argmax(dim=-1).item())

    P = int(full_prefix_ids.shape[1])
    full = torch.cat(
        [full_prefix_ids, torch.tensor([path_tokens], device=device, dtype=torch.long)],
        dim=1,
    )
    logits = target_m(
        full,
        attention_mask=torch.ones_like(full, device=device),
        use_cache=False,
    ).logits[0]
    mlen = 0
    corr: Optional[int] = None
    for j, t in enumerate(path_tokens):
        g = int(logits[P - 1 + j].argmax(dim=-1).item())
        if g == t:
            mlen += 1
        else:
            corr = g
            break
    if mlen == len(path_tokens):
        nxt = int(logits[P - 1 + len(path_tokens)].argmax(dim=-1).item())  # bonus
    else:
        nxt = int(corr)
    return mlen, nxt


# -----------------------------------------------------------------------------
# EAGLE 的编排类：
#   - draft_topk_fn：封装了你的小模型 +target hidden 的接入方式；
#   - cfg：树形状、取哪层 hidden、verify 模式。
# -----------------------------------------------------------------------------
class Eagle1Generator:
    """
    Target hidden 条件 draft（经 draft_topk_fn）+ 静态 topk 树 + 参考 verify。

    draft_topk_fn 可由 draft_model.make_eagle1_draft_topk_fn（轻量 EAGLE1 draft）
    或 make_hidden_residual_draft_topk_fn（外部完整模型 + proj）构造。
    """

    def __init__(
        self,
        target: Union[PreTrainedModel, nn.Module],
        cfg: Eagle1Config,
        draft_topk_fn: DraftTopkFn,
        *,
        refresh_target_each_node: bool = True,
    ) -> None:
        """保存 target、配置与 draft topk 回调。

        refresh_target_each_node=True（默认）：展开树时对每个内部节点重新前向 target 取 hidden
        （接受率更高、但每节点一次 target forward）；超大模型建议 False（只用根 hidden，建树仅跑 draft）。
        """
        self.target = target
        self.target_m = _unwrap(target)
        self.cfg = cfg
        self.draft_topk_fn = draft_topk_fn
        self.refresh_target_each_node = refresh_target_each_node
        # 诊断计数：轮数、采纳 token 数（accept/round = n_accepted_tokens / n_rounds）
        self.n_rounds = 0
        self.n_accepted_tokens = 0

    def _refresh_hidden(self, prefix_ids: torch.LongTensor) -> torch.Tensor:
        h, _, _ = forward_target_last_hidden(
            self.target,
            prefix_ids,
            hidden_layer_index=self.cfg.hidden_layer_index,
            use_cache=False,
        )
        return h

    # ------------------------------------------------------------------
    # 单轮：prefix + past → 建树 →（可选）逐叶 verify → 得到本步要追加的 token 列
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_step_payload(
        self,
        full_prefix_ids: torch.LongTensor,
        past_before: Any,
        last_token_id: torch.LongTensor,
        recovery_logits: Optional[torch.Tensor] = None,
        root_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.LongTensor], TreeVerifyLayout, torch.Tensor, TreeDraftResult]:
        """
        单轮投机：(append_chunk, layout, attn_bias_2d, draft_result)。

        返回的 attn_bias_2d：reference_paths 为 [L,L] 树内掩码；
        full_model_tree 为 [L, past+L] cross 掩码。

        root_hidden：可由 runner 从 prefill/append-replay 的 output_hidden_states 透传，省掉本轮
        对整段前缀的额外 target 前向（大模型加速关键）；None 时退回干净整段前向。
        """
        device = full_prefix_ids.device
        # 根 token 的 target 末 hidden：优先用透传的（避免重算整段前缀），否则干净整段前向。
        if root_hidden is not None:
            h_root = root_hidden
        else:
            h_root, _, _ = forward_target_last_hidden(
                self.target,
                full_prefix_ids,
                hidden_layer_index=self.cfg.hidden_layer_index,
                use_cache=False,
            )

        def refresh(prefix: torch.LongTensor) -> torch.Tensor:
            return self._refresh_hidden(prefix)

        # EAGLE1：静态 top-k 树（大模型上 refresh=None 只用根 hidden，避免每节点一次 target forward）
        _refresh = refresh if self.refresh_target_each_node else None
        draft_res = expand_draft_tree(
            self.cfg,
            h_root,
            last_token_id,
            self.draft_topk_fn,
            target_hidden_refresh=_refresh,
            initial_prefix_ids=full_prefix_ids if self.refresh_target_each_node else None,
        )
        layout = tree_draft_to_verify_layout(draft_res)

        past_len = int(full_prefix_ids.shape[1])
        leaves = _leaf_indices(draft_res.parent)
        # 没有叶子，说明只有root token，没有path tokens
        if not leaves:
            _, nxt = _verify_path_clean(self.target_m, full_prefix_ids, [], device)
            bias = build_tree_sdpa_attn_bias(layout, device=device, dtype=torch.float32)
            return torch.tensor([[nxt]], device=device, dtype=torch.long), layout, bias, draft_res
        if self.cfg.verify_mode == "full_model_tree":
            if recovery_logits is None:
                raise ValueError("full_model_tree verify requires recovery_logits")
            # 复用前缀 KV 的 extend verify（只前向 L 个树节点，不重算前缀）；正确 parent→child 接受。
            # backend=triton_tree（或 auto+topk>1+triton 可用）；否则 HF sdpa。
            backend = self.cfg.verify_attn_backend
            use_triton = backend == "triton_tree" or (
                backend == "auto" and self.cfg.topk > 1 and triton_extend_available()
            )
            verify_fn = full_tree_verify_triton if use_triton else full_tree_verify_extend
            append_list, _, _, _ = verify_fn(
                self.target_m,
                layout,
                past_before,
                past_len,
                recovery_logits,
                parent=draft_res.parent,
                token_ids=draft_res.token_ids,
                leaves=leaves,
                device=device,
            )
            append = torch.tensor([append_list], device=device, dtype=torch.long)
            cross_bias = build_tree_cross_attn_bias_with_prefix(
                layout, past_len, device=device, dtype=torch.float32
            )
            return append, layout, cross_bias, draft_res

        if self.cfg.verify_mode != "reference_paths":
            raise ValueError(f"unhandled verify_mode={self.cfg.verify_mode!r}")
        
        # 为reference_paths时走到下面代码
        # 逐叶子路径各做一次 target 前向（_verify_path_clean），有几个叶子就跑几次；不用树注意力
        bias = build_tree_sdpa_attn_bias(layout, device=device, dtype=torch.float32)
        best_m = -1
        best_li = 10**9
        best_next = 0
        best_leaf = 0
        for li, leaf in enumerate(leaves):
            path = _path_tokens_to_leaf(draft_res.parent, draft_res.token_ids, leaf)
            mlen, nxt = _verify_path_clean(self.target_m, full_prefix_ids, path, device)
            if mlen > best_m or (mlen == best_m and li < best_li):
                best_m, best_li, best_next, best_leaf = mlen, li, nxt, leaf

        path = _path_tokens_to_leaf(draft_res.parent, draft_res.token_ids, int(best_leaf))
        if best_m == len(path):
            append_list = path + [best_next]
        else:
            append_list = path[:best_m] + [best_next]

        append = torch.tensor([append_list], device=device, dtype=torch.long)
        return append, layout, bias, draft_res

    # ------------------------------------------------------------------
    # 多轮自回归：每轮 = generate_step_payload + 用 target 重放 append 以对齐 KV。
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """多轮外层循环；支持 full_model_tree、reference_paths。"""
        if self.cfg.verify_mode not in (
            "full_model_tree",
            "reference_paths",
        ):
            raise ValueError(
                "generate() requires verify_mode in full_model_tree, reference_paths"
            )

        device = input_ids.device
        out_ids = input_ids.clone()
        new_tok = 0
        # Prefill：建立 target KV（past）
        att = torch.ones_like(out_ids, device=device)
        # 拿到第一个token
        out0 = self.target_m(
            out_ids, attention_mask=att, use_cache=True, output_hidden_states=True
        )
        #         out0
        # ├── logits
        # │   └── shape = [1, 2, vocab_size]
        # │
        # ├── past_key_values
        # │   └── 保存 How、can 在所有 Transformer 层的 K/V
        # │
        # └── hidden_states
        #     ├── hidden_states[0]   Embedding 输出
        #     ├── hidden_states[1]   第0层输出
        #     ├── hidden_states[2]   第1层输出
        #     ├── ...
        #     └── hidden_states[L]   最终层输出
        #         out0.logits[0, 0]
        #     看完 "How"
        #     预测下一个 token

        # out0.logits[0, 1]
        #     看完 "How can"
        #     预测下一个 token

        # 拿到过去的prompt的kvcache
        past = out0.past_key_values
        # out_ids 之后那一位的 target logits（extend verify 的 root 接受参照）+ 末 token 的 hidden
        # （draft 树根条件）。从同一次前向取出，省掉每轮额外的整段前缀 forward。
        # P(next_token | "How can")
        recovery_logits = out0.logits[0, -1]
        root_hidden = out0.hidden_states[self.cfg.hidden_layer_index][:, -1, :]

        while new_tok < max_new_tokens:
            if eos_token_id is not None and out_ids[0, -1].item() == eos_token_id:
                break  # 已到 EOS，停止
            last_t = out_ids[:, -1:]
            # 一轮：建树 + verify，得到本轮要采纳的 token 段 append，仅仅是draft生成的token路径
            append, _, _, _ = self.generate_step_payload(
                out_ids, past, last_t, recovery_logits, root_hidden
            )
            # 把 append 重放进 target KV。注意：带 past 时 attention_mask 必须覆盖 [past + append]
            # 全长，否则 append 不会 attend 到前缀 → out_sync.logits（recovery_logits）错。
            # （extend verify 在 past 的副本上跑、不污染 past；故 past 始终 = out_ids 的 KV。）
            full_attn = torch.ones(
                (1, int(past.get_seq_length()) + int(append.shape[1])),
                device=device,
                dtype=torch.long,
            )
            out_sync = self.target_m(
                append,
                past_key_values=past,
                use_cache=True,
                attention_mask=full_attn,
                output_hidden_states=True,
            )
            past = out_sync.past_key_values
            recovery_logits = out_sync.logits[0, -1]
            root_hidden = out_sync.hidden_states[self.cfg.hidden_layer_index][:, -1, :]
            out_ids = torch.cat([out_ids, append], dim=1)
            new_tok += int(append.numel())
            self.n_rounds += 1
            self.n_accepted_tokens += int(append.numel())
            if eos_token_id is not None and (append == eos_token_id).any():
                break
        return out_ids


# -----------------------------------------------------------------------------
# 另一种常见的 draft 输入构造：token 的 embedding 与 target hidden 作为输入经过linear，
# 其输出作为该步 draft Transformer 的 inputs_embeds，再取末位置
# logits 做 top-k。
# draft_topk_fn：get_input_embeddings(token) + proj(hidden) → draft_model(inputs_embeds=...) → topk
# -----------------------------------------------------------------------------
def make_hidden_residual_draft_topk_fn(
    draft_model: Union[PreTrainedModel, nn.Module],
    proj: torch.nn.Linear,
    topk: int,
) -> DraftTopkFn:
    """
    工厂：token_emb + proj(target_hidden) 作为 draft 的输入嵌入，再经整模前向
    得到 logits 的 top-k（体现在 target hidden 上建 draft）。

    proj：in_features = target_hidden_dim，out_features = embedding_dim。
    """

    dm = _unwrap(draft_model)

    def fn(hidden: torch.Tensor, parent_token_id: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs_embeds = W_e[token] + W_p * h_target ；h 来自 expand_draft_tree 侧传入
        emb = dm.get_input_embeddings()(parent_token_id)
        if emb.shape[-1] != proj.out_features:
            raise ValueError(
                f"proj.out_features={proj.out_features} must match embedding dim={emb.shape[-1]}"
            )
        x = emb + proj(hidden).unsqueeze(1)
        b, s, _ = x.shape
        att = torch.ones(b, s, device=x.device, dtype=torch.long)
        out = dm(inputs_embeds=x, attention_mask=att, use_cache=False)
        logits = out.logits[:, -1, :]
        k = min(topk, logits.shape[-1])
        log_probs = torch.log_softmax(logits, dim=-1)
        lp, idx = torch.topk(log_probs, k=k, dim=-1)
        return idx, lp

    return fn
