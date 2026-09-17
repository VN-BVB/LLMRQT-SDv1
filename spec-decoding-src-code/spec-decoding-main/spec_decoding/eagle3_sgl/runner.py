# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3 外层调度：target 多层 act 条件 draft + 树 + verify → 多步生成。

典型调用栈（generate() 一轮 decode）：

    generate()
      ├─ target prefill → past, recovery_logits, acts_all
      ├─ draft_model.prefill() → draft_state（AR 路径）
      └─ loop:
           topk=1 fast path → pending + draft 链一次 target forward，并保留 accepted KV
           否则 generate_step_payload() → tree verify → target replay append
           draft_model.extend_tokens() → 更新 draft_state（AR 路径）

verify_mode=full_model_tree（默认）：extend verify，复用前缀 KV，树掩码一次前向 L 节点。
verify_mode=reference_paths：逐叶干净前向，实现简单、便于对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .config import Eagle3SglConfig
from .hidden import (
    forward_target_eagle_acts,
    forward_target_selected_eagle_acts,
)
from .tree_draft import (
    DraftTopkFn,
    TreeDraftResult,
    expand_draft_tree,
    expand_draft_tree_ar,
    expand_draft_tree_topk,
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


def _num_hidden_layers(m: Union[PreTrainedModel, nn.Module]) -> int:
    """从 target config 推断 decoder 层数（兼容 num_hidden_layers / n_layer）。"""
    cfg = getattr(_unwrap(m), "config", None)
    for attr in ("num_hidden_layers", "n_layer"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    raise ValueError("cannot infer num_hidden_layers from target config")


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
# Verify（参考实现， 串行verify）：用 一次干净整段前向（无共享增量 KV）核对一条 draft 路径。
# 位置对齐：logits[P-1+j] 是 path[j] 所在位置的 target greedy 预测：
#   - path[0] 的真值 = logits[P-1]；
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
    """对 prefix + path 做一次干净前向，返回 (match_len, next_token)。

    Parameters
    ----------
    target_m
        完整 HF CausalLM（带 lm_head）。
    full_prefix_ids
        当前已接受前缀 [1, P]（prompt + 此前各轮 append）。
    path_tokens
        某条 draft 叶路径上的 token 列表（根→叶）；空列表表示无前向 path，只取 greedy 下一 token。
    device
        计算设备。

    Returns
    -------
    match_len
        path 上连续被 target greedy 接受的长度。
    next_token
        全中时 = bonus（logits[P-1+len(path)] 的 argmax）；否则 = 首个不匹配处 target 修正 token。
    """
    if not path_tokens:
        out = target_m(
            full_prefix_ids,
            attention_mask=torch.ones_like(full_prefix_ids, device=device),
            use_cache=False,
        )
        return 0, int(out.logits[0, -1, :].argmax(dim=-1).item())

    P = int(full_prefix_ids.shape[1])
    # 整段 [prefix | path] 一次 forward；不写入 runner 的 past（use_cache=False）
    full = torch.cat(
        [full_prefix_ids, torch.tensor([path_tokens], device=device, dtype=torch.long)],
        dim=1,
    )
    logits = target_m(
        full,
        attention_mask=torch.ones_like(full, device=device),
        use_cache=False,
    ).logits[0]
    # logits[P-1+j] = 在已见 prefix + path[:j]条件下对 path[j] 的预测
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


@dataclass
class _DraftKvState:
    """AR draft 增量状态；generate() 每轮 extend 后更新，供 expand_draft_tree_ar 复用。"""

    kv: Any
    """draft midlayer 的 (k, v)；序列长度 = prefix_len - 1（EAGLE shift）。"""
    prefix_len: int
    """当前 target 已接受前缀长度 P；与 out_ids.shape[1] 对齐。"""
    root_logits: torch.Tensor
    """prefill/extend 末步 draft logits [1, V]；根层 topk 来源。"""
    root_hidden: torch.Tensor
    """prefill/extend 末步 draft hidden [1, H]；各 BFS 分支初始 cond（feature 递归起点）。"""


# -----------------------------------------------------------------------------
# eagle3投机编排：draft 树 + target verify + 每轮投机后draft状态更新
# -----------------------------------------------------------------------------
class Eagle3SglGenerator:
    """
    EAGLE3 多层 act 条件 draft（经 draft_topk_fn）+ 静态 topk 树 + 树掩码 verify。

    draft 的条件特征是 target 的多层 hidden 拼接（eagle_layers），
    由 hidden.forward_target_eagle_acts 取出；

    draft_topk_fn 可由 draft_model.make_eagle3_sgl_draft_topk_fn（轻量 EAGLE3 draft）
    """

    def __init__(
        self,
        target: Union[PreTrainedModel, nn.Module],
        cfg: Eagle3SglConfig,
        draft_topk_fn: Optional[DraftTopkFn] = None,
        *,
        draft_model=None,
        autoregressive_draft: bool = True,
        refresh_target_each_node: bool = True,
    ) -> None:
        """保存配置、draft 回调，并解析多层特征下标 eagle_layers。

        autoregressive_draft=True（默认）：draft hidden feature 递归（接受率更高）；

        autoregressive_draft=False：用 draft_topk_fn 单 token 建树；refresh_target_each_node
        控制是否逐节点刷新 target act。
        """
        self.target = target
        self.target_m = _unwrap(target)
        self.cfg = cfg
        self.draft_topk_fn = draft_topk_fn
        self.draft_model = draft_model
        self.autoregressive_draft = autoregressive_draft
        self.refresh_target_each_node = refresh_target_each_node
        if autoregressive_draft and draft_model is None:
            raise ValueError("autoregressive_draft=True requires draft_model=...")
        if not autoregressive_draft and draft_topk_fn is None:
            raise ValueError("autoregressive_draft=False requires draft_topk_fn")
        # EAGLE3 多层特征下标（draft 条件来自这些层的末 token hidden 拼接）
        self.eagle_layers: List[int] = cfg.resolve_eagle_layers(_num_hidden_layers(target))
        # 诊断计数：投机采样的轮数、采纳 的token 数（accept/round = n_accepted_tokens / n_rounds）
        self.n_rounds = 0
        self.n_accepted_tokens = 0
        self.n_matched_draft_tokens = 0
        # 单独统计真正命中的 draft token；n_accepted_tokens 为兼容旧日志仍表示总输出 token。
        self.n_proposed_draft_tokens = 0
        # proposed 作为接受率分母，避免把“每轮输出长度”和“draft 接受率”混为一谈。
        self._last_matched_draft_tokens = 0
        self._last_proposed_draft_tokens = 0

    def _refresh_hidden(self, prefix_ids: torch.LongTensor) -> torch.Tensor:
        """非 AR 建树的根条件：前缀末 token 的多层 act [1, n*H]。

        Parameters
        ----------
        prefix_ids
            当前已接受前缀 [1, P]。

        用 use_cache=False 干净前向，不污染 generate() 维护的 target past。
        """
        acts, _, _ = forward_target_eagle_acts(
            self.target,
            prefix_ids,
            self.eagle_layers,
            use_cache=False,
        )
        return acts

    def _refresh_all(self, prefix_ids: torch.LongTensor) -> torch.Tensor:
        """各 token 位置的 target 多层 act [1, P, n*H]，供eagle3 draft prefill"""
        acts, _, _ = forward_target_eagle_acts(
            self.target,
            prefix_ids,
            self.eagle_layers,
            use_cache=False,
            last_token_only=False,
        )
        return acts

    def _build_tree(
        self,
        prefix_ids: torch.LongTensor,
        last_token_id: torch.LongTensor,
        root_acts: Optional[torch.Tensor] = None,
        acts_all: Optional[torch.Tensor] = None,
        draft_state: Optional[_DraftKvState] = None,
    ) -> TreeDraftResult:
        """根据 autoregressive_draft 选择 AR 建树（eagle3）或 draft_topk_fn（eagle1/eagle2） 建树。

        Parameters
        ----------
        prefix_ids
            当前已接受前缀 [1, P]。
        last_token_id
            前缀末 token [1, 1]；非 AR 路径作为 draft_topk_fn 根层父 token。
        root_acts
            末 token 多层 act [1, n*H]；非 AR 建树根条件（透传可省 target 重算）。
        acts_all
            全前缀各位置 act [1, P, n*H]；AR 路径无 draft_state 时 prefill fallback。
        draft_state
            增量 draft KV + 末步输出；AR decode 阶段复用，避免每轮 prefill 整段 prefix。

        Returns
        -------
        TreeDraftResult
            摊平后的 draft 树（token_ids / parent / node_depth 等）。
        """
        if self.autoregressive_draft:
            # AR：decode 轮用上一轮 extend 后的 root_logits/root_hidden/kv
            if draft_state is not None:
                if draft_state.prefix_len != int(prefix_ids.shape[1]):
                    raise ValueError(
                        f"draft_state.prefix_len {draft_state.prefix_len} != "
                        f"prefix len {prefix_ids.shape[1]}"
                    )
                root_logits = draft_state.root_logits
                root_hidden = draft_state.root_hidden
                root_kv = draft_state.kv
            else:
                if acts_all is None:
                    acts_all = self._refresh_all(prefix_ids)
                elif acts_all.shape[1] != prefix_ids.shape[1]:
                    raise ValueError(
                        f"acts_all seq len {acts_all.shape[1]} != prefix len {prefix_ids.shape[1]}"
                    )
                root_logits, root_hidden, root_kv = self.draft_model.prefill(
                    prefix_ids, acts_all
                )
            return expand_draft_tree_ar(
                self.cfg,
                self.draft_model,
                root_logits,
                root_hidden,
                root_kv,
                int(prefix_ids.shape[1]) - 1,
                device=prefix_ids.device,
            )
        # 非 AR：每步只调 draft_topk_fn(parent_token)；可选每节点重算 target act
        if root_acts is not None:
            h_root = root_acts
        else:
            h_root = self._refresh_hidden(prefix_ids)
        refresh = self._refresh_hidden if self.refresh_target_each_node else None
        expand_fn = (
            expand_draft_tree_topk
            if self.cfg.tree_expand_mode == "cumulative"
            else expand_draft_tree
        )
        return expand_fn(
            self.cfg,
            h_root,
            last_token_id,
            self.draft_topk_fn,
            target_hidden_refresh=refresh,
            initial_prefix_ids=prefix_ids if self.refresh_target_each_node else None,
        )

    # ------------------------------------------------------------------
    # 单轮：prefix + past → 建树 → verify → 得到本步要追加的 tokens
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_step_payload(
        self,
        full_prefix_ids: torch.LongTensor,
        past_before: Any,
        last_token_id: torch.LongTensor,
        recovery_logits: Optional[torch.Tensor] = None,
        root_acts: Optional[torch.Tensor] = None,
        acts_all: Optional[torch.Tensor] = None,
        draft_state: Optional[_DraftKvState] = None,
    ) -> Tuple[Optional[torch.LongTensor], TreeVerifyLayout, torch.Tensor, TreeDraftResult]:
        """
        单轮投机返回 (append_chunk, layout, attn_bias_2d, draft_result)。

        Parameters
        ----------
        full_prefix_ids
            当前已接受序列 [1, P]（prompt + 此前各轮 append）。
        past_before
            target 对 full_prefix_ids 的 KV cache；extend verify 在其上临时 extend 后 crop。
        last_token_id
            前缀末 token [1, 1]；非 AR 建树时作为 draft 根层父 token。
        recovery_logits
            前缀末位预测下一 token 的 logits [V]；full_model_tree 必填（根层 parent→child 接受）。
        root_acts
            末 token 多层 act [1, n*H]；非 AR 建树根条件（透传省重算）。
        acts_all
            全前缀各位置 act [1, P, n*H]；AR draft prefill fallback。
        draft_state
            增量维护的 draft KV + root 输出；AR 路径避免每轮 prefill 整段 prefix。

        Returns
        -------
        append_chunk
            本轮采纳 token [1, A]（含 path + bonus 或 corr）。
        layout
            TreeVerifyLayout，供 verify / 调试。
        attn_bias_2d
            reference_paths → [L,L] 树内掩码；
            full_model_tree → [L, past+L] cross 掩码。
        draft_result
            原始 TreeDraftResult。
        """
        device = full_prefix_ids.device
        # ① 建树（AR 或 topk_fn）
        draft_res = self._build_tree(
            full_prefix_ids, last_token_id, root_acts, acts_all, draft_state
        )
        self._last_proposed_draft_tokens = len(draft_res.token_ids)
        # 记录实际送去 verify 的节点数；树模式不能简单用 topk * num_steps 代替。
        layout = tree_draft_to_verify_layout(draft_res)

        past_len = int(full_prefix_ids.shape[1])
        leaves = _leaf_indices(draft_res.parent)
        # 退化：树只有空或无叶节点时，等价于投机解码失败，直接 target生成下一 token
        if not leaves:
            _, nxt = _verify_path_clean(self.target_m, full_prefix_ids, [], device)
            bias = build_tree_sdpa_attn_bias(layout, device=device, dtype=torch.float32)
            self._last_matched_draft_tokens = 0
            return torch.tensor([[nxt]], device=device, dtype=torch.long), layout, bias, draft_res
        if self.cfg.verify_mode == "full_model_tree":
            if recovery_logits is None:
                raise ValueError("full_model_tree verify requires recovery_logits")
            # 复用前缀 KV 的 extend verify（只前向 L 个树节点，不重算前缀）；
            # backend=triton_tree（或 auto+topk>1+triton 可用）→ Triton 树注意力 kernel；否则走HF自带的torch sdpa。
            backend = self.cfg.verify_attn_backend
            use_triton = backend == "triton_tree" or (
                backend == "auto" and self.cfg.topk > 1 and triton_extend_available()
            )
            verify_fn = full_tree_verify_triton if use_triton else full_tree_verify_extend
            append_list, best_m, _, _ = verify_fn(
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
            self._last_matched_draft_tokens = best_m
            # 保存本轮真实命中的 draft 数，避免把必然追加的 target 修正 token 误算成“接受”。
            append = torch.tensor([append_list], device=device, dtype=torch.long)
            cross_bias = build_tree_cross_attn_bias_with_prefix(
                layout, past_len, device=device, dtype=torch.float32
            )
            return append, layout, cross_bias, draft_res

        if self.cfg.verify_mode != "reference_paths":
            raise ValueError(f"unhandled verify_mode={self.cfg.verify_mode!r}")

        # reference_paths：每条叶路径一次完整 causal forward（串行verify），选 match 最长的叶（慢、易对照）
        bias = build_tree_sdpa_attn_bias(layout, device=device, dtype=torch.float32)
        best_m = -1
        best_li = 10**9
        best_next = 0
        best_leaf = 0
        for li, leaf in enumerate(leaves):
            path = _path_tokens_to_leaf(draft_res.parent, draft_res.token_ids, leaf)
            mlen, nxt = _verify_path_clean(self.target_m, full_prefix_ids, path, device)
            # 平局：leaf 下标更小者优先（与 extend verify 的 tie-break 一致）
            if mlen > best_m or (mlen == best_m and li < best_li):
                best_m, best_li, best_next, best_leaf = mlen, li, nxt, leaf

        path = _path_tokens_to_leaf(draft_res.parent, draft_res.token_ids, int(best_leaf))
        # 全中：采纳整条 path + bonus；否则采纳 path[:m] + target 纠正的下一 token
        if best_m == len(path):
            append_list = path + [best_next]
        else:
            append_list = path[:best_m] + [best_next]

        self._last_matched_draft_tokens = best_m
        # reference_paths 也记录真实 draft 命中数，使两个 verify 后端的统计口径一致。
        append = torch.tensor([append_list], device=device, dtype=torch.long)
        return append, layout, bias, draft_res

    @torch.inference_mode()
    def _verify_chain_and_commit(
        self,
        pending_token_id: torch.LongTensor,
        past_before: Any,
        draft_res: TreeDraftResult,
        *,
        max_append_tokens: int,
        eos_token_id: Optional[int],
    ) -> Tuple[torch.LongTensor, torch.Tensor, int]:
        """验证 top-1 draft 链，并直接保留已接受节点的 target KV。

        target KV 故意比 out_ids 少最后一个 pending token。把 pending 与 draft 链放进
        同一次标准 causal forward 后，logits[0] 验证第一个 draft token；这样无需再
        用 target 重放整段 append。该优化只用于 topk=1，分支树仍走原验证路径。
        """
        token_ids = draft_res.token_ids
        expected_parent = [-1] + list(range(max(0, len(token_ids) - 1)))
        if draft_res.parent != expected_parent:
            raise ValueError("fast chain verify requires a single top-1 chain")

        device = pending_token_id.device
        candidate_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
        verify_ids = torch.cat([pending_token_id, candidate_ids], dim=1)
        before = int(past_before.get_seq_length())
        q_len = int(verify_ids.shape[1])
        full_attn = torch.ones(
            (1, before + q_len), device=device, dtype=torch.long
        )
        position_ids = torch.arange(
            before, before + q_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        out, all_shift_acts = forward_target_selected_eagle_acts(
            self.target_m.model,
            self.eagle_layers,
            last_token_only=False,
            input_ids=verify_ids,
            past_key_values=past_before,
            position_ids=position_ids,
            use_cache=True,
            attention_mask=full_attn,
        )
        # 只跑 backbone 并抓三层 act；完整词表 logits 在下面对短链一次批量计算。

        hidden_states = out.last_hidden_state[0]
        lm_head = self.target_m.get_output_embeddings()
        pred_ids = lm_head(hidden_states).argmax(dim=-1).tolist()
        # 短链所有位置合并成一次 GEMM 和一次 D2H；实测快于按接受路径逐位置 GEMV。
        matched = 0
        for idx, token_id in enumerate(token_ids):
            if pred_ids[idx] != token_id:
                break
            matched += 1
        next_token = pred_ids[matched]
        # mismatch 与全中都从同一批 logits 取 correction/bonus，保持贪心接受规则不变。
        append_list = token_ids[:matched] + [next_token]
        append_list = append_list[:max_append_tokens]
        if eos_token_id is not None and eos_token_id in append_list:
            append_list = append_list[: append_list.index(eos_token_id) + 1]
        # 在 token 上限或 EOS 处立刻裁剪，防止一轮 append 越过调用方要求的边界。

        kept_draft = min(matched, len(append_list))
        past_before.crop(before + 1 + kept_draft)
        # 保留 pending 与已接受 draft 的 KV；末尾 target token 留到下一轮一起验证。
        shift_acts = all_shift_acts[:, : len(append_list), :]
        # EAGLE shift 只消费每个 append token 前一位置的 target act，恰好对应此前向的前 A 项。
        append = torch.tensor([append_list], device=device, dtype=torch.long)
        return append, shift_acts, kept_draft

    # ------------------------------------------------------------------
    # 多轮自回归：top-1 合并 verify/replay；多分支保持原 tree verify + replay。
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """多轮外层循环：prefill → 单 target-forward 短链或树验证 → draft extend。

        Parameters
        ----------
        input_ids
            Prompt [1, P0]。
        max_new_tokens
            最多新生成 token 数（按 append 总 token 计，非轮数）。
        eos_token_id
            若给定，前缀末或 append 中出现该 id 则提前停止。

        Returns
        -------
        out_ids
            prompt + 全部已采纳 token [1, P0 + total_new]。
        """
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
        fast_chain = (
            self.autoregressive_draft
            and self.cfg.verify_mode == "full_model_tree"
            and self.cfg.topk == 1
            and self.cfg.verify_attn_backend != "triton_tree"
        )
        # top-1 链可合并 verify/replay；多分支 Triton 树保持原路径。
        # ② Target prefill：KV + 末位 recovery_logits + 全前缀 acts（供 draft / 非 AR）
        att = torch.ones_like(out_ids, device=device)
        out0, acts_all = forward_target_selected_eagle_acts(
            self.target_m,
            self.eagle_layers,
            last_token_only=False,
            input_ids=out_ids,
            attention_mask=att,
            use_cache=True,
            logits_to_keep=1,
        )
        # prefill 只捕获三层辅助状态，并只为末位置计算完整词表 logits。
        past = out0.past_key_values
        recovery_logits = out0.logits[0, -1]
        # 全前缀各位置token的多层act特征（AR draft prefill）；末 token act 供非 AR 路径
        root_acts = acts_all[:, -1, :]
        draft_state: Optional[_DraftKvState] = None
        if self.autoregressive_draft:
            # ③ Draft 与 prefix 对齐的整段 prefill（EAGLE shift 在 draft_model.prefill 内）
            d_logits, d_hidden, d_kv = self.draft_model.prefill(out_ids, acts_all)
            draft_state = _DraftKvState(
                kv=d_kv,
                prefix_len=int(out_ids.shape[1]),
                root_logits=d_logits,
                root_hidden=d_hidden,
            )

        if fast_chain:
            past.crop(int(out_ids.shape[1]) - 1)
            # 快路径让 target KV 落后一个 token，下一轮把该 pending token 与 draft 一起前向。

        while new_tok < max_new_tokens:
            if eos_token_id is not None and out_ids[0, -1].item() == eos_token_id:
                break  # 已到 EOS，停止
            last_t = out_ids[:, -1:]
            # ④ 单轮：建树 → verify → append
            if fast_chain:
                draft_res = self._build_tree(
                    out_ids, last_t, root_acts, acts_all, draft_state
                )
                proposed_draft = len(draft_res.token_ids)
                append, draft_shift_acts, matched_draft = self._verify_chain_and_commit(
                    last_t,
                    past,
                    draft_res,
                    max_append_tokens=max_new_tokens - new_tok,
                    eos_token_id=eos_token_id,
                )
            else:
                append, _, _, _ = self.generate_step_payload(
                    out_ids, past, last_t, recovery_logits, root_acts, acts_all, draft_state
                )
                append = append[:, : max_new_tokens - new_tok]
                if eos_token_id is not None:
                    eos_pos = (append[0] == eos_token_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        append = append[:, : int(eos_pos[0, 0].item()) + 1]
                # 原树路径也在 replay 前裁剪，修复 max_new_tokens 越界及 EOS 后仍追加 token。
                matched_draft = min(
                    self._last_matched_draft_tokens, int(append.shape[1])
                )
                proposed_draft = self._last_proposed_draft_tokens
                # 把 append 重放进 target KV。带 past 时 attention_mask 必须覆盖 [past + append] 全长，
                # 否则 append 不会 attend 到前缀 → out_sync.logits（recovery_logits）错。
                # （extend verify 在 past 上临时 extend 后 crop、不污染 past；故 past 始终 = out_ids 的 KV。）
                full_attn = torch.ones(
                    (1, int(past.get_seq_length()) + int(append.shape[1])),
                    device=device,
                    dtype=torch.long,
                )
                old_root_acts = root_acts
                out_sync, append_acts = forward_target_selected_eagle_acts(
                    self.target_m,
                    self.eagle_layers,
                    last_token_only=False,
                    input_ids=append,
                    past_key_values=past,
                    use_cache=True,
                    attention_mask=full_attn,
                    logits_to_keep=1,
                )
                # replay 只取三层输入 residual，且 recovery 只需要末位置 logits。
                past = out_sync.past_key_values
                recovery_logits = out_sync.logits[0, -1]
                draft_shift_acts = torch.cat(
                    [old_root_acts.unsqueeze(1), append_acts[:, :-1, :]], dim=1
                )
                # 直接拼“旧末位 act + 新 append（除末位）”，避免每轮复制不断增长的 acts_all。
                root_acts = append_acts[:, -1, :]
            if self.autoregressive_draft and draft_state is not None:
                # draft extend：对本轮 append 增量写 KV；act 段与 EAGLE shift 对齐
                p_old = draft_state.prefix_len
                p_new = int(out_ids.shape[1]) + int(append.shape[1])
                d_logits, d_hidden, d_kv = self.draft_model.extend_tokens(
                    append,
                    draft_shift_acts,
                    draft_state.kv,
                    p_old - 1,  # RoPE 从旧 prefix 末位置续
                )
                draft_state = _DraftKvState(
                    kv=d_kv,
                    prefix_len=p_new,
                    root_logits=d_logits,
                    root_hidden=d_hidden,
                )
            out_ids = torch.cat([out_ids, append], dim=1)
            new_tok += int(append.numel())
            self.n_rounds += 1
            self.n_accepted_tokens += int(append.numel())
            self.n_matched_draft_tokens += matched_draft
            # 真实 draft 接受数与总输出数分开累计，便于和 vLLM acceptance rate 公平比较。
            self.n_proposed_draft_tokens += proposed_draft
            # 分母按真正构造的 draft 节点累计，兼容链式和多分支树。
            if eos_token_id is not None and (append == eos_token_id).any():
                break
        return out_ids


# -----------------------------------------------------------------------------
# 注意:以下目前暂时不使用!!另一种常见的 draft 输入构造：token 的 embedding 与 target hidden 作为输入经过linear，
# 其输出作为该步 draft Transformer 的 inputs_embeds，再取末位置logits 做 top-k。
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
