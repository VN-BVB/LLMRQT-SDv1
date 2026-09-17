# SPDX-License-Identifier: Apache-2.0
"""
EAGLE 外层调度：把hidden 条件 draft + 树 + verify串成一步 / 多步生成

完整 EAGLE 一轮投机：

  ① Target 跑当前前缀（可带 KV cache）→ 取 末 token 的 hidden  
  ② Draft 在该 hidden 条件下 topk 展开成树  
  ③ Verify：要么 树注意力 一次算清（需tree_verify_mask），要么逐叶子路径 用 HF causal forward 并与 greedy 对齐

--------------------------------------------------------------------
当前实现细节（读代码时对照）
--------------------------------------------------------------------

verify_mode=reference_paths：对每个 叶子 路径调用 _verify_path_clean，
选 接受长度最长 的路径（同长取更前叶子）；再拼 append 并用 target
重放一次以对齐 past_key_values。

verify_mode=full_model_tree：tree_verify_full 对 整段 target 一次
forward，各层 attention 注入树掩码。

generate() 支持 full_model_tree、reference_paths。
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

# 仅当 EAGLE3_SGL_PROFILE=1 时启用 torch.profiler.record_function，
# 否则返回 nullcontext（零开销）。
_PROFILE = os.environ.get("EAGLE3_SGL_PROFILE", "0") == "1"


def _prof(name: str):
    if _PROFILE:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()

from .config import Eagle3SglConfig
from .hidden import (
    EagleActCapture,
    forward_target_eagle_acts,
    select_eagle_acts_from_hidden_states,
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
# 串行Verify：用 一次干净整段前向（无共享增量 KV）核对一条 draft 路径。
# 位置对齐：logits[P-1+j] 是 path[j] 所在位置的 target 预测：
#   - path[0] 的真值 = logits[P-1]；
#   - 全中时 bonus = logits[P-1+len(path)]。
# 用 use_cache=False 避免 HF DynamicCache 原地修改（把被拒绝的token留进了缓存）。
# -----------------------------------------------------------------------------
@torch.inference_mode()
def _verify_path_clean(
    target_m: PreTrainedModel,
    full_prefix_ids: torch.LongTensor,
    path_tokens: List[int],
    device: torch.device,
) -> Tuple[int, int]:
    """对 prefix + path 做一次干净前向，返回 (match_len, next_token)。"""
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


@dataclass
class _DraftKvState:
    """AR draft KV + 末步 root 输出；prefix_len 为 target 前缀长度 P（draft KV 长 = P-1）。"""

    kv: Any
    prefix_len: int
    root_logits: torch.Tensor
    root_hidden: torch.Tensor


# -----------------------------------------------------------------------------
# EAGLE3 的编排类：
#   - draft_topk_fn：封装了你的小模型 + target hidden 的接入方式；
#   - cfg：树形状、取哪层 hidden、verify 模式。
# -----------------------------------------------------------------------------
class Eagle3SglGenerator:
    """
    EAGLE3 多层 act 条件 draft（经 draft_topk_fn）+ 静态 topk 树 + 树掩码 verify。

    与 EAGLE1 的唯一区别：draft 的条件特征是 target 的多层 hidden 拼接（eagle_layers），
    由 hidden.forward_target_eagle_acts 取出；树verify 完全一致。

    draft_topk_fn 可由 draft_model.make_eagle3_sgl_draft_topk_fn（轻量 EAGLE3 draft）
    或 make_hidden_residual_draft_topk_fn（外部完整模型 + proj，proj.in = n_acts*H）构造。
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

        autoregressive_draft=True（默认）：draft prefill + KV cache + feature 递归建树
        （提高接受率）；需传 draft_model。

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
        # 解析 EAGLE3 多层特征下标（draft 条件来自这些层的末 token hidden 拼接）
        self.eagle_layers: List[int] = cfg.resolve_eagle_layers(_num_hidden_layers(target))
        # 诊断计数：轮数、采纳 token 数（accept/round = n_accepted_tokens / n_rounds）
        self.n_rounds = 0
        self.n_accepted_tokens = 0
        self.n_matched_draft_tokens = 0
        self.n_proposed_draft_tokens = 0
        self._last_matched_draft_tokens = 0
        self._last_proposed_draft_tokens = 0

    def _refresh_hidden(self, prefix_ids: torch.LongTensor) -> torch.Tensor:
        acts, _, _ = forward_target_eagle_acts(
            self.target,
            prefix_ids,
            self.eagle_layers,
            use_cache=False,
        )
        return acts

    def _refresh_all(self, prefix_ids: torch.LongTensor) -> torch.Tensor:
        """各 token 位置的 target 多层 act [1, P, n*H]，供 draft prefill。"""
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
        if self.autoregressive_draft:
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
    # 单轮：prefix + past → 建树 →逐叶 verify → 得到本步要追加的 token 序列
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
        单轮投机：(append_chunk, layout, attn_bias_2d, draft_result)。

        返回的 attn_bias_2d：reference_paths 为 [L,L] 树内掩码；
        full_model_tree 为 [L, past+L] cross 掩码。

        root_acts：末 token 多层 act [B, n*H]。
        acts_all：全前缀各位置多层 act [B, P, n*H]。
        draft_state：增量维护的 draft KV + root 输出，避免每轮 prefill 整段 prefix。
        """
        device = full_prefix_ids.device
        with _prof("stage:draft_build_tree"):
            draft_res = self._build_tree(
                full_prefix_ids, last_token_id, root_acts, acts_all, draft_state
            )
        self._last_proposed_draft_tokens = len(draft_res.token_ids)
        layout = tree_draft_to_verify_layout(draft_res)

        past_len = int(full_prefix_ids.shape[1])
        leaves = _leaf_indices(draft_res.parent)
        # 没有叶子，说明只有root token
        if not leaves:
            _, nxt = _verify_path_clean(self.target_m, full_prefix_ids, [], device)
            bias = build_tree_sdpa_attn_bias(layout, device=device, dtype=torch.float32)
            self._last_matched_draft_tokens = 0
            return torch.tensor([[nxt]], device=device, dtype=torch.long), layout, bias, draft_res
        if self.cfg.verify_mode == "full_model_tree":
            if recovery_logits is None:
                raise ValueError("full_model_tree verify requires recovery_logits")
            # 复用前缀 KV 的 extend verify（只前向 L 个树节点，不重算前缀）；
            # backend=triton_tree；否则 HF sdpa。
            backend = self.cfg.verify_attn_backend
            use_triton = backend == "triton_tree" or (
                backend == "auto" and self.cfg.topk > 1 and triton_extend_available()
            )
            verify_fn = full_tree_verify_triton if use_triton else full_tree_verify_extend
            with _prof("stage:target_verify"):
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
            append = torch.tensor([append_list], device=device, dtype=torch.long)
            cross_bias = build_tree_cross_attn_bias_with_prefix(
                layout, past_len, device=device, dtype=torch.float32
            )
            return append, layout, cross_bias, draft_res

        if self.cfg.verify_mode != "reference_paths":
            raise ValueError(f"unhandled verify_mode={self.cfg.verify_mode!r}")
        
        # 为reference_paths时走到下面代码
        # 逐叶子路径各做一次 target 前向（_verify_path_clean），有几个叶子就跑几次；没有用树注意力
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

        self._last_matched_draft_tokens = best_m
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
        if getattr(self.cfg, "decode_cuda_graph", False):
            if not (self.cfg.verify_mode == "full_model_tree" and self.autoregressive_draft):
                raise ValueError(
                    "decode_cuda_graph requires full_model_tree + autoregressive_draft"
                )
            return self._generate_static_graph(
                input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
            )

        device = input_ids.device
        out_ids = input_ids.clone()
        new_tok = 0
        # Prefill：建立 target KV（past）+ 取末位置 logits / 多层 act
        att = torch.ones_like(out_ids, device=device)
        with _prof("stage:target_prefill"):
            # 只 hook 采集 eagle_layers 三层输入残差流，避免 output_hidden_states 收集全部 33 层
            with EagleActCapture(self.target_m, self.eagle_layers) as _cap:
                out0 = self.target_m(out_ids, attention_mask=att, use_cache=True)
        past = out0.past_key_values
        recovery_logits = out0.logits[0, -1]
        # 全前缀各位置 aux act；
        acts_all = _cap.acts(last_token_only=False)
        root_acts = acts_all[:, -1, :]
        draft_state: Optional[_DraftKvState] = None
        if self.autoregressive_draft:
            with _prof("stage:draft_prefill_init"):
                d_logits, d_hidden, d_kv = self.draft_model.prefill(out_ids, acts_all)
            draft_state = _DraftKvState(
                kv=d_kv,
                prefix_len=int(out_ids.shape[1]),
                root_logits=d_logits,
                root_hidden=d_hidden,
            )

        while new_tok < max_new_tokens:
            if eos_token_id is not None and out_ids[0, -1].item() == eos_token_id:
                break  # 已到 EOS，停止
            last_t = out_ids[:, -1:]
            append, _, _, _ = self.generate_step_payload(
                out_ids, past, last_t, recovery_logits, root_acts, acts_all, draft_state
            )
            # Enforce the public generation boundary before mutating target or
            # draft KV.  Otherwise a multi-token accepted chunk can overshoot
            # max_new_tokens or retain tokens after EOS while still being
            # reported as lossless by a prefix-only comparison.
            append = append[:, : max_new_tokens - new_tok]
            if eos_token_id is not None:
                eos_pos = (append[0] == eos_token_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    append = append[:, : int(eos_pos[0, 0].item()) + 1]
            matched_draft = min(
                self._last_matched_draft_tokens, int(append.shape[1])
            )
            proposed_draft = self._last_proposed_draft_tokens
            # 把 append 重放进 target KV。带 past 时 attention_mask 必须覆盖 [past + append] 全长，
            # 否则 append 不会 attend 到前缀
            # （extend verify 在 past 上临时 extend 后 crop、不污染 past；故 past 始终 = out_ids 的 KV。）
            full_attn = torch.ones(
                (1, int(past.get_seq_length()) + int(append.shape[1])),
                device=device,
                dtype=torch.long,
            )
            with _prof("stage:target_replay"):
                with EagleActCapture(self.target_m, self.eagle_layers) as _cap:
                    out_sync = self.target_m(
                        append,
                        past_key_values=past,
                        use_cache=True,
                        attention_mask=full_attn,
                    )
            past = out_sync.past_key_values
            recovery_logits = out_sync.logits[0, -1]
            append_acts = _cap.acts(last_token_only=False)
            acts_all = torch.cat([acts_all, append_acts], dim=1)
            root_acts = acts_all[:, -1, :]
            if self.autoregressive_draft and draft_state is not None:
                p_old = draft_state.prefix_len
                p_new = int(out_ids.shape[1]) + int(append.shape[1])
                ext_acts = acts_all[:, p_old - 1 : p_new - 1, :]
                with _prof("stage:draft_extend"):
                    d_logits, d_hidden, d_kv = self.draft_model.extend_tokens(
                        append, ext_acts, draft_state.kv, p_old - 1
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
            self.n_proposed_draft_tokens += proposed_draft
            if eos_token_id is not None and (append == eos_token_id).any():
                break
        return out_ids

    # ------------------------------------------------------------------
    # Step5/6：static KV + CUDA graph；top-1 时用 pending-token commit 消除 replay。
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_static_graph(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        from .target_graph import (
            StaticGraphVerifier,
            build_causal_replay_mask,
            make_static_cache,
        )

        device = input_ids.device
        dtype = next(self.target_m.parameters()).dtype
        out_ids = input_ids.clone()
        P = int(out_ids.shape[1])
        L_tree = int(self.cfg.max_tree_nodes)
        eliminate_replay = bool(getattr(self.cfg, "eliminate_replay", True))
        fast_chain = eliminate_replay and self.cfg.topk == 1
        fast_tree = eliminate_replay and self.cfg.topk > 1
        # 缓存上限：前缀 + 新 token + verify 临时写入的 L 个树 slot + 余量
        need = P + max_new_tokens + L_tree + int(self.cfg.graph_cache_margin)
        # static cache + 捕获好的 verify graph 跨 generate() 调用复用（避免每次 cudaMalloc + 重捕获）；
        # 仅当所需长度变大或树规模变化时才重建。每次调用 reset 清空逻辑内容。
        sg = getattr(self, "_sg_state", None)
        if sg is None or sg["max_len"] < need or sg["L"] != L_tree:
            sc = make_static_cache(self.target_m.config, need, device, dtype)
            verifier = StaticGraphVerifier(
                self.target_m, sc, need, device, dtype,
                enable_graph=bool(getattr(self.cfg, "graph_capture", True)),
                eagle_layers=self.eagle_layers,
            )
            self._sg_state = {"sc": sc, "verifier": verifier, "max_len": need, "L": L_tree}
        else:
            sc = sg["sc"]
            verifier = sg["verifier"]
            sc.reset()
        max_len = self._sg_state["max_len"]

        # Prefill：写入 static cache，取末位 logits + 多层 act
        with _prof("stage:target_prefill"):
            with EagleActCapture(self.target_m, self.eagle_layers) as _cap:
                out0 = self.target_m(
                    out_ids,
                    past_key_values=sc,
                    cache_position=torch.arange(P, device=device),
                    use_cache=True,
                )
        recovery_logits = out0.logits[0, -1]
        acts_all = _cap.acts(last_token_only=False)
        cur_len = P

        with _prof("stage:draft_prefill_init"):
            d_logits, d_hidden, d_kv = self.draft_model.prefill(out_ids, acts_all)
        draft_state = _DraftKvState(
            kv=d_kv, prefix_len=P, root_logits=d_logits, root_hidden=d_hidden
        )

        if eliminate_replay:
            # Step6 不变量：target KV 覆盖 out_ids[:-1]，末 token 留作下一轮 pending。
            cur_len = P - 1
            verifier._set_cache_length(cur_len)

        new_tok = 0
        while new_tok < max_new_tokens:
            if eos_token_id is not None and out_ids[0, -1].item() == eos_token_id:
                break
            with _prof("stage:draft_build_tree"):
                draft_res = self._build_tree(
                    out_ids, out_ids[:, -1:], None, None, draft_state
                )
            layout = tree_draft_to_verify_layout(draft_res)
            leaves = _leaf_indices(draft_res.parent)
            shift_acts = None
            if fast_chain:
                with _prof("stage:target_verify"):
                    append, shift_acts, matched_draft, cur_len = (
                        verifier.verify_chain_and_commit(
                            out_ids[:, -1:],
                            draft_res.token_ids,
                            draft_res.parent,
                            cur_len,
                            max_append_tokens=max_new_tokens - new_tok,
                            eos_token_id=eos_token_id,
                        )
                    )
                append_list = append[0].tolist()
            elif fast_tree:
                if not leaves:
                    raise RuntimeError("multi-branch replay elimination requires a non-empty tree")
                with _prof("stage:target_verify"):
                    append, shift_acts, matched_draft, cur_len = (
                        verifier.verify_tree_and_commit(
                            out_ids[:, -1:],
                            layout,
                            cur_len,
                            parent=draft_res.parent,
                            token_ids=draft_res.token_ids,
                            leaves=leaves,
                            max_append_tokens=max_new_tokens - new_tok,
                            eos_token_id=eos_token_id,
                        )
                    )
                append_list = append[0].tolist()
            elif not leaves:
                append_list = [int(recovery_logits.argmax(dim=-1).item())]
                matched_draft = 0
            else:
                with _prof("stage:target_verify"):
                    append_list, matched_draft, _, _ = verifier.verify(
                        layout, cur_len, recovery_logits,
                        parent=draft_res.parent, token_ids=draft_res.token_ids, leaves=leaves,
                    )
            if not eliminate_replay:
                append = torch.tensor([append_list], device=device, dtype=torch.long)
                append = append[:, : max_new_tokens - new_tok]
                if eos_token_id is not None:
                    eos_pos = (append[0] == eos_token_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        append = append[:, : int(eos_pos[0, 0].item()) + 1]
                matched_draft = min(matched_draft, int(append.shape[1]))
            A = int(append.shape[1])

            if not eliminate_replay:
                # Step5 安全路径：多分支树的接受 KV 不连续，仍需 replay 顺序写入。
                replay_mask = build_causal_replay_mask(cur_len, A, max_len, device, dtype)
                with _prof("stage:target_replay"):
                    with EagleActCapture(self.target_m, self.eagle_layers) as _cap:
                        out_sync = self.target_m(
                            append,
                            past_key_values=sc,
                            position_ids=torch.arange(cur_len, cur_len + A, device=device).unsqueeze(0),
                            cache_position=torch.arange(cur_len, cur_len + A, device=device),
                            attention_mask=replay_mask,
                            use_cache=True,
                        )
                recovery_logits = out_sync.logits[0, -1]
                append_acts = _cap.acts(last_token_only=False)
                acts_all = torch.cat([acts_all, append_acts], dim=1)
                cur_len += A

            p_old = draft_state.prefix_len
            p_new = int(out_ids.shape[1]) + A
            ext_acts = (
                shift_acts
                if eliminate_replay
                else acts_all[:, p_old - 1 : p_new - 1, :]
            )
            with _prof("stage:draft_extend"):
                d_logits, d_hidden, d_kv = self.draft_model.extend_tokens(
                    append, ext_acts, draft_state.kv, p_old - 1
                )
            draft_state = _DraftKvState(
                kv=d_kv, prefix_len=p_new, root_logits=d_logits, root_hidden=d_hidden
            )
            out_ids = torch.cat([out_ids, append], dim=1)
            new_tok += A
            self.n_rounds += 1
            self.n_accepted_tokens += A
            self.n_matched_draft_tokens += matched_draft
            self.n_proposed_draft_tokens += len(draft_res.token_ids)
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
