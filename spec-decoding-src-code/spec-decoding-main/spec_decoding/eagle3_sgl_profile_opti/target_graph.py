# SPDX-License-Identifier: Apache-2.0
"""Step 5：target decode 的 CUDA graph（仅 verify 前向）。

设计：

- target KV 换成 连续 static 缓存（HF StaticCache），按 cache_position 用
  index_copy_ 原地写 → 没有 DynamicCache 的逐步 cat/realloc，shape 全程固定。
- verify 前向：对固定 L 个树节点跑一次 target，注意力用 固定 shape 的 4D 树 mask
  + 普通 SDPA（mask 宽度 padding 到 max_len），读整段 static KV。
- 因为输入/输出/KV 全是固定 shape 的持久 buffer，且写位置经由 cache_position
  驱动（index_copy_ 运行时读索引），整个 verify 前向可被 CUDA graph 捕获，
  每轮只 copy_ 新的 tokens/position/cache_position/mask 进静态 buffer 后 graph.replay()。

Step 5 中 verify 写入的树 KV 由 replay/下一次 verify 覆盖。Step 6 则让正式 KV 落后
一个 pending token：top-1 直接提交连续 KV，多分支把 accepted-path KV compact 到连续
slots；correction/bonus 留作下一轮 pending，因此不再运行 target replay。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .hidden import EagleActCapture
from .tree_verify_full import _select_append_root_child
from .tree_verify_mask import (
    TreeVerifyLayout,
    _ancestor_self_matrix,
    layout_tree_position_ids,
)


def make_static_cache(config: Any, max_len: int, device: torch.device, dtype: torch.dtype):
    """构造 HF StaticCache"""
    from transformers import StaticCache

    for kwargs in (
        dict(config=config, max_batch_size=1, max_cache_len=max_len, device=device, dtype=dtype),
        dict(config=config, max_cache_len=max_len, device=device, dtype=dtype),
    ):
        try:
            return StaticCache(**kwargs)
        except TypeError:
            continue
    return StaticCache(config, 1, max_len, device, dtype)


def build_padded_tree_mask(
    layout: TreeVerifyLayout,
    cur_len: int,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[1,1,L,max_len] bias：前缀 [0:cur_len]=0，树段 [cur_len:cur_len+L] 按祖先矩阵，
    其余（未写/未来 slot）=-inf。可写进 out（graph 静态 buffer）以原地更新。"""
    L = len(layout)
    neg = torch.finfo(dtype).min
    A = _ancestor_self_matrix(layout.parent, L, device)  # [L,L] bool
    allow = torch.zeros((L, max_len), dtype=torch.bool, device=device)
    allow[:, :cur_len] = True
    allow[:, cur_len:cur_len + L] = A
    if out is None:
        out = torch.empty((1, 1, L, max_len), dtype=dtype, device=device)
    out.fill_(neg)
    out[0, 0][allow] = 0.0
    return out


def build_causal_replay_mask(
    cur_len: int, a_len: int, max_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """[1,1,A,max_len] bias：replay 第 j 个 token（位置 cur_len+j）causal attend
    [0:cur_len+j]，其余（未来/未写 slot）=-inf"""
    neg = torch.finfo(dtype).min
    rows = torch.arange(a_len, device=device).unsqueeze(1)
    cols = torch.arange(max_len, device=device).unsqueeze(0)
    allow = cols <= (cur_len + rows)  # [A, max_len] bool
    m = torch.full((1, 1, a_len, max_len), neg, dtype=dtype, device=device)
    m[0, 0][allow] = 0.0
    return m


def build_pending_tree_mask(
    layout: TreeVerifyLayout,
    before: int,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Step6 的 ``pending + tree`` 固定形状 attention bias。

    row 0 是 pending token；后续 L 行是树节点。所有行可见正式前缀，树节点还可见
    pending 和自己的祖先。树 KV 仍写入连续临时 slots，接受后再按 path index compact。
    """
    L = len(layout)
    q_len = 1 + L
    neg = torch.finfo(dtype).min
    allow = torch.zeros((q_len, max_len), dtype=torch.bool, device=device)
    allow[0, : before + 1] = True
    allow[1:, : before + 1] = True
    allow[1:, before + 1 : before + 1 + L] = _ancestor_self_matrix(
        layout.parent, L, device
    )
    if out is None:
        out = torch.empty((1, 1, q_len, max_len), dtype=dtype, device=device)
    out.fill_(neg)
    out[0, 0][allow] = 0.0
    return out


class StaticGraphVerifier:
    """对 L=固定 的树 verify 前向做一次 CUDA graph 捕获，后续轮次仅 replay。

    L 与捕获时不同则回退回 eager 前向。
    """

    def __init__(
        self,
        target_m: torch.nn.Module,
        static_cache: Any,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        enable_graph: bool = True,
        eagle_layers: Optional[List[int]] = None,
    ) -> None:
        self.target_m = target_m
        self.sc = static_cache
        self.max_len = int(max_len)
        self.device = device
        self.dtype = dtype
        self.enable_graph = enable_graph and torch.cuda.is_available()
        self.eagle_layers = list(eagle_layers) if eagle_layers is not None else None
        self._L: Optional[int] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        # 静态输入/输出 buffer（捕获后地址固定）
        self._tok_buf: Optional[torch.Tensor] = None
        self._pos_buf: Optional[torch.Tensor] = None
        self._cachepos_buf: Optional[torch.Tensor] = None
        self._mask_buf: Optional[torch.Tensor] = None
        self._logits_static: Optional[torch.Tensor] = None
        # Step6 top-1 pending-token graph。它与旧 tree verify graph 分开保存，
        # 输入 q_len = 1(pending) + draft chain，输出同时保留预测 id 和 EAGLE acts。
        self._chain_q: Optional[int] = None
        self._chain_graph: Optional[torch.cuda.CUDAGraph] = None
        self._chain_tok_buf: Optional[torch.Tensor] = None
        self._chain_pos_buf: Optional[torch.Tensor] = None
        self._chain_cachepos_buf: Optional[torch.Tensor] = None
        self._chain_mask_buf: Optional[torch.Tensor] = None
        self._chain_pred_static: Optional[torch.Tensor] = None
        self._chain_acts_static: Optional[torch.Tensor] = None
        # 多分支 Step6：pending + tree verify，以及 verify 后的路径 KV compact。
        self._tree_commit_q: Optional[int] = None
        self._tree_commit_graph: Optional[torch.cuda.CUDAGraph] = None
        self._tree_commit_tok_buf: Optional[torch.Tensor] = None
        self._tree_commit_pos_buf: Optional[torch.Tensor] = None
        self._tree_commit_cachepos_buf: Optional[torch.Tensor] = None
        self._tree_commit_mask_buf: Optional[torch.Tensor] = None
        self._tree_commit_logits_static: Optional[torch.Tensor] = None
        self._tree_commit_acts_static: Optional[torch.Tensor] = None

    def _set_cache_length(self, length: int) -> None:
        """Restore the logical append position of recent HF StaticCache layers.

        Transformers 5.14 StaticLayer.update() ignores cache_position and writes
        at cumulative_length.  A verify forward is temporary, so its L draft
        nodes must not advance that counter before accepted-token replay.  Older
        cache implementations without this counter continue to use the explicit
        cache_position buffers and need no adjustment.
        """
        for layer in getattr(self.sc, "layers", ()):
            cumulative = getattr(layer, "cumulative_length", None)
            if isinstance(cumulative, torch.Tensor):
                cumulative.fill_(length)
            elif isinstance(cumulative, int):
                layer.cumulative_length = length

    # --- 每轮把当轮的值写进静态输入 buffer ---
    def _fill_inputs(self, layout: TreeVerifyLayout, cur_len: int) -> None:
        L = len(layout)
        self._tok_buf[0].copy_(torch.tensor(layout.token_ids, device=self.device, dtype=torch.long))
        self._pos_buf[0].copy_(layout_tree_position_ids(layout, cur_len).to(self.device))
        self._cachepos_buf.copy_(torch.arange(cur_len, cur_len + L, device=self.device))
        build_padded_tree_mask(layout, cur_len, self.max_len, self.device, self.dtype, out=self._mask_buf)

    def _run_forward(self) -> torch.Tensor:
        out = self.target_m(
            self._tok_buf,
            past_key_values=self.sc,
            position_ids=self._pos_buf,
            cache_position=self._cachepos_buf,
            attention_mask=self._mask_buf,
            use_cache=True,
        )
        return out.logits

    def _fill_chain_inputs(
        self,
        pending_token_id: torch.LongTensor,
        token_ids: List[int],
        before: int,
    ) -> None:
        candidate_ids = torch.tensor(
            [token_ids], device=self.device, dtype=torch.long
        )
        verify_ids = torch.cat([pending_token_id, candidate_ids], dim=1)
        self._chain_tok_buf.copy_(verify_ids)
        q_len = int(verify_ids.shape[1])
        positions = torch.arange(
            before, before + q_len, device=self.device, dtype=torch.long
        )
        self._chain_pos_buf[0].copy_(positions)
        self._chain_cachepos_buf.copy_(positions)
        self._chain_mask_buf.copy_(
            build_causal_replay_mask(
                before, q_len, self.max_len, self.device, self.dtype
            )
        )

    def _run_chain_forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.eagle_layers is None:
            raise ValueError("top-1 replay elimination requires eagle_layers")
        with EagleActCapture(self.target_m, self.eagle_layers) as capture:
            out = self.target_m(
                self._chain_tok_buf,
                past_key_values=self.sc,
                position_ids=self._chain_pos_buf,
                cache_position=self._chain_cachepos_buf,
                attention_mask=self._chain_mask_buf,
                use_cache=True,
            )
        # argmax 与三层 act 的拼接也捕获进图；replay 后只读取固定输出 buffer。
        return out.logits[0].argmax(dim=-1), capture.acts(last_token_only=False)

    def _capture_chain(
        self,
        pending_token_id: torch.LongTensor,
        token_ids: List[int],
        before: int,
    ) -> None:
        q_len = 1 + len(token_ids)
        self._chain_q = q_len
        self._chain_tok_buf = torch.zeros(
            (1, q_len), dtype=torch.long, device=self.device
        )
        self._chain_pos_buf = torch.zeros(
            (1, q_len), dtype=torch.long, device=self.device
        )
        self._chain_cachepos_buf = torch.zeros(
            (q_len,), dtype=torch.long, device=self.device
        )
        self._chain_mask_buf = torch.empty(
            (1, 1, q_len, self.max_len), dtype=self.dtype, device=self.device
        )
        self._fill_chain_inputs(pending_token_id, token_ids, before)
        if not self.enable_graph:
            return

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._set_cache_length(before)
                self._run_chain_forward()
        torch.cuda.current_stream().wait_stream(stream)
        self._set_cache_length(before)
        self._chain_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._chain_graph):
            pred, acts = self._run_chain_forward()
        self._chain_pred_static = pred
        self._chain_acts_static = acts
        self._set_cache_length(before)

    @torch.inference_mode()
    def verify_chain_and_commit(
        self,
        pending_token_id: torch.LongTensor,
        token_ids: List[int],
        parent: List[int],
        before: int,
        *,
        max_append_tokens: int,
        eos_token_id: Optional[int],
    ) -> Tuple[torch.LongTensor, torch.Tensor, int, int]:
        """用一次 causal forward 验证 top-1 链并直接提交连续 KV。

        返回 ``(append, shift_acts, matched, committed_len)``。最后生成的
        correction/bonus token 保持 pending，不写 target KV；下一轮把它放到固定输入
        buffer 的第 0 位。这样删除 replay，同时不改变 CUDA Graph 中任何 tensor 地址。
        """
        expected_parent = [-1] + list(range(max(0, len(token_ids) - 1)))
        if parent != expected_parent:
            raise ValueError("replay elimination requires a single top-1 chain")
        q_len = 1 + len(token_ids)
        if self._chain_q is None:
            self._capture_chain(pending_token_id, token_ids, before)
        elif q_len != self._chain_q:
            raise ValueError(
                f"top-1 CUDA graph requires fixed q_len={self._chain_q}, got {q_len}"
            )
        else:
            self._fill_chain_inputs(pending_token_id, token_ids, before)

        self._set_cache_length(before)
        if self.enable_graph:
            if self._chain_graph is not None and self._chain_pred_static is not None:
                # 首次调用已经由 capture 产生有效输出；后续调用才 replay。
                if getattr(self, "_chain_capture_consumed", False):
                    self._chain_graph.replay()
                else:
                    self._chain_capture_consumed = True
                pred = self._chain_pred_static
                acts = self._chain_acts_static
            else:
                raise RuntimeError("top-1 CUDA graph capture produced no outputs")
        else:
            pred, acts = self._run_chain_forward()

        pred_ids = pred.tolist()
        matched = 0
        for idx, token_id in enumerate(token_ids):
            if pred_ids[idx] != token_id:
                break
            matched += 1
        append_list = token_ids[:matched] + [pred_ids[matched]]
        append_list = append_list[:max_append_tokens]
        if eos_token_id is not None and eos_token_id in append_list:
            append_list = append_list[: append_list.index(eos_token_id) + 1]

        kept_draft = min(matched, len(append_list))
        committed_len = before + 1 + kept_draft
        # 只推进逻辑长度；StaticCache 的 K/V storage 地址和容量始终不变。
        self._set_cache_length(committed_len)
        append = torch.tensor([append_list], device=self.device, dtype=torch.long)
        shift_acts = acts[:, : len(append_list), :]
        return append, shift_acts, kept_draft, committed_len

    def _fill_tree_commit_inputs(
        self,
        pending_token_id: torch.LongTensor,
        layout: TreeVerifyLayout,
        before: int,
    ) -> None:
        tree_ids = torch.tensor(
            [layout.token_ids], device=self.device, dtype=torch.long
        )
        verify_ids = torch.cat([pending_token_id, tree_ids], dim=1)
        self._tree_commit_tok_buf.copy_(verify_ids)
        q_len = int(verify_ids.shape[1])
        self._tree_commit_pos_buf[0, 0] = before
        self._tree_commit_pos_buf[0, 1:].copy_(
            layout_tree_position_ids(layout, before + 1).to(self.device)
        )
        self._tree_commit_cachepos_buf.copy_(
            torch.arange(before, before + q_len, device=self.device)
        )
        build_pending_tree_mask(
            layout,
            before,
            self.max_len,
            self.device,
            self.dtype,
            out=self._tree_commit_mask_buf,
        )

    def _run_tree_commit_forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.eagle_layers is None:
            raise ValueError("tree replay elimination requires eagle_layers")
        with EagleActCapture(self.target_m, self.eagle_layers) as capture:
            out = self.target_m(
                self._tree_commit_tok_buf,
                past_key_values=self.sc,
                position_ids=self._tree_commit_pos_buf,
                cache_position=self._tree_commit_cachepos_buf,
                attention_mask=self._tree_commit_mask_buf,
                use_cache=True,
            )
        return out.logits[0], capture.acts(last_token_only=False)

    def _capture_tree_commit(
        self,
        pending_token_id: torch.LongTensor,
        layout: TreeVerifyLayout,
        before: int,
    ) -> None:
        q_len = 1 + len(layout)
        self._tree_commit_q = q_len
        self._tree_commit_tok_buf = torch.zeros(
            (1, q_len), dtype=torch.long, device=self.device
        )
        self._tree_commit_pos_buf = torch.zeros(
            (1, q_len), dtype=torch.long, device=self.device
        )
        self._tree_commit_cachepos_buf = torch.zeros(
            (q_len,), dtype=torch.long, device=self.device
        )
        self._tree_commit_mask_buf = torch.empty(
            (1, 1, q_len, self.max_len), dtype=self.dtype, device=self.device
        )
        self._fill_tree_commit_inputs(pending_token_id, layout, before)
        if not self.enable_graph:
            return

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._set_cache_length(before)
                self._run_tree_commit_forward()
        torch.cuda.current_stream().wait_stream(stream)
        self._set_cache_length(before)
        self._tree_commit_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._tree_commit_graph):
            logits, acts = self._run_tree_commit_forward()
        self._tree_commit_logits_static = logits
        self._tree_commit_acts_static = acts
        self._set_cache_length(before)

    def _compact_tree_path(
        self, before: int, path_indices: List[int], kept_draft: int
    ) -> None:
        """把非连续 tree slots gather 到 pending 后的连续正式 slots。

        只移动 K/V，不再运行 target。StaticCache backing tensors 本身不替换、不 resize，
        因而已经捕获的 CUDA Graph 地址仍有效。
        """
        if kept_draft <= 0:
            return
        source = torch.tensor(
            [before + 1 + i for i in path_indices[:kept_draft]],
            device=self.device,
            dtype=torch.long,
        )
        destination = torch.arange(
            before + 1,
            before + 1 + kept_draft,
            device=self.device,
            dtype=torch.long,
        )
        for layer in getattr(self.sc, "layers", ()):
            keys = layer.keys.index_select(2, source)
            values = layer.values.index_select(2, source)
            layer.keys.index_copy_(2, destination, keys)
            layer.values.index_copy_(2, destination, values)

    @torch.inference_mode()
    def verify_tree_and_commit(
        self,
        pending_token_id: torch.LongTensor,
        layout: TreeVerifyLayout,
        before: int,
        *,
        parent: List[int],
        token_ids: List[int],
        leaves: List[int],
        max_append_tokens: int,
        eos_token_id: Optional[int],
    ) -> Tuple[torch.LongTensor, torch.Tensor, int, int]:
        """多分支树的无 replay 提交：graph verify + accepted-path KV compact。"""
        from .tree_verify_full import path_node_indices_root_to_leaf

        q_len = 1 + len(layout)
        if self._tree_commit_q is None:
            self._capture_tree_commit(pending_token_id, layout, before)
        elif q_len != self._tree_commit_q:
            raise ValueError(
                f"tree CUDA graph requires fixed q_len={self._tree_commit_q}, got {q_len}"
            )
        else:
            self._fill_tree_commit_inputs(pending_token_id, layout, before)

        self._set_cache_length(before)
        if self.enable_graph:
            if self._tree_commit_graph is None:
                raise RuntimeError("tree CUDA graph capture produced no graph")
            if getattr(self, "_tree_commit_capture_consumed", False):
                self._tree_commit_graph.replay()
            else:
                self._tree_commit_capture_consumed = True
            logits = self._tree_commit_logits_static
            acts = self._tree_commit_acts_static
        else:
            logits, acts = self._run_tree_commit_forward()

        append_list, matched, best_leaf, _ = _select_append_root_child(
            logits[1:], logits[0], parent, token_ids, leaves
        )
        append_list = append_list[:max_append_tokens]
        if eos_token_id is not None and eos_token_id in append_list:
            append_list = append_list[: append_list.index(eos_token_id) + 1]
        kept_draft = min(matched, len(append_list))
        path_indices = path_node_indices_root_to_leaf(parent, best_leaf)
        self._compact_tree_path(before, path_indices, kept_draft)
        committed_len = before + 1 + kept_draft
        self._set_cache_length(committed_len)

        # 第 j 个 append token 消费其前一位置的 target act：pending + 已接受路径前缀。
        act_indices = [0] + [
            1 + i for i in path_indices[: max(0, len(append_list) - 1)]
        ]
        act_index = torch.tensor(act_indices, device=self.device, dtype=torch.long)
        shift_acts = acts.index_select(1, act_index)
        append = torch.tensor([append_list], device=self.device, dtype=torch.long)
        return append, shift_acts, kept_draft, committed_len

    def _capture(self, L: int, layout: TreeVerifyLayout, cur_len: int) -> None:
        self._L = L
        self._tok_buf = torch.zeros((1, L), dtype=torch.long, device=self.device)
        self._pos_buf = torch.zeros((1, L), dtype=torch.long, device=self.device)
        self._cachepos_buf = torch.zeros((L,), dtype=torch.long, device=self.device)
        self._mask_buf = torch.empty((1, 1, L, self.max_len), dtype=self.dtype, device=self.device)
        self._fill_inputs(layout, cur_len)
        if not self.enable_graph:
            return
        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._set_cache_length(cur_len)
                self._run_forward()
        torch.cuda.current_stream().wait_stream(s)
        self._set_cache_length(cur_len)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            logits = self._run_forward()
        self._logits_static = logits
        self._set_cache_length(cur_len)

    @torch.inference_mode()
    def verify(
        self,
        layout: TreeVerifyLayout,
        cur_len: int,
        recovery_logits: torch.Tensor,
        *,
        parent: List[int],
        token_ids: List[int],
        leaves: List[int],
    ) -> Tuple[List[int], int, int, List[int]]:
        L = len(layout)
        if self._L is None:
            self._capture(L, layout, cur_len)
            node_logits = (self._logits_static if self.enable_graph else self._run_forward())[0]
            result = _select_append_root_child(
                node_logits, recovery_logits, parent, token_ids, leaves
            )
            self._set_cache_length(cur_len)
            return result
        if L != self._L or not self.enable_graph:
            # L 与捕获不同 → eager 前向
            return self._verify_eager(
                layout, cur_len, recovery_logits,
                parent=parent, token_ids=token_ids, leaves=leaves,
            )
        self._fill_inputs(layout, cur_len)
        self._set_cache_length(cur_len)
        try:
            self._graph.replay()
            node_logits = self._logits_static[0]
            return _select_append_root_child(
                node_logits, recovery_logits, parent, token_ids, leaves
            )
        finally:
            self._set_cache_length(cur_len)

    @torch.inference_mode()
    def _verify_eager(self, layout, cur_len, recovery_logits, *, parent, token_ids, leaves):
        L = len(layout)
        tok = torch.tensor([layout.token_ids], device=self.device, dtype=torch.long)
        pos = layout_tree_position_ids(layout, cur_len).unsqueeze(0).to(self.device)
        cpos = torch.arange(cur_len, cur_len + L, device=self.device)
        mask = build_padded_tree_mask(layout, cur_len, self.max_len, self.device, self.dtype)
        self._set_cache_length(cur_len)
        try:
            out = self.target_m(tok, past_key_values=self.sc, position_ids=pos,
                                cache_position=cpos, attention_mask=mask, use_cache=True)
            node_logits = out.logits[0]
            return _select_append_root_child(
                node_logits, recovery_logits, parent, token_ids, leaves
            )
        finally:
            self._set_cache_length(cur_len)
