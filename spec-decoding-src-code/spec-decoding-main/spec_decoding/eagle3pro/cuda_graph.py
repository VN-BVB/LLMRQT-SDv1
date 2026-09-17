# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph replay for the fixed-shape EAGLE3Pro top-1 decode path."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .hidden import forward_target_selected_eagle_acts


class Eagle3ProCudaGraphUnavailable(RuntimeError):
    """Raised after a capture/replay failure so the runner can use eager safely."""


class Eagle3ProCudaGraphRunner:
    """Own fixed-shape Target and accepted-length-bucketed Draft graphs.

    Token ids, EAGLE acts, positions and KV storage retain fixed addresses.  The
    logical KV lengths live in CUDA int32 tensors, so every replay may operate at
    a different decode position without recapturing the graph.
    """

    def __init__(
        self,
        target_m: Any,
        draft_model: Any,
        eagle_layers: Sequence[int],
        target_cache: Any,
        draft_cache: Any,
        *,
        target_query_len: int,
        max_draft_append: int,
        warmup_iters: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise Eagle3ProCudaGraphUnavailable("CUDA is unavailable")
        if not getattr(target_cache, "graph_safe", False):
            raise Eagle3ProCudaGraphUnavailable("target cache is not graph-safe")
        if not getattr(draft_cache, "graph_safe", False):
            raise Eagle3ProCudaGraphUnavailable("draft cache is not graph-safe")

        self.target_m = target_m
        self.draft_model = draft_model
        self.eagle_layers = tuple(int(i) for i in eagle_layers)
        self.target_cache = target_cache
        self.draft_cache = draft_cache
        self.target_query_len = int(target_query_len)
        self.max_draft_append = int(max_draft_append)
        self.warmup_iters = int(warmup_iters)
        if self.target_query_len < 2 or self.max_draft_append < 2:
            raise Eagle3ProCudaGraphUnavailable("multi-step graph sizes must be >= 2")
        self.enabled = True
        self.disabled_reason: str | None = None

        self._target_graph: torch.cuda.CUDAGraph | None = None
        self._target_ids: torch.Tensor | None = None
        self._target_positions: torch.Tensor | None = None
        self._target_pred_ids: torch.Tensor | None = None
        self._target_acts: torch.Tensor | None = None
        self._position_table = torch.arange(
            target_cache.max_cache_len,
            device=target_cache.key_cache[0].device,
            dtype=torch.long,
        )

        self._draft_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._draft_ids: dict[int, torch.Tensor] = {}
        self._draft_acts: dict[int, torch.Tensor] = {}
        self._draft_logits: dict[int, torch.Tensor] = {}
        self._draft_hidden: dict[int, torch.Tensor] = {}

        self._proposal_graph: torch.cuda.CUDAGraph | None = None
        self._proposal_first_id: torch.Tensor | None = None
        self._proposal_root_hidden: torch.Tensor | None = None
        self._proposal_chain: torch.Tensor | None = None
        self._proposal_capture_start = 0

        self.target_captures = 0
        self.draft_captures = 0
        self.proposal_captures = 0
        self.target_replays = 0
        self.draft_replays = 0
        self.proposal_replays = 0
        self.fallbacks = 0

    def compatible(self, target_cache: Any, draft_cache: Any) -> bool:
        return self.target_cache is target_cache and self.draft_cache is draft_cache

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "target_captures": self.target_captures,
            "draft_captures": self.draft_captures,
            "proposal_captures": self.proposal_captures,
            "target_replays": self.target_replays,
            "draft_replays": self.draft_replays,
            "proposal_replays": self.proposal_replays,
            "fallbacks": self.fallbacks,
        }

    def disable(self, reason: str) -> None:
        if self.enabled:
            self.fallbacks += 1
        self.enabled = False
        self.disabled_reason = reason

    def _target_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        out, acts = forward_target_selected_eagle_acts(
            self.target_m.model,
            self.eagle_layers,
            last_token_only=False,
            input_ids=self._target_ids,
            past_key_values=self.target_cache,
            position_ids=self._target_positions,
            use_cache=True,
            attention_mask={"full_attention": None},
        )
        logits = self.target_m.get_output_embeddings()(out.last_hidden_state[0])
        return logits.argmax(dim=-1), acts

    def _warmup_target(self, before: int) -> None:
        current = torch.cuda.current_stream()
        side = torch.cuda.Stream(device=current.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.target_cache.set_length(before)
                self._target_forward()
        current.wait_stream(side)

    def _capture_target(self, verify_ids: torch.Tensor, before: int) -> None:
        self._target_ids = torch.empty_like(verify_ids)
        self._target_positions = torch.empty_like(verify_ids)
        self._target_ids.copy_(verify_ids)
        self._target_positions.copy_(
            self._position_table[
                before : before + self.target_query_len
            ].view(1, self.target_query_len)
        )
        self._warmup_target(before)
        self.target_cache.set_length(before)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pred_ids, acts = self._target_forward()
        self._target_graph = graph
        self._target_pred_ids = pred_ids
        self._target_acts = acts
        self.target_cache.set_length(before)
        self.target_captures += 1

    @torch.inference_mode()
    def verify_target(
        self,
        verify_ids: torch.Tensor,
        *,
        before: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            raise Eagle3ProCudaGraphUnavailable(self.disabled_reason or "graph disabled")
        if before + self.target_query_len > self.target_cache.max_cache_len:
            raise Eagle3ProCudaGraphUnavailable("target sequence exceeds graph workspace")
        if tuple(verify_ids.shape) != (1, self.target_query_len):
            raise Eagle3ProCudaGraphUnavailable(
                f"target graph requires shape [1, {self.target_query_len}]"
            )

        try:
            if self._target_graph is None:
                self._capture_target(verify_ids, before)
            self._target_ids.copy_(verify_ids)
            self._target_positions.copy_(
                self._position_table[
                    before : before + self.target_query_len
                ].view(1, self.target_query_len)
            )
            self.target_cache.set_length(before)
            self._target_graph.replay()
            self.target_cache.mark_replayed(before + self.target_query_len)
            self.target_replays += 1
            return self._target_pred_ids, self._target_acts
        except Eagle3ProCudaGraphUnavailable:
            raise
        except Exception as exc:
            self.target_cache.set_length(before)
            self.disable(f"target capture/replay failed: {type(exc).__name__}: {exc}")
            raise Eagle3ProCudaGraphUnavailable(self.disabled_reason) from exc

    def _draft_forward(self, size: int, start_position: int):
        return self.draft_model.extend_tokens(
            self._draft_ids[size],
            self._draft_acts[size],
            self.draft_cache,
            start_position,
        )

    def _proposal_forward(self) -> torch.Tensor:
        token = self._proposal_first_id
        hidden = self._proposal_root_hidden
        self._proposal_chain[:, :1].copy_(token)
        for depth in range(1, self.target_query_len - 1):
            logits, hidden, _ = self.draft_model.step_flash_cache(
                token,
                hidden,
                self.draft_cache,
                self._proposal_capture_start + depth - 1,
            )
            token = self.draft_model.greedy_target_ids(logits)
            token = token[:, :1]
            self._proposal_chain[:, depth : depth + 1].copy_(token)
        return self._proposal_chain

    def _warmup_proposal(self, start_position: int) -> None:
        current = torch.cuda.current_stream()
        side = torch.cuda.Stream(device=current.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.draft_cache.set_length(start_position)
                self._proposal_forward()
        current.wait_stream(side)
        self.draft_cache.set_length(start_position)

    def _capture_proposal(
        self,
        first_id: torch.Tensor,
        root_hidden: torch.Tensor,
        start_position: int,
    ) -> None:
        steps = self.target_query_len - 1
        self._proposal_first_id = torch.empty_like(first_id)
        self._proposal_root_hidden = torch.empty_like(root_hidden)
        self._proposal_chain = torch.empty(
            (1, steps), device=first_id.device, dtype=torch.long
        )
        self._proposal_capture_start = int(start_position)
        self._proposal_first_id.copy_(first_id)
        self._proposal_root_hidden.copy_(root_hidden)
        self._warmup_proposal(start_position)

        self.draft_cache.set_length(start_position)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._proposal_forward()
        self._proposal_graph = graph
        self.draft_cache.set_length(start_position)
        self.proposal_captures += 1

    @torch.inference_mode()
    def build_draft_chain(
        self,
        first_id: torch.Tensor,
        root_hidden: torch.Tensor,
        *,
        start_position: int,
    ) -> torch.LongTensor:
        """Replay all recursive top-1 proposal steps as one CUDA Graph."""
        if not self.enabled:
            raise Eagle3ProCudaGraphUnavailable(self.disabled_reason or "graph disabled")
        try:
            if self._proposal_graph is None:
                self._capture_proposal(first_id, root_hidden, start_position)
            self._proposal_first_id.copy_(first_id)
            self._proposal_root_hidden.copy_(root_hidden)
            self.draft_cache.set_length(start_position)
            self._proposal_graph.replay()
            # Proposal KV occupies scratch tail only; accepted commit overwrites it.
            self.draft_cache.set_length(start_position)
            self.proposal_replays += 1
            return self._proposal_chain
        except Eagle3ProCudaGraphUnavailable:
            raise
        except Exception as exc:
            self.draft_cache.set_length(start_position)
            self.disable(f"proposal capture/replay failed: {type(exc).__name__}: {exc}")
            raise Eagle3ProCudaGraphUnavailable(self.disabled_reason) from exc

    def _warmup_draft(self, size: int, start_position: int) -> None:
        current = torch.cuda.current_stream()
        side = torch.cuda.Stream(device=current.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.draft_cache.set_length(start_position)
                self._draft_forward(size, start_position)
        current.wait_stream(side)

    def _capture_draft(
        self,
        size: int,
        sample_ids: torch.Tensor,
        sample_acts: torch.Tensor,
        start_position: int,
    ) -> None:
        self._draft_ids[size] = torch.empty_like(sample_ids)
        self._draft_acts[size] = torch.empty_like(sample_acts)
        self._draft_ids[size].copy_(sample_ids)
        self._draft_acts[size].copy_(sample_acts)
        self._warmup_draft(size, start_position)
        self.draft_cache.set_length(start_position)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits, hidden, _ = self._draft_forward(size, start_position)
        self._draft_graphs[size] = graph
        self._draft_logits[size] = logits
        self._draft_hidden[size] = hidden
        self.draft_cache.set_length(start_position)
        self.draft_captures += 1

    def _capture_all_draft_shapes(
        self,
        token_ids: torch.Tensor,
        target_acts: torch.Tensor,
        start_position: int,
    ) -> None:
        one_ids = token_ids[:, :1]
        one_acts = target_acts[:, :1, :]
        for size in range(1, self.max_draft_append + 1):
            if token_ids.shape[1] >= size:
                sample_ids = token_ids[:, :size]
                sample_acts = target_acts[:, :size, :]
            else:
                sample_ids = one_ids.repeat(1, size)
                sample_acts = one_acts.repeat(1, size, 1)
            self._capture_draft(
                size, sample_ids, sample_acts, start_position
            )
        self.draft_cache.set_length(start_position)

    @torch.inference_mode()
    def extend_draft(
        self,
        token_ids: torch.Tensor,
        target_acts: torch.Tensor,
        *,
        start_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if not self.enabled:
            raise Eagle3ProCudaGraphUnavailable(self.disabled_reason or "graph disabled")
        size = int(token_ids.shape[1])
        if size < 1 or size > self.max_draft_append:
            raise Eagle3ProCudaGraphUnavailable(
                f"draft graph requires append length 1..{self.max_draft_append}"
            )
        if start_position + size > self.draft_cache.max_cache_len:
            raise Eagle3ProCudaGraphUnavailable("draft sequence exceeds graph workspace")

        try:
            if not self._draft_graphs:
                self._capture_all_draft_shapes(
                    token_ids, target_acts, start_position
                )
            self._draft_ids[size].copy_(token_ids)
            self._draft_acts[size].copy_(target_acts)
            self.draft_cache.set_length(start_position)
            self._draft_graphs[size].replay()
            self.draft_cache.mark_replayed(start_position + size)
            self.draft_replays += 1
            return (
                self._draft_logits[size],
                self._draft_hidden[size],
                self.draft_cache,
            )
        except Eagle3ProCudaGraphUnavailable:
            raise
        except Exception as exc:
            self.draft_cache.set_length(start_position)
            self.disable(f"draft capture/replay failed: {type(exc).__name__}: {exc}")
            raise Eagle3ProCudaGraphUnavailable(self.disabled_reason) from exc
