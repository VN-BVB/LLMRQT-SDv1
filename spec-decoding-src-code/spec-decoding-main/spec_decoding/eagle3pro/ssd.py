# SPDX-License-Identifier: Apache-2.0
"""Speculative Speculative Decoding (SSD) for the EAGLE3Pro fast chain.

This is deliberately a small integration layer instead of a copy of the SSD
runtime.  EAGLE3Pro already owns target verification, EAGLE feature capture,
and its KV caches; this module only adds the SSD policy:

* while the target verifies the current top-1 draft chain, fork the draft for
  likely ``(accepted_length, recovery_token)`` outcomes;
* cache the next draft state and proposal under that outcome;
* use the cached state on a hit, and fall back to the exact target-activation
  ``extend_tokens`` path on a miss.

The speculative branch uses the EAGLE draft hidden recursively when the
current target activations do not exist yet.  That can change future proposal
quality, but never the target's greedy acceptance rule or generated tokens.

The first implementation intentionally supports batch 1, greedy decoding and
the EAGLE3Pro ``topk=1/full_model_tree`` fast chain.  It uses immutable tuple
draft KV for safe forks; the target may still use EAGLE3Pro Flash KV.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Optional, Tuple

import torch

from .hidden import forward_target_selected_eagle_acts
from .runner import Eagle3SglGenerator, _DraftKvState
from .target_optim import Eagle3FlashKVCache
from .tree_draft import TreeDraftResult, expand_draft_tree_ar


@dataclass(frozen=True)
class SSDCacheKey:
    """Verification outcome used to select a pre-drafted continuation."""

    accepted_length: int
    recovery_token: int


@dataclass
class SSDCacheEntry:
    """Draft state after the outcome plus its already-built next proposal."""

    draft_state: _DraftKvState
    draft_tree: TreeDraftResult


@dataclass
class Eagle3ProSSDConfig:
    """SSD policy controls kept separate from the EAGLE tree configuration."""

    fan_out: int = 3
    overlap: bool = True

    def __post_init__(self) -> None:
        if self.fan_out < 1:
            raise ValueError("fan_out must be >= 1")


@dataclass(frozen=True)
class Eagle3ProSSDStats:
    """Per-generator counters for judging whether SSD hides useful work."""

    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    forked_branches: int
    populate_seconds: float
    wait_seconds: float


class _CompletedFuture:
    """Tiny Future-compatible holder used for CPU and no-overlap diagnosis."""

    def __init__(self, value: Dict[SSDCacheKey, SSDCacheEntry]) -> None:
        self._value = value

    def result(self) -> Dict[SSDCacheKey, SSDCacheEntry]:
        return self._value


def _as_tuple_kv(kv: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reject mutable Flash KV: speculative branches must not share writes."""

    if getattr(kv, "_eagle3pro_draft_flash_cache", False):
        raise ValueError(
            "SSD draft forking requires tuple KV; do not convert the draft "
            "cache to Eagle3DraftFlashKVCache"
        )
    if not isinstance(kv, tuple) or len(kv) != 2:
        raise TypeError("SSD currently requires draft KV as a (key, value) tuple")
    return kv


def _repeat_kv(
    kv: Tuple[torch.Tensor, torch.Tensor], batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make a read-only batch view; draft ``step`` returns new concatenated KV."""

    key, value = kv
    if key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("SSD prototype supports batch-1 draft state")
    shape = (batch_size,) + tuple(key.shape[1:])
    return key.expand(shape), value.expand((batch_size,) + tuple(value.shape[1:]))


def _select_recovery_tokens(
    draft_model: Any,
    logits: torch.Tensor,
    fan_out: int,
    *,
    excluded_token: Optional[int],
) -> list[int]:
    """Return top-F target-vocabulary recovery tokens, excluding the path token."""

    # Only one token can be excluded in a top-1 chain, so F+1 is sufficient.
    request_k = min(int(logits.shape[-1]), fan_out + int(excluded_token is not None))
    token_ids, _ = draft_model.topk_target_ids(logits, request_k)
    selected: list[int] = []
    for token in token_ids[0].tolist():
        token = int(token)
        if excluded_token is not None and token == excluded_token:
            continue
        if token not in selected:
            selected.append(token)
        if len(selected) == fan_out:
            break
    if len(selected) != fan_out:
        raise RuntimeError(
            f"could only select {len(selected)} unique recovery tokens for fan_out={fan_out}"
        )
    return selected


class Eagle3ProSSDGenerator(Eagle3SglGenerator):
    """EAGLE3Pro generator with an SSD next-round speculation cache.

    ``target`` and ``draft_model`` are the same objects and weights accepted by
    :class:`Eagle3SglGenerator`; no SSD-specific training is required.
    """

    def __init__(
        self,
        target: Any,
        cfg: Any,
        draft_topk_fn: Any = None,
        *,
        draft_model: Any,
        ssd_cfg: Optional[Eagle3ProSSDConfig] = None,
        autoregressive_draft: bool = True,
        refresh_target_each_node: bool = True,
    ) -> None:
        super().__init__(
            target,
            cfg,
            draft_topk_fn,
            draft_model=draft_model,
            autoregressive_draft=autoregressive_draft,
            refresh_target_each_node=refresh_target_each_node,
        )
        self.ssd_cfg = ssd_cfg or Eagle3ProSSDConfig()
        self._validate_ssd_mode()

        self.n_ssd_cache_hit = 0
        self.n_ssd_cache_miss = 0
        self.n_ssd_forked_branches = 0
        self.ssd_populate_seconds = 0.0
        self.ssd_wait_seconds = 0.0

        draft_device = next(self.draft_model.parameters()).device
        self._ssd_stream: Optional[torch.cuda.Stream] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        if self.ssd_cfg.overlap and draft_device.type == "cuda":
            self._ssd_stream = torch.cuda.Stream(device=draft_device)
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="eagle3pro-ssd"
            )
        self._close_lock = threading.Lock()
        self._closed = False

    def _validate_ssd_mode(self) -> None:
        errors = []
        if not self.autoregressive_draft:
            errors.append("autoregressive_draft must be enabled")
        if self.cfg.topk != 1:
            errors.append("topk must be 1 (branch-tree SSD is not implemented yet)")
        if self.cfg.verify_mode != "full_model_tree":
            errors.append("verify_mode must be full_model_tree")
        if self.cfg.verify_attn_backend == "triton_tree":
            errors.append("triton_tree is incompatible with the top-1 fast chain")
        if getattr(self.draft_model, "extra_layers", None):
            if len(self.draft_model.extra_layers) != 0:
                errors.append("the draft model must have exactly one midlayer")
        if errors:
            raise ValueError("EAGLE3Pro SSD requires: " + "; ".join(errors))

    @property
    def ssd_stats(self) -> Eagle3ProSSDStats:
        requests = self.n_ssd_cache_hit + self.n_ssd_cache_miss
        return Eagle3ProSSDStats(
            cache_hits=self.n_ssd_cache_hit,
            cache_misses=self.n_ssd_cache_miss,
            cache_hit_rate=self.n_ssd_cache_hit / max(1, requests),
            forked_branches=self.n_ssd_forked_branches,
            populate_seconds=self.ssd_populate_seconds,
            wait_seconds=self.ssd_wait_seconds,
        )

    def close(self) -> None:
        """Release the background worker after all submitted work has completed."""

        with self._close_lock:
            if self._closed:
                return
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._closed = True

    def _tree_from_state(self, state: _DraftKvState, device: torch.device) -> TreeDraftResult:
        return expand_draft_tree_ar(
            self.cfg,
            self.draft_model,
            state.root_logits,
            state.root_hidden,
            state.kv,
            state.prefix_len - 1,
            device=device,
        )

    @torch.inference_mode()
    def _populate_speculation_cache(
        self,
        state: _DraftKvState,
        current_tree: TreeDraftResult,
    ) -> Dict[SSDCacheKey, SSDCacheEntry]:
        """Fork every accepted length and the top-F likely recovery tokens."""

        started = time.perf_counter()
        path = current_tree.token_ids
        expected_parent = [-1] + list(range(max(0, len(path) - 1)))
        if current_tree.parent != expected_parent:
            raise ValueError("SSD cache population requires a single top-1 chain")

        logits = state.root_logits
        hidden = state.root_hidden
        kv = _as_tuple_kv(state.kv)
        device = logits.device
        cache: Dict[SSDCacheKey, SSDCacheEntry] = {}

        # At keep=m, logits/hidden/KV describe the prefix plus path[:m].
        for accepted_length in range(len(path) + 1):
            excluded = (
                int(path[accepted_length])
                if accepted_length < len(path)
                else None
            )
            recovery_tokens = _select_recovery_tokens(
                self.draft_model,
                logits,
                self.ssd_cfg.fan_out,
                excluded_token=excluded,
            )
            recovery = torch.tensor(
                recovery_tokens, device=device, dtype=torch.long
            ).unsqueeze(1)
            fan_out = len(recovery_tokens)
            branch_logits, branch_hidden, branch_kv = self.draft_model.step(
                recovery,
                hidden.expand(fan_out, -1),
                _repeat_kv(kv, fan_out),
                state.prefix_len - 1 + accepted_length,
            )
            branch_key, branch_value = _as_tuple_kv(branch_kv)
            for branch_idx, recovery_token in enumerate(recovery_tokens):
                branch_state = _DraftKvState(
                    kv=(
                        branch_key[branch_idx : branch_idx + 1],
                        branch_value[branch_idx : branch_idx + 1],
                    ),
                    prefix_len=state.prefix_len + accepted_length + 1,
                    root_logits=branch_logits[branch_idx : branch_idx + 1],
                    root_hidden=branch_hidden[branch_idx : branch_idx + 1],
                )
                cache[SSDCacheKey(accepted_length, recovery_token)] = SSDCacheEntry(
                    draft_state=branch_state,
                    draft_tree=self._tree_from_state(branch_state, device),
                )

            if accepted_length < len(path):
                token = torch.tensor(
                    [[path[accepted_length]]], device=device, dtype=torch.long
                )
                logits, hidden, kv = self.draft_model.step(
                    token,
                    hidden,
                    kv,
                    state.prefix_len - 1 + accepted_length,
                )
                kv = _as_tuple_kv(kv)

        if device.type == "cuda" and self._ssd_stream is not None:
            self._ssd_stream.synchronize()
        self.ssd_populate_seconds += time.perf_counter() - started
        self.n_ssd_forked_branches += len(cache)
        return cache

    def _submit_cache_population(
        self,
        state: _DraftKvState,
        current_tree: TreeDraftResult,
    ) -> Future | _CompletedFuture:
        if self._executor is None or self._ssd_stream is None:
            return _CompletedFuture(self._populate_speculation_cache(state, current_tree))

        device = state.root_logits.device
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))

        def work() -> Dict[SSDCacheKey, SSDCacheEntry]:
            with torch.cuda.device(device), torch.cuda.stream(self._ssd_stream):
                self._ssd_stream.wait_event(ready)
                return self._populate_speculation_cache(state, current_tree)

        return self._executor.submit(work)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate greedily with SSD pre-drafting around EAGLE3Pro verification."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("EAGLE3Pro SSD currently supports input shape [1, sequence]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if max_new_tokens == 0:
            return input_ids.clone()
        if self._closed:
            raise RuntimeError("Eagle3ProSSDGenerator is closed")

        device = input_ids.device
        out_ids = input_ids.clone()
        new_tokens = 0

        attention_mask = torch.ones_like(out_ids, device=device)
        target_out, acts_all = forward_target_selected_eagle_acts(
            self.target_m,
            self.eagle_layers,
            last_token_only=False,
            input_ids=out_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        past = target_out.past_key_values
        if getattr(self.target_m, "_eagle3pro_flash_kv_enabled", False):
            past = Eagle3FlashKVCache.from_dynamic(
                past,
                max_cache_len=(
                    int(out_ids.shape[1])
                    + int(max_new_tokens)
                    + int(self.cfg.num_steps)
                    + 1
                ),
            )

        draft_logits, draft_hidden, draft_kv = self.draft_model.prefill(
            out_ids, acts_all
        )
        # Unlike the base Pro path, keep tuple KV even for step=1: SSD must fork it.
        draft_state = _DraftKvState(
            kv=_as_tuple_kv(draft_kv),
            prefix_len=int(out_ids.shape[1]),
            root_logits=draft_logits,
            root_hidden=draft_hidden,
        )
        past.crop(int(out_ids.shape[1]) - 1)
        draft_tree = self._tree_from_state(draft_state, device)

        while new_tokens < max_new_tokens:
            if eos_token_id is not None and int(out_ids[0, -1]) == eos_token_id:
                break

            remaining = max_new_tokens - new_tokens
            # No following round can consume the cache when only one token remains.
            populate = (
                self._submit_cache_population(draft_state, draft_tree)
                if remaining > 1
                else None
            )

            append, target_shift_acts, matched = self._verify_chain_and_commit(
                out_ids[:, -1:],
                past,
                draft_tree,
                max_append_tokens=remaining,
                eos_token_id=eos_token_id,
            )
            proposed = len(draft_tree.token_ids)
            append_count = int(append.shape[1])
            will_stop = append_count >= remaining or (
                eos_token_id is not None and bool((append == eos_token_id).any())
            )

            next_tree: Optional[TreeDraftResult] = None
            if not will_stop:
                if populate is None:
                    raise RuntimeError("missing SSD population for a continuing sequence")
                wait_started = time.perf_counter()
                speculation_cache = populate.result()
                self.ssd_wait_seconds += time.perf_counter() - wait_started
                recovery_token = int(append[0, -1])
                entry = speculation_cache.get(
                    SSDCacheKey(matched, recovery_token)
                )
                if entry is not None:
                    draft_state = entry.draft_state
                    next_tree = entry.draft_tree
                    self.n_ssd_cache_hit += 1
                else:
                    self.n_ssd_cache_miss += 1
                    draft_logits, draft_hidden, draft_kv = self.draft_model.extend_tokens(
                        append,
                        target_shift_acts,
                        draft_state.kv,
                        draft_state.prefix_len - 1,
                    )
                    draft_state = _DraftKvState(
                        kv=_as_tuple_kv(draft_kv),
                        prefix_len=int(out_ids.shape[1]) + append_count,
                        root_logits=draft_logits,
                        root_hidden=draft_hidden,
                    )
                    next_tree = self._tree_from_state(draft_state, device)
            elif populate is not None:
                # Do not leave a worker touching model tensors after generate returns.
                wait_started = time.perf_counter()
                populate.result()
                self.ssd_wait_seconds += time.perf_counter() - wait_started

            out_ids = torch.cat([out_ids, append], dim=1)
            new_tokens += append_count
            self.n_rounds += 1
            self.n_accepted_tokens += append_count
            self.n_matched_draft_tokens += matched
            self.n_proposed_draft_tokens += proposed

            if will_stop:
                break
            if draft_state.prefix_len != int(out_ids.shape[1]):
                raise RuntimeError(
                    f"SSD draft prefix {draft_state.prefix_len} != target prefix {out_ids.shape[1]}"
                )
            if next_tree is None:
                raise RuntimeError("SSD did not prepare the next draft tree")
            draft_tree = next_tree

        return out_ids


__all__ = [
    "Eagle3ProSSDConfig",
    "Eagle3ProSSDGenerator",
    "Eagle3ProSSDStats",
    "SSDCacheEntry",
    "SSDCacheKey",
]
