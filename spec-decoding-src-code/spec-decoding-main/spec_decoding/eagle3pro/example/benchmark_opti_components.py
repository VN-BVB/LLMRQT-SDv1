#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated old/new benchmarks for the Profile-Opti code adopted by EAGLE3Pro.

The legacy functions are imported from ``eagle3_sgl_llama``.  Production Pro
code is not switched back and forth, so this benchmark cannot accidentally
leave a legacy path enabled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


DEFAULT_TARGET = "Qwen/Qwen3-1.7B"
DEFAULT_DRAFT = "AngelSlim/Qwen3-1.7B_eagle3"
DEFAULT_PROMPT = "请用中文解释投机解码为什么能够保持输出无损，并给出一个简单例子。"


def _repo_file(repo_or_dir: str, filename: str) -> str:
    local = Path(repo_or_dir).expanduser()
    if local.is_dir():
        return str(local / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_or_dir, filename, local_files_only=True)


def _sync() -> None:
    torch.cuda.synchronize()


def _paired_times(
    old_fn: Callable[[], object],
    new_fn: Callable[[], object],
    *,
    warmup: int,
    repetitions: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        old_fn()
        new_fn()
    _sync()
    old_ms: list[float] = []
    new_ms: list[float] = []
    for rep in range(repetitions):
        order = ((old_fn, old_ms), (new_fn, new_ms))
        if rep % 2:
            order = tuple(reversed(order))
        for fn, samples in order:
            _sync()
            started = time.perf_counter()
            fn()
            _sync()
            samples.append((time.perf_counter() - started) * 1000.0)
    return old_ms, new_ms


def _timing_summary(old_ms: list[float], new_ms: list[float]) -> dict[str, object]:
    old_median = statistics.median(old_ms)
    new_median = statistics.median(new_ms)
    return {
        "old_ms": old_ms,
        "new_ms": new_ms,
        "old_median_ms": old_median,
        "new_median_ms": new_median,
        "speedup": old_median / new_median,
        "latency_reduction_percent": (1.0 - new_median / old_median) * 100.0,
    }


def _static_forest_parent(topk: int, nodes: int) -> list[int]:
    parent = [-1] * min(topk, nodes)
    while len(parent) < nodes:
        parent.append((len(parent) - topk) // topk)
    return parent


def _leaves(parent: list[int]) -> list[int]:
    has_child = [False] * len(parent)
    for value in parent:
        if value >= 0:
            has_child[value] = True
    return [i for i, value in enumerate(has_child) if not value]


def _path(parent: list[int], leaf: int) -> list[int]:
    result = []
    while leaf >= 0:
        result.append(leaf)
        leaf = parent[leaf]
    return list(reversed(result))


def _legacy_accept_counts(
    node_logits: torch.Tensor,
    recovery_logits: torch.Tensor,
    parent: list[int],
    token_ids: list[int],
    leaves: list[int],
) -> tuple[int, int]:
    argmax_calls = 0
    d2h_calls = 0
    for leaf in leaves:
        indices = _path(parent, leaf)
        matched = 0
        for offset, node in enumerate(indices):
            source = recovery_logits if offset == 0 else node_logits[indices[offset - 1]]
            prediction = int(source.argmax(dim=-1).item())
            argmax_calls += 1
            d2h_calls += 1
            if prediction != token_ids[node]:
                break
            matched += 1
        if matched == len(indices):
            node_logits[indices[-1]].argmax(dim=-1).item()
            argmax_calls += 1
            d2h_calls += 1
    return argmax_calls, d2h_calls


class _CountingDraft:
    def __init__(self, draft) -> None:
        self.draft = draft
        self.step_calls = 0
        self.processed_rows = 0

    def step(self, token_ids, *args, **kwargs):
        self.step_calls += 1
        self.processed_rows += int(token_ids.shape[0])
        return self.draft.step(token_ids, *args, **kwargs)

    def topk_target_ids(self, *args, **kwargs):
        return self.draft.topk_target_ids(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--max-tree-nodes", type=int, default=64)
    parser.add_argument("--past-len", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from spec_decoding.eagle3_sgl_llama.tree_draft import (
        expand_draft_tree_ar as expand_tree_old,
    )
    from spec_decoding.eagle3_sgl_llama.tree_verify_full import (
        _select_append_root_child as accept_old,
    )
    from spec_decoding.eagle3_sgl_llama.tree_verify_mask import (
        build_tree_cross_attn_bias_with_prefix as mask_old,
    )
    from spec_decoding.eagle3pro.config import Eagle3SglConfig
    from spec_decoding.eagle3pro.draft_model import build_eagle3_sgl_draft
    from spec_decoding.eagle3pro.hidden import (
        forward_target_selected_eagle_acts,
        select_eagle_acts_from_hidden_states,
    )
    from spec_decoding.eagle3pro.target_optim import (
        enable_qwen3_packed_projections,
        enable_vllm_fused_kernels,
    )
    from spec_decoding.eagle3pro.tree_draft import expand_draft_tree_ar as expand_tree_new
    from spec_decoding.eagle3pro.tree_verify_full import (
        _select_append_root_child as accept_new,
    )
    from spec_decoding.eagle3pro.tree_verify_mask import (
        TreeVerifyLayout,
        build_tree_cross_attn_bias_with_prefix as mask_new,
    )

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    enable_qwen3_packed_projections(target)

    cfg = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        tree_expand_mode="static",
        verify_mode="full_model_tree",
        verify_attn_backend="auto",
    )
    eagle_layers = cfg.resolve_eagle_layers(target.config.num_hidden_layers)
    draft_config = json.loads(Path(_repo_file(args.draft, "config.json")).read_text())
    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        num_layers=1,
        draft_vocab_size=int(draft_config["draft_vocab_size"]),
        draft_weights_path=_repo_file(args.draft, "pytorch_model.bin"),
        device=device,
        dtype=torch.float16,
    )
    enable_vllm_fused_kernels(
        target,
        draft,
        rms_norm=True,
        qk_norm_rope=True,
        flash_kv_cache=False,
    )

    input_ids = tokenizer(args.target + " " + DEFAULT_PROMPT, return_tensors="pt")[
        "input_ids"
    ].to(device)
    attention_mask = torch.ones_like(input_ids)

    # Step 3: old all-hidden collection versus the selected-layer pre-hooks.
    def hidden_old():
        outputs = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return select_eagle_acts_from_hidden_states(
            outputs.hidden_states, eagle_layers, last_token_only=False
        )

    def hidden_new():
        _, acts = forward_target_selected_eagle_acts(
            target,
            eagle_layers,
            last_token_only=False,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return acts

    hidden_expected = hidden_old()
    hidden_actual = hidden_new()
    hidden_equal = torch.equal(hidden_expected, hidden_actual)
    hidden_old_ms, hidden_new_ms = _paired_times(
        hidden_old, hidden_new, warmup=2, repetitions=args.repetitions
    )
    hidden_result = _timing_summary(hidden_old_ms, hidden_new_ms)
    hidden_result.update(
        {
            "exact_equal": hidden_equal,
            "max_abs_diff": float((hidden_expected - hidden_actual).abs().max().item()),
            "old_hidden_tensor_references": int(target.config.num_hidden_layers) + 1,
            "new_captured_tensor_references": len(eagle_layers),
        }
    )

    # Prepare one real Qwen draft state for Step 2.
    _, acts_all = forward_target_selected_eagle_acts(
        target,
        eagle_layers,
        last_token_only=False,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    root_logits, root_hidden, root_kv = draft.prefill(input_ids, acts_all)
    tree_args = (cfg, draft, root_logits, root_hidden, root_kv, input_ids.shape[1] - 1)

    tree_expected = expand_tree_old(*tree_args, device=device)
    tree_actual = expand_tree_new(*tree_args, device=device)
    tree_equal = (
        tree_expected.token_ids == tree_actual.token_ids
        and tree_expected.parent == tree_actual.parent
        and tree_expected.node_depth == tree_actual.node_depth
        and tree_expected.bfs_index == tree_actual.bfs_index
    )
    tree_old_ms, tree_new_ms = _paired_times(
        lambda: expand_tree_old(*tree_args, device=device),
        lambda: expand_tree_new(*tree_args, device=device),
        warmup=2,
        repetitions=args.repetitions,
    )
    old_counter = _CountingDraft(draft)
    new_counter = _CountingDraft(draft)
    expand_tree_old(
        cfg, old_counter, root_logits, root_hidden, root_kv, input_ids.shape[1] - 1,
        device=device,
    )
    expand_tree_new(
        cfg, new_counter, root_logits, root_hidden, root_kv, input_ids.shape[1] - 1,
        device=device,
    )
    tree_result = _timing_summary(tree_old_ms, tree_new_ms)
    tree_result.update(
        {
            "exact_tree_equal": tree_equal,
            "old_step_calls": old_counter.step_calls,
            "new_step_calls": new_counter.step_calls,
            "old_processed_rows": old_counter.processed_rows,
            "new_processed_rows": new_counter.processed_rows,
        }
    )

    # Step 1: a 64-node static forest and full target-vocabulary logits.
    parent = _static_forest_parent(args.topk, args.max_tree_nodes)
    leaves = _leaves(parent)
    generator = torch.Generator(device=device).manual_seed(20260912)
    vocab_size = int(target.config.vocab_size)
    node_logits = torch.randn(
        args.max_tree_nodes,
        vocab_size,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    recovery_logits = torch.randn(
        vocab_size, device=device, dtype=torch.float16, generator=generator
    )
    token_ids = [int(x) for x in torch.randint(
        0, vocab_size, (args.max_tree_nodes,), device=device, generator=generator
    ).tolist()]
    accept_expected = accept_old(node_logits, recovery_logits, parent, token_ids, leaves)
    accept_actual = accept_new(node_logits, recovery_logits, parent, token_ids, leaves)
    accept_old_ms, accept_new_ms = _paired_times(
        lambda: accept_old(node_logits, recovery_logits, parent, token_ids, leaves),
        lambda: accept_new(node_logits, recovery_logits, parent, token_ids, leaves),
        warmup=2,
        repetitions=args.repetitions,
    )
    old_argmax, old_d2h = _legacy_accept_counts(
        node_logits, recovery_logits, parent, token_ids, leaves
    )
    accept_result = _timing_summary(accept_old_ms, accept_new_ms)
    accept_result.update(
        {
            "exact_result_equal": accept_expected == accept_actual,
            "leaves": len(leaves),
            "old_argmax_calls": old_argmax,
            "new_argmax_calls": 2,
            "old_d2h_calls": old_d2h,
            "new_d2h_calls": 1,
        }
    )

    # Step 4: hot cross-mask shape used by the Triton tree verifier.
    layout = TreeVerifyLayout(list(range(len(parent))), parent, list(range(len(parent))))
    mask_expected = mask_old(
        layout, args.past_len, device=device, dtype=torch.float32
    )
    mask_actual = mask_new(
        layout, args.past_len, device=device, dtype=torch.float32
    )
    mask_old_ms, mask_new_ms = _paired_times(
        lambda: mask_old(layout, args.past_len, device=device, dtype=torch.float32),
        lambda: mask_new(layout, args.past_len, device=device, dtype=torch.float32),
        warmup=1,
        repetitions=args.repetitions,
    )
    ancestor_entries = sum(len(_path(parent, node)) for node in range(len(parent)))
    mask_result = _timing_summary(mask_old_ms, mask_new_ms)
    mask_result.update(
        {
            "exact_mask_equal": bool(torch.equal(mask_expected, mask_actual)),
            "shape": list(mask_actual.shape),
            "old_scalar_cuda_writes": len(parent) * args.past_len + ancestor_entries,
            "new_vectorized_scatter_rounds": max(
                len(_path(parent, node)) - 1 for node in range(len(parent))
            ),
        }
    )

    result = {
        "gpu": torch.cuda.get_device_name(device),
        "target": args.target,
        "draft": args.draft,
        "dtype": "float16",
        "tree_config": {
            "topk": args.topk,
            "num_steps": args.num_steps,
            "max_tree_nodes": args.max_tree_nodes,
            "past_len": args.past_len,
        },
        "timing": "wall clock with torch.cuda.synchronize; alternating old/new; median reported",
        "repetitions": args.repetitions,
        "step1_accept_selection": accept_result,
        "step2_depth_batched_draft": tree_result,
        "step3_selected_layer_capture": hidden_result,
        "step4_ancestor_mask": mask_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
