#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end A/B for the three multi-branch Opti ports in EAGLE3Pro."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


TARGET = "Qwen/Qwen3-1.7B"
DRAFT = "AngelSlim/Qwen3-1.7B_eagle3"
PROMPTS = (
    "请用中文解释投机解码为什么能够保持输出无损，并给出一个简单例子。",
    "Write a short Python function that checks whether a string is a palindrome, then explain it.",
    "Solve step by step: A shop discounts an 800 yuan item by 15%, then applies a 5% coupon. What is the final price?",
)


def _repo_file(repo_or_dir: str, filename: str) -> str:
    local = Path(repo_or_dir).expanduser()
    if local.is_dir():
        return str(local / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_or_dir, filename, local_files_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--max-tree-nodes", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    import spec_decoding.eagle3pro.runner as runner_module
    import spec_decoding.eagle3pro.tree_verify_attn as attn_module
    import spec_decoding.eagle3pro.tree_verify_full as full_module
    from spec_decoding.eagle3_sgl_llama.tree_draft import (
        expand_draft_tree_ar as legacy_expand,
    )
    from spec_decoding.eagle3_sgl_llama.tree_verify_full import (
        _select_append_root_child as legacy_accept,
    )
    from spec_decoding.eagle3_sgl_llama.tree_verify_mask import (
        build_tree_cross_attn_bias_with_prefix as legacy_cross_mask,
    )
    from spec_decoding.eagle3pro import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
        enable_qwen3_packed_projections,
        enable_vllm_fused_kernels,
        make_eagle3_sgl_draft_topk_fn,
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
    draft_cfg = json.loads(Path(_repo_file(args.draft, "config.json")).read_text())
    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        num_layers=1,
        draft_vocab_size=int(draft_cfg["draft_vocab_size"]),
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
    generator = Eagle3SglGenerator(
        target,
        cfg,
        make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk),
        draft_model=draft,
        autoregressive_draft=True,
    )

    current = {
        "expand": runner_module.expand_draft_tree_ar,
        "accept": full_module._select_append_root_child,
        "attn_cross": attn_module.build_tree_cross_attn_bias_with_prefix,
        "full_cross": full_module.build_tree_cross_attn_bias_with_prefix,
        "runner_cross": runner_module.build_tree_cross_attn_bias_with_prefix,
    }

    def select_implementation(legacy: bool) -> None:
        runner_module.expand_draft_tree_ar = legacy_expand if legacy else current["expand"]
        full_module._select_append_root_child = legacy_accept if legacy else current["accept"]
        cross = legacy_cross_mask if legacy else current["attn_cross"]
        attn_module.build_tree_cross_attn_bias_with_prefix = cross
        full_module.build_tree_cross_attn_bias_with_prefix = (
            legacy_cross_mask if legacy else current["full_cross"]
        )
        runner_module.build_tree_cross_attn_bias_with_prefix = (
            legacy_cross_mask if legacy else current["runner_cross"]
        )

    generation_config = GenerationConfig.from_model_config(target.config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.eos_token_id = None
    generation_config.pad_token_id = tokenizer.pad_token_id or 0

    def run_eagle(legacy: bool, token_count: int) -> dict[str, object]:
        select_implementation(legacy)
        rounds_before = generator.n_rounds
        matched_before = generator.n_matched_draft_tokens
        proposed_before = generator.n_proposed_draft_tokens
        outputs = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                generated = generator.generate(
                    ids, max_new_tokens=token_count, eos_token_id=None
                )
                outputs.append(generated[0, ids.shape[1] :].tolist())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rounds = generator.n_rounds - rounds_before
        matched = generator.n_matched_draft_tokens - matched_before
        proposed = generator.n_proposed_draft_tokens - proposed_before
        total = sum(map(len, outputs))
        return {
            "elapsed_seconds": elapsed,
            "tokens_per_second": total / elapsed,
            "total_tokens": total,
            "rounds": rounds,
            "matched_draft_tokens": matched,
            "proposed_draft_tokens": proposed,
            "acceptance_length": total / rounds,
            "draft_acceptance_rate": matched / proposed,
            "token_ids": outputs,
        }

    def run_greedy() -> list[list[int]]:
        outputs = []
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                generated = target.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    generation_config=generation_config,
                )
                outputs.append(generated[0, ids.shape[1] :].tolist())
        return outputs

    # Both implementations get a short warm-up before the measured old -> new pair.
    run_eagle(True, 2)
    run_eagle(False, 2)
    old = run_eagle(True, args.max_new_tokens)
    new = run_eagle(False, args.max_new_tokens)
    select_implementation(False)
    greedy_ids = run_greedy()
    old["exact_vs_greedy"] = old["token_ids"] == greedy_ids
    new["exact_vs_greedy"] = new["token_ids"] == greedy_ids
    exact_old_new = old["token_ids"] == new["token_ids"]
    old.pop("token_ids")
    new.pop("token_ids")
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "target": args.target,
        "draft": args.draft,
        "tree_config": {
            "topk": args.topk,
            "num_steps": args.num_steps,
            "max_tree_nodes": args.max_tree_nodes,
            "tree_expand_mode": "static",
        },
        "prompts": len(PROMPTS),
        "tokens_per_prompt": args.max_new_tokens,
        "old_step1_step2_step4": old,
        "new_step1_step2_step4": new,
        "exact_old_new": exact_old_new,
        "tps_gain_percent": (
            new["tokens_per_second"] / old["tokens_per_second"] - 1.0
        ) * 100.0,
        "latency_reduction_percent": (
            1.0 - new["elapsed_seconds"] / old["elapsed_seconds"]
        ) * 100.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
