#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fair Qwen3 benchmark for the original, profile-opti, and Pro packages.

Each invocation loads one implementation, warms up HF greedy and EAGLE3, then
generates the same three fixed prompts.  Correctness requires equal lengths and
equal token ids for every prompt; prefix-only matches are not accepted.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import time
from pathlib import Path

import torch


PROMPTS = [
    "请用中文解释投机解码为什么能够保持输出无损，并给出一个简单例子。",
    "Write a short Python function that checks whether a string is a palindrome, then explain it.",
    "Solve step by step: A shop discounts an 800 yuan item by 15%, then applies a 5% coupon. What is the final price?",
]

PACKAGE_BY_IMPLEMENTATION = {
    "original": "spec_decoding.eagle3_sgl",
    "profile-opti": "spec_decoding.eagle3_sgl_profile_opti",
    "pro": "spec_decoding.eagle3pro",
}


def _repo_file(repo_or_dir: str, filename: str, *, local_files_only: bool) -> str:
    local = Path(repo_or_dir).expanduser()
    if local.is_dir():
        path = local / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_or_dir, filename, local_files_only=local_files_only
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=sorted(PACKAGE_BY_IMPLEMENTATION),
        required=True,
    )
    parser.add_argument("--target", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft", default="AngelSlim/Qwen3-1.7B_eagle3")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--max-tree-nodes", type=int, default=1)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="profile-opti only: enable its StaticCache/CUDA Graph verify path",
    )
    parser.add_argument(
        "--static-eager",
        action="store_true",
        help="profile-opti only: use StaticCache path without CUDA graph capture",
    )
    parser.add_argument(
        "--keep-replay",
        action="store_true",
        help="profile-opti only: keep Step5 target replay as the Step6 A/B baseline",
    )
    parser.add_argument(
        "--pro-optimization",
        choices=["none", "packed-vllm-flashkv"],
        default="packed-vllm-flashkv",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this throughput benchmark requires CUDA")
    if (args.cuda_graph or args.static_eager) and args.implementation != "profile-opti":
        raise ValueError("--cuda-graph/--static-eager are only valid for profile-opti")
    if args.keep_replay and args.implementation != "profile-opti":
        raise ValueError("--keep-replay is only valid for profile-opti")
    if args.cuda_graph and args.static_eager:
        raise ValueError("choose either --cuda-graph or --static-eager")

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    package_name = PACKAGE_BY_IMPLEMENTATION[args.implementation]
    package = importlib.import_module(package_name)
    local_only = not args.allow_download
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target, local_files_only=local_only
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=torch.float16,
        local_files_only=local_only,
    ).to(device).eval()
    target.config._attn_implementation = "sdpa"

    optimization_report: dict[str, int] = {}
    if args.implementation == "pro" and args.pro_optimization != "none":
        optimization_report.update(package.enable_qwen3_packed_projections(target))

    config_path = _repo_file(
        args.draft, "config.json", local_files_only=local_only
    )
    weights_path = _repo_file(
        args.draft, "pytorch_model.bin", local_files_only=local_only
    )
    with open(config_path, encoding="utf-8") as handle:
        draft_config = json.load(handle)

    config_kwargs = dict(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        verify_mode="full_model_tree",
        verify_attn_backend="auto",
    )
    if args.implementation == "profile-opti":
        config_kwargs["decode_cuda_graph"] = args.cuda_graph or args.static_eager
        config_kwargs["graph_capture"] = not args.static_eager
        config_kwargs["eliminate_replay"] = not args.keep_replay
    config = package.Eagle3SglConfig(**config_kwargs)
    eagle_layers = config.resolve_eagle_layers(target.config.num_hidden_layers)
    draft = package.build_eagle3_sgl_draft(
        target,
        eagle_layers,
        num_layers=1,
        draft_vocab_size=int(draft_config["draft_vocab_size"]),
        draft_weights_path=weights_path,
        device=device,
        dtype=torch.float16,
    )
    if args.implementation == "pro" and args.pro_optimization != "none":
        optimization_report.update(
            package.enable_vllm_fused_kernels(
                target,
                draft,
                rms_norm=True,
                qk_norm_rope=True,
                flash_kv_cache=True,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()

    generator = package.Eagle3SglGenerator(
        target,
        config,
        package.make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk),
        draft_model=draft,
        autoregressive_draft=True,
    )

    generation_config = GenerationConfig.from_model_config(target.config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.eos_token_id = None
    generation_config.pad_token_id = tokenizer.pad_token_id or 0

    warmup_ids = tokenizer(PROMPTS[-1], return_tensors="pt")["input_ids"].to(device)
    graph_warmup_tokens = (
        args.max_new_tokens if args.cuda_graph else args.warmup_tokens
    )
    with torch.inference_mode():
        target.generate(
            warmup_ids,
            max_new_tokens=args.warmup_tokens,
            generation_config=generation_config,
        )
        generator.generate(
            warmup_ids,
            max_new_tokens=graph_warmup_tokens,
            eos_token_id=None,
        )
    torch.cuda.reset_peak_memory_stats(device)

    rounds_before = generator.n_rounds
    matched_before = generator.n_matched_draft_tokens
    proposed_before = generator.n_proposed_draft_tokens
    eagle_token_ids: list[list[int]] = []
    _sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for prompt in PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            output_ids = generator.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=None,
            )
            eagle_token_ids.append(output_ids[0, input_ids.shape[1] :].tolist())
    _sync()
    eagle_seconds = time.perf_counter() - start

    greedy_token_ids: list[list[int]] = []
    _sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for prompt in PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            output_ids = target.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                generation_config=generation_config,
            )
            greedy_token_ids.append(output_ids[0, input_ids.shape[1] :].tolist())
    _sync()
    greedy_seconds = time.perf_counter() - start

    rounds = generator.n_rounds - rounds_before
    matched = generator.n_matched_draft_tokens - matched_before
    proposed = generator.n_proposed_draft_tokens - proposed_before
    exact_match = [
        actual == expected
        for actual, expected in zip(eagle_token_ids, greedy_token_ids)
    ]
    total_tokens = sum(map(len, eagle_token_ids))
    result = {
        "implementation": args.implementation,
        "package": package_name,
        "target": args.target,
        "draft": args.draft,
        "topk": args.topk,
        "num_steps": args.num_steps,
        "max_tree_nodes": args.max_tree_nodes,
        "cuda_graph": args.cuda_graph,
        "static_eager": args.static_eager,
        "eliminate_replay": (
            not args.keep_replay if args.implementation == "profile-opti" else None
        ),
        "pro_optimization": (
            args.pro_optimization if args.implementation == "pro" else None
        ),
        "optimization_report": optimization_report,
        "total_tokens": total_tokens,
        "eagle_seconds": eagle_seconds,
        "eagle_tps": total_tokens / eagle_seconds,
        "greedy_seconds": greedy_seconds,
        "greedy_tps": total_tokens / greedy_seconds,
        "speedup_vs_hf_greedy": greedy_seconds / eagle_seconds,
        "rounds": rounds,
        "acceptance_length": total_tokens / max(1, rounds),
        "matched_draft_tokens": matched,
        "proposed_draft_tokens": proposed,
        "draft_acceptance_rate": matched / max(1, proposed),
        "lengths": [len(ids) for ids in eagle_token_ids],
        "exact_match": exact_match,
        "gpu_name": torch.cuda.get_device_name(device),
        "memory_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "greedy_token_ids": greedy_token_ids,
        "eagle_token_ids": eagle_token_ids,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"[qwen3-compare] report -> {output_path.resolve()}")
    if not all(exact_match):
        raise AssertionError(
            f"{args.implementation} output length or token ids diverged from greedy"
        )


if __name__ == "__main__":
    main()
