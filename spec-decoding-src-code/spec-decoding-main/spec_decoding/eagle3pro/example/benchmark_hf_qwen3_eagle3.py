#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EAGLE3Pro 的可复现 HF greedy/EAGLE3 严格基准。

流程路径：加载同一 Qwen3 target → 可选 packed target 转换 → 构造真实 EAGLE3
draft → 分别强制生成相同 token 数 → 完整比较 token id 和长度。
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch


DEFAULT_TARGET = "Qwen/Qwen3-1.7B"
DEFAULT_DRAFT = "AngelSlim/Qwen3-1.7B_eagle3"
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
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="top-1 draft chain length; 1 matches the vLLM comparison setting",
    )
    parser.add_argument("--max-tree-nodes", type=int, default=None)
    parser.add_argument(
        "--tree-expand-mode",
        choices=("cumulative", "static"),
        default="cumulative",
    )
    parser.add_argument(
        "--verify-mode",
        choices=("full_model_tree", "reference_paths"),
        default="full_model_tree",
        help="reference_paths is a slow correctness oracle for multi-branch tree verify",
    )
    parser.add_argument(
        "--verify-attn-backend",
        choices=("auto", "eager", "triton_tree"),
        default="auto",
    )
    parser.add_argument(
        "--scheduler",
        choices=["eagle", "ssd"],
        default="eagle",
        help="eagle=原 EAGLE3Pro；ssd=并发预生成下一轮 recovery 分支",
    )
    parser.add_argument(
        "--ssd-fan-out",
        type=int,
        default=3,
        help="SSD 为每个可能接受长度预生成的 recovery token 数",
    )
    parser.add_argument(
        "--ssd-no-overlap",
        action="store_true",
        help="保留 SSD 缓存语义但串行执行，用于量化并发本身的收益",
    )
    parser.add_argument(
        "--benchmark-order",
        choices=["eagle-first", "greedy-first"],
        default="eagle-first",
        help="默认先计时 EAGLE，避免先跑长 greedy 后的笔记本 GPU 热降频偏差",
    )
    parser.add_argument(
        "--target-optimization",
        choices=[
            "none",
            "packed",
            "packed-vllm-silu",
            "packed-vllm",
            "packed-vllm-qk",
            "packed-vllm-flashkv",
        ],
        default="none",
        help="packed 合并投影；qk 再融合 QK norm/RoPE；flashkv 使用预分配 KV cache",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flash_attention_2"],
        default="sdpa",
        help="target 标准链式 attention 后端；树形 Triton verify 不受此项控制",
    )
    parser.add_argument(
        "--disable-gpu-resident-top1",
        action="store_true",
        help="A/B 对照：恢复单候选 topk().tolist() 和回传 CUDA 的旧路径",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="num_steps=S 时复用 Target[1,S+1] 与 Draft[1,1..S+1] decode Graph",
    )
    parser.add_argument(
        "--cuda-graph-max-seq-len",
        type=int,
        default=1024,
        help="Graph 模式的持久 KV workspace 容量",
    )
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--compare-json",
        default="",
        help="可选：与另一轮 JSON 的 greedy/eagle token id 做完整比较",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="默认只读本地 HF cache；显式指定后才允许下载",
    )
    args = parser.parse_args()

    if args.cuda_graph and args.target_optimization != "packed-vllm-flashkv":
        parser.error("--cuda-graph requires --target-optimization packed-vllm-flashkv")
    if args.cuda_graph and args.scheduler != "eagle":
        parser.error("--cuda-graph currently supports --scheduler eagle")
    if args.cuda_graph and args.disable_gpu_resident_top1:
        parser.error("--cuda-graph requires GPU-resident top-1")
    if args.cuda_graph and args.topk != 1:
        parser.error("--cuda-graph currently requires --topk 1")
    if args.num_prompts < 1:
        parser.error("--num-prompts must be >= 1")
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")

    if not torch.cuda.is_available():
        raise RuntimeError("this throughput benchmark requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from spec_decoding.eagle3pro import (
        Eagle3ProSSDConfig,
        Eagle3ProSSDGenerator,
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
        enable_qwen3_packed_projections,
        enable_vllm_fused_kernels,
        make_eagle3_sgl_draft_topk_fn,
    )
    from spec_decoding.eagle3pro.example.benchmark_cases import select_prompts

    local_only = not args.allow_download
    prompts = select_prompts(args.num_prompts)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target, local_files_only=local_only
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=torch.float16,
        local_files_only=local_only,
    ).to(device).eval()
    target.config._attn_implementation = args.attention_backend
    # EAGLE3PRO: 显式记录并设置 backend，避免依赖 Transformers 环境默认值。

    optimization_report: dict[str, int] = {}
    if args.target_optimization != "none":
        optimization_report.update(enable_qwen3_packed_projections(target))
        # EAGLE3PRO: target 先完成权重打包；vLLM 融合标记要等 draft 创建后一起设置。

    config_path = _repo_file(
        args.draft, "config.json", local_files_only=local_only
    )
    weights_path = _repo_file(
        args.draft, "pytorch_model.bin", local_files_only=local_only
    )
    with open(config_path, encoding="utf-8") as handle:
        draft_config = json.load(handle)

    max_tree_nodes = args.max_tree_nodes
    if max_tree_nodes is None:
        max_tree_nodes = args.num_steps if args.topk == 1 else 64
    config = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=max_tree_nodes,
        tree_expand_mode=args.tree_expand_mode,
        verify_mode=args.verify_mode,
        verify_attn_backend=args.verify_attn_backend,
        gpu_resident_top1=not args.disable_gpu_resident_top1,
        cuda_graph=args.cuda_graph,
        cuda_graph_max_seq_len=args.cuda_graph_max_seq_len,
    )
    # EAGLE3PRO: top-1 固定触发单 target-forward 快路径；step-1 与 vLLM 对比口径一致。
    eagle_layers = config.resolve_eagle_layers(target.config.num_hidden_layers)
    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        num_layers=1,
        draft_vocab_size=int(draft_config["draft_vocab_size"]),
        draft_weights_path=weights_path,
        device=device,
        dtype=torch.float16,
    )
    if args.target_optimization in (
        "packed-vllm-silu",
        "packed-vllm",
        "packed-vllm-qk",
        "packed-vllm-flashkv",
    ):
        optimization_report.update(
            enable_vllm_fused_kernels(
                target,
                draft,
                rms_norm=args.target_optimization
                in (
                    "packed-vllm",
                    "packed-vllm-qk",
                    "packed-vllm-flashkv",
                ),
                qk_norm_rope=args.target_optimization
                in (
                    "packed-vllm-qk",
                    "packed-vllm-flashkv",
                ),
                flash_kv_cache=args.target_optimization
                == "packed-vllm-flashkv",
            )
        )
    if args.target_optimization != "none":
        gc.collect()
        torch.cuda.empty_cache()
        # EAGLE3PRO: 所有结构转换都在 generator 创建前结束，正式计时不含打包/释放缓存。
    generator_cls = (
        Eagle3ProSSDGenerator if args.scheduler == "ssd" else Eagle3SglGenerator
    )
    generator_kwargs = {}
    if args.scheduler == "ssd":
        generator_kwargs["ssd_cfg"] = Eagle3ProSSDConfig(
            fan_out=args.ssd_fan_out,
            overlap=not args.ssd_no_overlap,
        )
    generator = generator_cls(
        target,
        config,
        make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk),
        draft_model=draft,
        autoregressive_draft=True,
        **generator_kwargs,
    )

    generation_config = GenerationConfig.from_model_config(target.config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.eos_token_id = None
    generation_config.pad_token_id = tokenizer.pad_token_id or 0

    warmup = tokenizer("Warm up the model.", return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        target.generate(
            warmup,
            max_new_tokens=args.warmup_tokens,
            generation_config=generation_config,
        )
        generator.generate(warmup, max_new_tokens=args.warmup_tokens, eos_token_id=None)
    # EAGLE3PRO: 两条路径都预热，正式计时不混入首次 CUDA/allocator 初始化。
    torch.cuda.reset_peak_memory_stats(device)
    # EAGLE3PRO: 峰值从模型常驻显存开始统计，报告包含正式 greedy/EAGLE 两段的真实上限。
    ssd_before = None
    if args.scheduler == "ssd":
        ssd_before = generator.ssd_stats

    def measure_greedy() -> tuple[float, list[list[int]]]:
        token_ids: list[list[int]] = []
        _sync()
        start = time.perf_counter()
        with torch.inference_mode():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                output_ids = target.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    generation_config=generation_config,
                )
                token_ids.append(output_ids[0, input_ids.shape[1] :].tolist())
        _sync()
        return time.perf_counter() - start, token_ids

    def measure_eagle() -> tuple[float, list[list[int]], int, int, int, int]:
        rounds_before = generator.n_rounds
        matched_before = generator.n_matched_draft_tokens
        proposed_before = generator.n_proposed_draft_tokens
        corrections_before = generator.n_tree_causal_corrections
        token_ids: list[list[int]] = []
        _sync()
        start = time.perf_counter()
        with torch.inference_mode():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                output_ids = generator.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=None,
                )
                token_ids.append(output_ids[0, input_ids.shape[1] :].tolist())
        _sync()
        return (
            time.perf_counter() - start,
            token_ids,
            generator.n_rounds - rounds_before,
            generator.n_matched_draft_tokens - matched_before,
            generator.n_proposed_draft_tokens - proposed_before,
            generator.n_tree_causal_corrections - corrections_before,
        )

    eagle_trials: list[tuple[float, list[list[int]], int, int, int, int]] = []
    if args.benchmark_order == "eagle-first":
        eagle_trials = [measure_eagle() for _ in range(args.repetitions)]
        greedy_seconds, greedy_token_ids = measure_greedy()
    else:
        greedy_seconds, greedy_token_ids = measure_greedy()
        eagle_trials = [measure_eagle() for _ in range(args.repetitions)]
    eagle_token_ids = eagle_trials[0][1]
    if any(trial[1] != eagle_token_ids for trial in eagle_trials[1:]):
        raise AssertionError("EAGLE deterministic token ids changed between repetitions")
    eagle_seconds = statistics.median(trial[0] for trial in eagle_trials)
    representative = min(eagle_trials, key=lambda trial: abs(trial[0] - eagle_seconds))
    _, _, rounds, matched, proposed, tree_corrections = representative
    # EAGLE3PRO: 把计时顺序写入报告，便于解释笔记本 GPU 温控引起的波动。
    total_tokens = sum(map(len, eagle_token_ids))
    exact_match = [
        actual == expected
        for actual, expected in zip(eagle_token_ids, greedy_token_ids)
    ]
    result = {
        "package": "spec_decoding.eagle3pro",
        "scheduler": args.scheduler,
        "target_optimization": args.target_optimization,
        "attention_backend": args.attention_backend,
        "benchmark_order": args.benchmark_order,
        "repetitions": args.repetitions,
        "eagle_trial_seconds": [trial[0] for trial in eagle_trials],
        "eagle_trial_tokens_per_second": [
            sum(map(len, trial[1])) / trial[0] for trial in eagle_trials
        ],
        "num_prompts": len(prompts),
        "execution_batch_size": 1,
        "execution_mode": "sequential_single_request",
        "topk": args.topk,
        "num_steps": args.num_steps,
        "max_tree_nodes": max_tree_nodes,
        "tree_expand_mode": args.tree_expand_mode,
        "verify_mode": args.verify_mode,
        "verify_attn_backend": args.verify_attn_backend,
        "gpu_resident_top1": not args.disable_gpu_resident_top1,
        "cuda_graph": args.cuda_graph,
        "cuda_graph_stats": generator.cuda_graph_stats,
        "optimization_report": optimization_report,
        "total_tokens": total_tokens,
        "greedy_seconds": greedy_seconds,
        "greedy_tps": total_tokens / greedy_seconds,
        "eagle_seconds": eagle_seconds,
        "eagle_tps": total_tokens / eagle_seconds,
        "speedup_vs_hf_greedy": greedy_seconds / eagle_seconds,
        "gpu_name": torch.cuda.get_device_name(device),
        "memory_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "rounds": rounds,
        "matched_draft_tokens": matched,
        "proposed_draft_tokens": proposed,
        "tree_causal_corrections": tree_corrections,
        "acceptance_length": total_tokens / max(1, rounds),
        "draft_acceptance_rate": matched / max(1, proposed),
        "lengths": [len(ids) for ids in eagle_token_ids],
        "exact_match": exact_match,
        "greedy_token_ids": greedy_token_ids,
        "eagle_token_ids": eagle_token_ids,
    }
    if args.scheduler == "ssd":
        ssd_after = generator.ssd_stats
        cache_hits = ssd_after.cache_hits - ssd_before.cache_hits
        cache_misses = ssd_after.cache_misses - ssd_before.cache_misses
        result["ssd"] = {
            "fan_out": args.ssd_fan_out,
            "overlap": not args.ssd_no_overlap,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / max(1, cache_hits + cache_misses),
            "forked_branches": (
                ssd_after.forked_branches - ssd_before.forked_branches
            ),
            "populate_seconds": (
                ssd_after.populate_seconds - ssd_before.populate_seconds
            ),
            "wait_seconds": ssd_after.wait_seconds - ssd_before.wait_seconds,
        }

    if args.compare_json:
        with open(args.compare_json, encoding="utf-8") as handle:
            reference = json.load(handle)
        result["reference_json"] = str(Path(args.compare_json).resolve())
        result["reference_greedy_exact"] = (
            reference.get("greedy_token_ids") == greedy_token_ids
        )
        result["reference_eagle_exact"] = (
            reference.get("eagle_token_ids") == eagle_token_ids
        )
        # EAGLE3PRO: 比较完整二维 token 数组，长度差异也会判为 False。

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"[eagle3pro] report -> {output_path.resolve()}")

    if not all(exact_match):
        raise AssertionError("EAGLE3Pro output diverged from same-engine greedy")
    if args.compare_json and not (
        result["reference_greedy_exact"] and result["reference_eagle_exact"]
    ):
        raise AssertionError("optimized output diverged from reference JSON")
    if args.scheduler == "ssd":
        generator.close()


if __name__ == "__main__":
    main()
