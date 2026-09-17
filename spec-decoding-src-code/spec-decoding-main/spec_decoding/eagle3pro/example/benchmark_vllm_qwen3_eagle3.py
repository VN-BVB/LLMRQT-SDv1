#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""在相同确定性提示词上分别测试 vLLM greedy / Qwen3 EAGLE3。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path


def _counter_values(llm) -> dict[str, float]:
    return {
        metric.name: float(metric.value)
        for metric in llm.get_metrics()
        if hasattr(metric, "value")
    }


def main() -> None:
    is_wsl = "microsoft" in platform.release().lower()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("greedy", "eagle3"), required=True)
    parser.add_argument("--target", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft", default="AngelSlim/Qwen3-1.7B_eagle3")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument(
        "--warmup-all-prompts",
        action="store_true",
        help="用三个不同内容但同类长度的 prompt 预热，覆盖正式批次的 speculative JIT shape",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="关闭 enforce_eager，让 vLLM 使用其默认 CUDA Graph 调度",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-json", type=Path)
    parser.add_argument(
        "--wsl-compat",
        action=argparse.BooleanOptionalAction,
        default=is_wsl,
        help="启用当前机器需要的 WSL 兼容环境变量",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")
    if args.num_prompts < 1:
        parser.error("--num-prompts must be >= 1")

    from spec_decoding.eagle3pro.example.benchmark_cases import select_prompts

    prompts = select_prompts(args.num_prompts)
    max_num_seqs = args.max_num_seqs or args.num_prompts

    if args.wsl_compat:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # 本机 WSL 无 UVA，且 FlashInfer sampler 的 JIT 缺 ninja；pickle IPC 仅限可信本地运行。

    from vllm import LLM, SamplingParams
    import torch

    speculative_config = None
    if args.mode == "eagle3":
        speculative_config = {
            "method": "eagle3",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        # 显式 method 与 draft 长度，避免仅凭仓库名猜测配置。

    llm = LLM(
        model=args.target,
        tokenizer=args.target,
        dtype="float16",
        seed=0,
        enforce_eager=not args.cuda_graph,
        max_model_len=1024,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.78,
        disable_log_stats=False,
        enable_prefix_caching=False,
        speculative_config=speculative_config,
    )
    # 8 GiB 4060 的保守配置；除 eager/Graph 开关外，greedy/EAGLE 使用相同 batch 和显存比例。

    warmup = SamplingParams(
        temperature=0, max_tokens=args.warmup_tokens, ignore_eos=True
    )
    warmup_prompts = (
        [f"Warm-up only. {prompt}" for prompt in prompts]
        if args.warmup_all_prompts
        else ["Warm up the model."]
    )
    llm.generate(warmup_prompts, warmup, use_tqdm=False)
    torch.cuda.reset_peak_memory_stats()
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=0,
    )
    trial_seconds: list[float] = []
    trial_token_ids: list[list[list[int]]] = []
    trial_deltas: list[dict[str, float]] = []
    for _ in range(args.repetitions):
        metrics_before = _counter_values(llm)
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        trial_seconds.append(time.perf_counter() - started)
        metrics_after = _counter_values(llm)
        trial_token_ids.append(
            [list(output.outputs[0].token_ids) for output in outputs]
        )
        trial_deltas.append(
            {
                name: value - metrics_before.get(name, 0.0)
                for name, value in metrics_after.items()
                if name.startswith("vllm:spec_decode_")
            }
        )

    token_ids = trial_token_ids[0]
    if any(ids != token_ids for ids in trial_token_ids[1:]):
        raise AssertionError("vLLM deterministic token ids changed between repetitions")
    elapsed = statistics.median(trial_seconds)
    total_tokens = sum(len(ids) for ids in token_ids)
    deltas = trial_deltas[0]
    drafts = deltas.get("vllm:spec_decode_num_drafts", 0.0)
    accepted = deltas.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    proposed = deltas.get("vllm:spec_decode_num_draft_tokens", 0.0)
    result = {
        "mode": args.mode,
        "num_prompts": len(prompts),
        "execution_batch_size": max_num_seqs,
        "execution_mode": "vllm_continuous_batch",
        "cuda_graph": args.cuda_graph,
        "enforce_eager": not args.cuda_graph,
        "elapsed_seconds": elapsed,
        "repetitions": args.repetitions,
        "trial_seconds": trial_seconds,
        "trial_tokens_per_second": [total_tokens / value for value in trial_seconds],
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed,
        "memory_allocated_mib": torch.cuda.memory_allocated() / (1024**2),
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "token_ids": token_ids,
        "spec_metrics": deltas,
        "acceptance_length": 1.0 + accepted / drafts if drafts else None,
        "acceptance_rate": accepted / proposed if proposed else None,
    }

    if args.compare_json is not None:
        reference = json.loads(args.compare_json.read_text(encoding="utf-8"))
        expected_ids = reference["token_ids"]
        result["exact_match_per_prompt"] = [
            actual == expected
            for actual, expected in zip(token_ids, expected_ids)
        ]
        result["exact_match"] = token_ids == expected_ids
        # 无损要求 token 数与每个 token id 都相同，不能只比较公共前缀。

    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {key: value for key, value in result.items() if key != "token_ids"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
