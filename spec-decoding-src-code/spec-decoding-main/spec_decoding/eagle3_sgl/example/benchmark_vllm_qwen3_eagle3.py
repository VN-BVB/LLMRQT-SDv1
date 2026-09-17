#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""在相同确定性提示词上分别测试 vLLM greedy / Qwen3 EAGLE3。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path


DEFAULT_PROMPTS = (
    "请用中文解释投机解码为什么能够保持输出无损，并给出一个简单例子。",
    "Write a short Python function that checks whether a string is a palindrome, then explain it.",
    "Solve step by step: A shop discounts an 800 yuan item by 15%, then applies a 5% coupon. What is the final price?",
)


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
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-json", type=Path)
    parser.add_argument(
        "--wsl-compat",
        action=argparse.BooleanOptionalAction,
        default=is_wsl,
        help="启用当前机器需要的 WSL 兼容环境变量",
    )
    args = parser.parse_args()

    if args.wsl_compat:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # 本机 WSL 无 UVA，且 FlashInfer sampler 的 JIT 缺 ninja；pickle IPC 仅限可信本地运行。

    from vllm import LLM, SamplingParams

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
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.78,
        disable_log_stats=False,
        speculative_config=speculative_config,
    )
    # 8 GiB 4060 的保守配置；greedy/EAGLE 使用相同 eager、batch 和显存比例。

    warmup = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    llm.generate(["Warm up the model."], warmup, use_tqdm=False)
    metrics_before = _counter_values(llm)

    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=0,
    )
    started = time.perf_counter()
    outputs = llm.generate(list(DEFAULT_PROMPTS), sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started
    metrics_after = _counter_values(llm)

    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    total_tokens = sum(len(ids) for ids in token_ids)
    deltas = {
        name: value - metrics_before.get(name, 0.0)
        for name, value in metrics_after.items()
        if name.startswith("vllm:spec_decode_")
    }
    drafts = deltas.get("vllm:spec_decode_num_drafts", 0.0)
    accepted = deltas.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    proposed = deltas.get("vllm:spec_decode_num_draft_tokens", 0.0)
    result = {
        "mode": args.mode,
        "elapsed_seconds": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed,
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
