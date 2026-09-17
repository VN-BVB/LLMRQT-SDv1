#!/usr/bin/env python3
"""Benchmark and profile Original/AWQ/FP8 Qwen3-0.6B models."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime_refact.core.api import AutoQuantForCausalLM
from runtime_refact.utils.common_utils import get_best_device


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    path: str
    quant_method: str
    torch_dtype: str

    @property
    def is_quantized(self) -> bool:
        return self.quant_method != "original"


MODEL_SPECS = {
    "original": ModelSpec(
        key="original",
        name="Qwen3 | Original BF16 | 0.6B",
        path="/mnt/e/aiinfra/llmqt_awq/models/Qwen--Qwen3-0.6B",
        quant_method="original",
        torch_dtype="bfloat16",
    ),
    "awq": ModelSpec(
        key="awq",
        name="Qwen3 | AWQ INT4 | 0.6B",
        path="/mnt/e/aiinfra/llmqt_awq/quantized/Qwen--Qwen3-0.6B-awq",
        quant_method="awq",
        torch_dtype="float16",
    ),
    "fp8_dynamic": ModelSpec(
        key="fp8_dynamic",
        name="Qwen3 | FP8 Dynamic | 0.6B",
        path="/mnt/e/aiinfra/llmqt_fp8/quantized/Qwen3-0.6B-fp8-dynamic",
        quant_method="fp8_dynamic_quant",
        torch_dtype="bfloat16",
    ),
    "fp8_static": ModelSpec(
        key="fp8_static",
        name="Qwen3 | FP8 Static | 0.6B",
        path="/mnt/e/aiinfra/llmqt_fp8/quantized/Qwen3-0.6B-fp8-static",
        quant_method="fp8_static_quant",
        torch_dtype="bfloat16",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qwen3-0.6B Original/AWQ/FP8 models, profile CUDA kernels, "
            "and compare every quantized model with the original model."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=tuple(MODEL_SPECS),
        help="Models to run. Defaults to all four models.",
    )
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=64,
        help="Timed decode forward steps after the first generated token.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable full CUDA kernel profiling (default: enabled).",
    )
    parser.add_argument("--profile-topk", type=int, default=10)
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Also save a Chrome trace JSON for every profiled model.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=SCRIPT_DIR / "benchmark_results" / "qwen3_0_6b_comparison.json",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed model instead of continuing.",
    )
    args = parser.parse_args()
    if args.input_length <= 1:
        parser.error("--input-length must be greater than 1")
    if args.decode_tokens <= 0:
        parser.error("--decode-tokens must be greater than 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.repeat_runs <= 0:
        parser.error("--repeat-runs must be greater than 0")
    if args.profile_topk <= 0:
        parser.error("--profile-topk must be greater than 0")
    return args


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def validate_model(spec: ModelSpec) -> Path:
    model_path = Path(spec.path).resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError(
            f"{spec.name}: expected model_type='qwen3', got {config.get('model_type')!r}"
        )
    actual_method = (
        config.get("quantization_config", {}).get("quant_method") or "original"
    )
    if actual_method != spec.quant_method:
        raise ValueError(
            f"{spec.name}: expected quant_method={spec.quant_method!r}, "
            f"got {actual_method!r}"
        )
    return model_path


def load_model(
    spec: ModelSpec,
    args: argparse.Namespace,
) -> tuple[Any, Any, torch.device, Path]:
    model_path = validate_model(spec)
    dtype = getattr(torch, spec.torch_dtype)
    device = torch.device(get_best_device())
    max_seq_len = args.input_length + args.decode_tokens + 8

    print("\n" + "=" * 100)
    print(f"Loading: {spec.name}")
    print("=" * 100)
    print(f"Model path:       {model_path}")
    print(f"Loader:           {'LLMQRT' if spec.is_quantized else 'Transformers'}")
    print(f"Quant method:     {spec.quant_method}")
    print(f"Torch dtype:      {spec.torch_dtype}")
    print(f"CUDA device mask: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    if spec.is_quantized:
        os.environ["BATCH_SIZE"] = str(args.batch_size)
        model = AutoQuantForCausalLM.from_quantized(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto",
            batch_size=args.batch_size,
            max_seq_len=max_seq_len,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    model.eval()
    print(f"Model ready on {device}")
    return model, tokenizer, device, model_path


def make_input_ids(
    tokenizer: Any,
    input_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    seed_text = (
        "Large language model inference contains a prefill phase and an autoregressive "
        "decode phase. This fixed prompt is repeated for a reproducible benchmark. "
    )
    seed_ids = tokenizer(seed_text, add_special_tokens=False).input_ids
    if not seed_ids:
        raise ValueError("Tokenizer returned an empty input")
    repeats = math.ceil(input_length / len(seed_ids))
    values = (seed_ids * repeats)[:input_length]
    return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0).repeat(
        batch_size, 1
    )


@torch.inference_mode()
def run_once(
    model: Any,
    input_ids: torch.Tensor,
    decode_tokens: int,
    device: torch.device,
    is_quantized: bool,
) -> dict[str, Any]:
    """Measure prefill and greedy decode; use the correct cache route per backend."""
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    if is_quantized:
        outputs = model(input_ids=input_ids)
        past_key_values = None
    else:
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - start) * 1000.0

    generated = [next_token]
    decode_latencies_ms: list[float] = []
    for _ in range(decode_tokens):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        if is_quantized:
            outputs = model(input_ids=next_token)
        else:
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        decode_latencies_ms.append((time.perf_counter() - start) * 1000.0)
        generated.append(next_token)

    return {
        "prefill_ms": prefill_ms,
        "decode_latencies_ms": decode_latencies_ms,
        "generated_ids": torch.cat(generated, dim=1).detach().cpu(),
    }


def get_event_cuda_times_us(event: Any) -> tuple[float, float]:
    """Read profiler CUDA timing across old and new PyTorch attribute names."""
    total = 0.0
    self_total = 0.0
    if hasattr(event, "device_time_total"):
        total = float(event.device_time_total)
        self_total = float(getattr(event, "self_device_time_total", 0.0))
    elif hasattr(event, "cuda_time_total"):
        total = float(event.cuda_time_total)
        self_total = float(getattr(event, "self_cuda_time_total", 0.0))
    elif hasattr(event, "cuda_time"):
        total = float(event.cuda_time)
        self_total = float(getattr(event, "self_cuda_time", 0.0))
    return total, self_total


def print_kernel_table(
    title: str,
    events: list[dict[str, Any]],
    time_key: str,
    topk: int,
) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(
        f"{'Rank':<6} {'Total CUDA (ms)':>18} {'Calls':>10} "
        f"{'Average (ms)':>16}  Kernel / operator"
    )
    print("-" * 120)
    for index, event in enumerate(sorted(events, key=lambda item: item[time_key], reverse=True)[:topk], 1):
        print(
            f"{index:<6} {event['cuda_time_ms']:>18.4f} {event['count']:>10} "
            f"{event['avg_time_ms']:>16.6f}  {event['name'][:72]}"
        )


def profile_kernels(
    spec: ModelSpec,
    model: Any,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 100)
    print(f"Full CUDA kernel profiling: {spec.name}")
    print("=" * 100)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with record_function(f"benchmark_{spec.key}"):
            run_once(
                model,
                input_ids,
                args.decode_tokens,
                device,
                spec.is_quantized,
            )

    averages = list(profiler.key_averages())
    cuda_events: list[dict[str, Any]] = []
    pure_cuda_events = 0
    cpu_ops_with_cuda = 0
    for event in averages:
        cuda_time_us, self_cuda_time_us = get_event_cuda_times_us(event)
        if cuda_time_us <= 0:
            continue
        device_type = str(getattr(event, "device_type", "N/A"))
        is_pure_cuda = "cuda" in device_type.lower()
        pure_cuda_events += int(is_pure_cuda)
        cpu_ops_with_cuda += int(not is_pure_cuda)
        count = int(event.count)
        cuda_events.append(
            {
                "name": event.key,
                "cuda_time_ms": cuda_time_us / 1000.0,
                "self_cuda_time_ms": self_cuda_time_us / 1000.0,
                "count": count,
                "avg_time_ms": cuda_time_us / count / 1000.0 if count else 0.0,
                "device_type": device_type,
            }
        )

    print(f"Profiler averaged events:            {len(averages)}")
    print(f"Pure CUDA kernel events:             {pure_cuda_events}")
    print(f"CPU operators containing CUDA time: {cpu_ops_with_cuda}")
    print(f"All events containing CUDA time:    {len(cuda_events)}")
    if averages:
        first = averages[0]
        time_attributes = [
            name
            for name in dir(first)
            if "time" in name.lower() and not name.startswith("_")
        ]
        print(f"First event: key={first.key!r}, count={first.count}")
        print(f"Available time attributes: {time_attributes}")

    by_total = sorted(cuda_events, key=lambda item: item["cuda_time_ms"], reverse=True)
    by_average = sorted(cuda_events, key=lambda item: item["avg_time_ms"], reverse=True)
    if cuda_events:
        print_kernel_table(
            f"Top {args.profile_topk} CUDA kernels/operators by total time",
            cuda_events,
            "cuda_time_ms",
            args.profile_topk,
        )
        print_kernel_table(
            f"Top {args.profile_topk} CUDA kernels/operators by average time",
            cuda_events,
            "avg_time_ms",
            args.profile_topk,
        )
    else:
        print("WARNING: no CUDA timing events found; raw profiler table follows.")

    print("\nRaw PyTorch profiler table:")
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=args.profile_topk,
        )
    )

    trace_file = None
    if args.save_traces:
        trace_file = output_dir / "traces" / f"{spec.key}_trace.json"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_file))
        print(f"Chrome trace saved: {trace_file}")

    total_cuda_event_time_ms = sum(event["cuda_time_ms"] for event in cuda_events)
    top_total_time_ms = sum(
        event["cuda_time_ms"] for event in by_total[: args.profile_topk]
    )
    print(f"Aggregated CUDA event time: {total_cuda_event_time_ms:.4f} ms")
    print(
        f"Top {args.profile_topk} aggregated time: {top_total_time_ms:.4f} ms "
        f"({top_total_time_ms / total_cuda_event_time_ms * 100.0 if total_cuda_event_time_ms else 0.0:.2f}%)"
    )

    return {
        "num_averaged_events": len(averages),
        "num_pure_cuda_kernel_events": pure_cuda_events,
        "num_cpu_ops_with_cuda_time": cpu_ops_with_cuda,
        "num_events_with_cuda_time": len(cuda_events),
        "aggregated_cuda_event_time_ms": total_cuda_event_time_ms,
        "top_by_total_time": by_total[: args.profile_topk],
        "top_by_average_time": by_average[: args.profile_topk],
        "trace_file": str(trace_file) if trace_file else None,
    }


def benchmark_model(spec: ModelSpec, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    model, tokenizer, device, model_path = load_model(spec, args)
    input_ids = make_input_ids(tokenizer, args.input_length, args.batch_size, device)
    print(f"Input shape: {tuple(input_ids.shape)}")

    print(f"\n[Warmup] {args.warmup_runs} run(s)")
    for index in range(args.warmup_runs):
        run_once(model, input_ids, args.decode_tokens, device, spec.is_quantized)
        print(f"  warmup {index + 1}/{args.warmup_runs} complete")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    runs: list[dict[str, Any]] = []
    sample_ids: torch.Tensor | None = None
    print(f"\n[Benchmark] {args.repeat_runs} measured run(s)")
    for index in range(args.repeat_runs):
        result = run_once(model, input_ids, args.decode_tokens, device, spec.is_quantized)
        generated_ids = result.pop("generated_ids")
        if sample_ids is None:
            sample_ids = generated_ids
        runs.append(result)
        decode_total_ms = sum(result["decode_latencies_ms"])
        decode_tps = args.batch_size * args.decode_tokens / (decode_total_ms / 1000.0)
        print(
            f"  run {index + 1}/{args.repeat_runs}: "
            f"prefill={result['prefill_ms']:.3f} ms, "
            f"decode={decode_total_ms:.3f} ms, "
            f"decode={decode_tps:.2f} tokens/s, "
            f"total={result['prefill_ms'] + decode_total_ms:.3f} ms"
        )

    prefill_values = [run["prefill_ms"] for run in runs]
    decode_values = [value for run in runs for value in run["decode_latencies_ms"]]
    decode_totals = [sum(run["decode_latencies_ms"]) for run in runs]
    total_values = [run["prefill_ms"] + total for run, total in zip(runs, decode_totals)]
    prefill = latency_stats(prefill_values)
    decode = latency_stats(decode_values)
    end_to_end = latency_stats(total_values)
    avg_decode_total_ms = statistics.fmean(decode_totals)
    metrics = {
        "prefill_latency_ms": prefill,
        "prefill_throughput_tokens_per_second": (
            args.batch_size * args.input_length / (prefill["mean"] / 1000.0)
        ),
        "decode_latency_ms_per_step": decode,
        "decode_throughput_tokens_per_second": (
            args.batch_size * args.decode_tokens / (avg_decode_total_ms / 1000.0)
        ),
        "end_to_end_latency_ms": end_to_end,
        "generation_throughput_tokens_per_second": (
            args.batch_size * (args.decode_tokens + 1) / (end_to_end["mean"] / 1000.0)
        ),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }

    assert sample_ids is not None
    sample_output = tokenizer.decode(sample_ids[0], skip_special_tokens=True)
    print("\n[Performance summary]")
    print(
        f"  Prefill: mean={prefill['mean']:.3f} ms, "
        f"P50/P95/P99={prefill['p50']:.3f}/{prefill['p95']:.3f}/{prefill['p99']:.3f} ms, "
        f"throughput={metrics['prefill_throughput_tokens_per_second']:.2f} tokens/s"
    )
    print(
        f"  Decode:  mean={decode['mean']:.3f} ms/step, "
        f"P50/P95/P99={decode['p50']:.3f}/{decode['p95']:.3f}/{decode['p99']:.3f} ms, "
        f"throughput={metrics['decode_throughput_tokens_per_second']:.2f} tokens/s"
    )
    print(
        f"  E2E:     mean={end_to_end['mean']:.3f} ms, "
        f"generation={metrics['generation_throughput_tokens_per_second']:.2f} tokens/s"
    )
    print(f"  Peak GPU memory: {metrics['peak_gpu_memory_mib']:.2f} MiB")
    print(f"  Sample output: {sample_output!r}")

    profiler_result = None
    if args.profile:
        profiler_result = profile_kernels(
            spec,
            model,
            input_ids,
            args,
            device,
            output_dir,
        )

    return {
        "key": spec.key,
        "name": spec.name,
        "status": "ok",
        "model_path": str(model_path),
        "quant_method": spec.quant_method,
        "torch_dtype": spec.torch_dtype,
        "batch_size": args.batch_size,
        "input_length": args.input_length,
        "decode_tokens": args.decode_tokens,
        "warmup_runs": args.warmup_runs,
        "repeat_runs": args.repeat_runs,
        "metrics": metrics,
        "runs": runs,
        "sample_output": sample_output,
        "profiler": profiler_result,
    }


def add_original_comparisons(results: list[dict[str, Any]]) -> None:
    original = next(
        (item for item in results if item["key"] == "original" and item["status"] == "ok"),
        None,
    )
    if original is None:
        return
    baseline = original["metrics"]
    for item in results:
        if item["status"] != "ok":
            continue
        metrics = item["metrics"]
        item["vs_original"] = {
            "prefill_speedup": (
                baseline["prefill_latency_ms"]["mean"]
                / metrics["prefill_latency_ms"]["mean"]
            ),
            "decode_speedup": (
                baseline["decode_latency_ms_per_step"]["mean"]
                / metrics["decode_latency_ms_per_step"]["mean"]
            ),
            "end_to_end_speedup": (
                baseline["end_to_end_latency_ms"]["mean"]
                / metrics["end_to_end_latency_ms"]["mean"]
            ),
            "decode_throughput_ratio": (
                metrics["decode_throughput_tokens_per_second"]
                / baseline["decode_throughput_tokens_per_second"]
            ),
            "peak_memory_ratio": (
                metrics["peak_gpu_memory_mib"]
                / baseline["peak_gpu_memory_mib"]
            ),
        }


def print_comparison(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print("Qwen3-0.6B comparison (speedup is relative to Original; higher is faster)")
    print("=" * 132)
    print(
        f"{'Model':<32} {'Status':<8} {'Prefill ms':>12} {'Decode ms':>12} "
        f"{'Decode tok/s':>14} {'E2E ms':>12} {'Memory MiB':>12} {'Decode x':>10}"
    )
    print("-" * 132)
    for item in results:
        if item["status"] != "ok":
            print(f"{item['name']:<32} {'failed':<8} {'-':>12} {'-':>12} {'-':>14} {'-':>12} {'-':>12} {'-':>10}")
            continue
        metrics = item["metrics"]
        ratio = item.get("vs_original", {}).get("decode_speedup")
        ratio_text = f"{ratio:.3f}x" if ratio is not None else "-"
        print(
            f"{item['name']:<32} {'ok':<8} "
            f"{metrics['prefill_latency_ms']['mean']:>12.3f} "
            f"{metrics['decode_latency_ms_per_step']['mean']:>12.3f} "
            f"{metrics['decode_throughput_tokens_per_second']:>14.2f} "
            f"{metrics['end_to_end_latency_ms']['mean']:>12.3f} "
            f"{metrics['peak_gpu_memory_mib']:>12.2f} {ratio_text:>10}"
        )
    print("=" * 132)


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    output_file = args.output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Qwen3-0.6B unified benchmark")
    print("=" * 100)
    print(f"Models:         {', '.join(args.models)}")
    print(f"Input length:   {args.input_length}")
    print(f"Decode tokens:  {args.decode_tokens}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Warmup/repeats: {args.warmup_runs}/{args.repeat_runs}")
    print(f"Kernel profile: {'enabled' if args.profile else 'disabled'}")
    print(f"Output:         {output_file}")

    results: list[dict[str, Any]] = []
    for index, model_key in enumerate(args.models, 1):
        spec = MODEL_SPECS[model_key]
        print(f"\n[{index}/{len(args.models)}] {spec.name}")
        try:
            result = benchmark_model(spec, args, output_file.parent)
        except Exception as error:
            result = {
                "key": spec.key,
                "name": spec.name,
                "status": "failed",
                "model_path": spec.path,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"ERROR: {result['error']}")
            if args.fail_fast:
                results.append(result)
                break
        results.append(result)
        clear_gpu_memory()

    add_original_comparisons(results)
    print_comparison(results)
    payload = {
        "input_length": args.input_length,
        "decode_tokens": args.decode_tokens,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup_runs,
        "repeat_runs": args.repeat_runs,
        "profile_enabled": args.profile,
        "profile_topk": args.profile_topk,
        "results": results,
    }
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nCombined benchmark/profile JSON: {output_file}")
    return 1 if any(item["status"] != "ok" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
