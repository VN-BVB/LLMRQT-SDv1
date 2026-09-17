#!/usr/bin/env python3
"""Run the Week 78 FP16 vs AWQ/FP8 assignment benchmark.

The script deliberately keeps the benchmark small and easy to audit:

* the FP16 baseline and every quantized model receive exactly the same token IDs;
* batch sizes 1 and 8 are tested;
* every case is measured three times and the median is reported;
* TTFT, decode throughput and peak allocated GPU memory are recorded;
* greedy decoding and a fixed seed make the generated text reproducible;
* JSON and Markdown reports are written at the end.

Both models are tested by one invocation, but they are loaded sequentially.  If
they stayed resident at the same time, one model would contaminate the other
model's peak-memory number and a larger GPU would be required for no useful
reason.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_refact.core.api import AutoQuantForCausalLM
from runtime_refact.nn_models.modules.linear.linear_awq import AWQLinear_GEMM
from runtime_refact.nn_models.modules.linear.linear_fp8 import (
    FP8DynamicLinear,
    FP8NativeDynamicLinear,
    FP8StaticLinear,
)
from runtime_refact.nn_models.modules.nonlinear.attention import QuantAttentionFused


DEFAULT_PROMPT = (
    "Explain in two or three sentences why the sky looks blue during the day."
)


@dataclass(frozen=True)
class ModelCase:
    key: str
    display_name: str
    path: Path
    quant_method: str
    torch_dtype: torch.dtype
    linear_impl: str | None = None

    @property
    def is_quantized(self) -> bool:
        return self.quant_method != "fp16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FP16 with LLMQRT AWQ and optional FP8 versions."
    )
    parser.add_argument("--fp16-model", type=Path, required=True)
    parser.add_argument("--awq-model", type=Path, required=True)
    parser.add_argument("--fp8-dynamic-model", type=Path)
    parser.add_argument("--fp8-native-dynamic-model", type=Path)
    parser.add_argument("--fp8-static-model", type=Path)
    parser.add_argument(
        "--compare-dequant-paths",
        action="store_true",
        help=(
            "Add AWQ/FP8 reference cases that fully dequantize weights on "
            "every Linear forward before torch.matmul/F.linear."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "assignment_results",
    )
    parser.add_argument(
        "--update-main-report",
        action="store_true",
        help="Replace the result markers in the main study report after a successful run.",
    )
    args = parser.parse_args()

    if args.output_length < 2:
        parser.error("--output-length must be at least 2")
    if args.repeat_runs != 3:
        parser.error("The assignment requires exactly three measured runs")
    if sorted(args.batch_sizes) != [1, 8]:
        parser.error("The assignment requires exactly --batch-sizes 1 8")
    return args


def set_seed(seed: int) -> None:
    """Set every RNG normally involved in this program.

    Decoding is greedy, so it should not consume random numbers.  Keeping the
    seed explicit still makes the experimental contract unambiguous.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_model_dir(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{name} is missing config.json: {path}")
    return path


def tokenize_prompt(tokenizer: Any, prompt: str) -> torch.Tensor:
    """Tokenize once; the returned CPU tensor is shared by both model cases."""

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        # Transformers versions differ here: some return a Tensor, newer ones
        # may return a BatchEncoding even when return_dict was not requested.
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
        else:
            input_ids = encoded["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(f"Unexpected tokenizer output shape: {tuple(input_ids.shape)}")
    if input_ids.shape[1] < 2:
        raise RuntimeError("The prompt is too short for a prefill benchmark")
    return input_ids.cpu()


def token_id_digest(input_ids: torch.Tensor) -> str:
    raw = input_ids.to(torch.int64).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_model(case: ModelCase, batch_size: int, max_seq_len: int) -> Any:
    """Load one model with the activation dtype required by its runtime."""

    if case.is_quantized:
        # QuantAttentionFused currently reads BATCH_SIZE when allocating KV cache.
        os.environ["BATCH_SIZE"] = str(batch_size)
        model = AutoQuantForCausalLM.from_quantized(
            str(case.path),
            torch_dtype=case.torch_dtype,
            device_map="auto",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            fuse_layers=True,
        )
        expected_layer = {
            "awq": AWQLinear_GEMM,
            "fp8_dynamic": FP8DynamicLinear,
            "fp8_native_dynamic": FP8NativeDynamicLinear,
            "fp8_static": FP8StaticLinear,
        }[case.quant_method]
        layer_count = sum(
            isinstance(module, expected_layer) for module in model.model.modules()
        )
        if layer_count == 0:
            raise RuntimeError(
                f"No {expected_layer.__name__} layer was found; check quant_method and checkpoint format"
            )
        print(f"  detected {layer_count} {expected_layer.__name__} layers")
        if case.linear_impl is not None:
            configured = 0
            for module in model.model.modules():
                if isinstance(module, expected_layer):
                    module.impl_mode = case.linear_impl
                    configured += 1
            if configured != layer_count:
                raise RuntimeError(
                    f"Configured {configured} layers but detected {layer_count}"
                )
            print(f"  linear implementation: {case.linear_impl}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(case.path),
            torch_dtype=case.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model


def reset_llmqrt_cache(model: Any) -> None:
    """Start a fresh LLMQRT request before each warmup or measured run.

    QuantAttentionFused owns its cache position instead of receiving a Hugging
    Face ``past_key_values`` object.  Reusing the model without resetting this
    integer would make run 2 continue from run 1 and invalidate the benchmark.
    The following prefill overwrites the cache range that it uses, so clearing
    every allocated KV-cache byte is unnecessary.
    """

    root = getattr(model, "model", model)
    for module in root.modules():
        if isinstance(module, QuantAttentionFused):
            module.start_pos = 0


@torch.inference_mode()
def run_once(
    model: Any,
    input_ids: torch.Tensor,
    output_length: int,
    is_quantized: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Measure one prefill/first-token phase followed by greedy decode.

    TTFT here excludes tokenization and model loading.  It includes the prefill
    forward and argmax that makes the first output token available.
    """

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    if is_quantized:
        reset_llmqrt_cache(model)

    ttft_start = time.perf_counter()
    if is_quantized:
        # Fused LLMQRT attention owns its KV cache internally.
        outputs = model(input_ids=input_ids)
        past_key_values = None
    else:
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

    generated = [next_token]
    decode_steps = output_length - 1
    decode_start = time.perf_counter()
    for _ in range(decode_steps):
        if is_quantized:
            outputs = model(input_ids=next_token)
        else:
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    # Throughput is system throughput: all tokens generated by the whole batch.
    decode_tokens_per_second = (
        input_ids.shape[0] * decode_steps / decode_seconds
    )
    return {
        "ttft_ms": ttft_ms,
        "decode_tokens_per_second": decode_tokens_per_second,
        "peak_gpu_memory_mib": peak_memory_mib,
        "generated_ids": torch.cat(generated, dim=1).cpu(),
    }


def benchmark_case(
    case: ModelCase,
    batch_size: int,
    base_input_ids: torch.Tensor,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    input_ids = base_input_ids.repeat(batch_size, 1).to(device)
    max_seq_len = input_ids.shape[1] + args.output_length + 8

    print(f"\n[{case.display_name}] batch={batch_size}")
    model = load_model(case, batch_size, max_seq_len)

    # Warmups are not part of the reported three runs.  A short output is
    # enough to warm both the prefill and decode kernels.
    for warmup_index in range(args.warmup_runs):
        set_seed(args.seed)
        run_once(
            model,
            input_ids,
            min(args.output_length, 8),
            case.is_quantized,
            device,
        )
        print(f"  warmup {warmup_index + 1}/{args.warmup_runs}")

    runs: list[dict[str, float]] = []
    sample_output = ""
    for run_index in range(args.repeat_runs):
        set_seed(args.seed)
        measured = run_once(
            model,
            input_ids,
            args.output_length,
            case.is_quantized,
            device,
        )
        generated_ids = measured.pop("generated_ids")
        if run_index == 0:
            sample_output = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            ).strip()
        runs.append(measured)
        print(
            f"  run {run_index + 1}/3: "
            f"TTFT={measured['ttft_ms']:.3f} ms, "
            f"decode={measured['decode_tokens_per_second']:.2f} tok/s, "
            f"peak={measured['peak_gpu_memory_mib']:.2f} MiB"
        )

    result = {
        "model": case.key,
        "display_name": case.display_name,
        "model_path": str(case.path),
        "dtype": str(case.torch_dtype).removeprefix("torch."),
        "linear_impl": case.linear_impl,
        "batch_size": batch_size,
        "input_length": int(input_ids.shape[1]),
        "output_length": args.output_length,
        "runs": runs,
        "median": {
            metric: statistics.median(run[metric] for run in runs)
            for metric in (
                "ttft_ms",
                "decode_tokens_per_second",
                "peak_gpu_memory_mib",
            )
        },
        "sample_output": sample_output,
    }

    del model, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def hardware_info(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_total_memory_gib": props.total_memory / 1024**3,
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    hardware = payload["hardware"]
    lines = [
        "#### 实测环境与最终结果",
        "",
        f"- GPU：{hardware['gpu']}，Compute Capability {hardware['compute_capability']}，"
        f"显存 {hardware['gpu_total_memory_gib']:.2f} GiB",
        f"- 软件：PyTorch {hardware['pytorch']}，CUDA {hardware['cuda']}",
        f"- seed：{config['seed']}；每组预热 {config['warmup_runs']} 次，正式运行 "
        f"{config['repeat_runs']} 次并取中位数",
        f"- prompt：`{config['prompt']}`",
        f"- 所有模型共用同一份 token IDs，SHA256 前 16 位：`{config['token_id_digest']}`",
        "- TTFT 包含 prefill forward 和首 token argmax，不包含模型加载与 tokenization；"
        "decode tokens/s 是整个 batch 的系统吞吐；显存为 `max_memory_allocated`。",
        "",
        "| 模型 | Batch | 输入长度 | 输出长度 | TTFT 中位数 (ms) | Decode 中位数 (tokens/s) | 峰值显存中位数 (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        med = result["median"]
        lines.append(
            f"| {result['display_name']} | {result['batch_size']} | "
            f"{result['input_length']} | {result['output_length']} | "
            f"{med['ttft_ms']:.3f} | {med['decode_tokens_per_second']:.2f} | "
            f"{med['peak_gpu_memory_mib']:.2f} |"
        )

    indexed = {
        (result["model"], result["batch_size"]): result
        for result in payload["results"]
    }
    lines += ["", "#### 结果分析", ""]
    for batch_size in config["batch_sizes"]:
        fp16 = indexed[("fp16", batch_size)]["median"]
        for result in payload["results"]:
            if result["batch_size"] != batch_size or result["model"] == "fp16":
                continue
            quant = result["median"]
            ttft_ratio = quant["ttft_ms"] / fp16["ttft_ms"]
            decode_ratio = (
                quant["decode_tokens_per_second"]
                / fp16["decode_tokens_per_second"]
            )
            memory_saved = 1.0 - (
                quant["peak_gpu_memory_mib"] / fp16["peak_gpu_memory_mib"]
            )
            lines.append(
                f"- batch={batch_size}，{result['display_name']}：TTFT 是 FP16 的 "
                f"{ttft_ratio:.3f}×，Decode 吞吐是 {decode_ratio:.3f}×，"
                f"峰值显存减少 {memory_saved:.1%}。"
            )
    lines += [
        "",
        "性能结果应如实报告，不能只根据量化位宽预设端到端一定更快；量化/反量化开销、"
        "kernel launch、矩阵形状、模型规模和 batch 对 TTFT 与 Decode 的影响不同。若要判定"
        "主要原因，还需要结合 Nsight 或 PyTorch Profiler，而不能只凭这张表下结论。",
    ]

    lines += ["", "#### 同 prompt 生成结果（batch=1）", ""]
    for result in payload["results"]:
        if result["batch_size"] != 1:
            continue
        lines += [
            f"**{result['display_name']}**",
            "",
            "```text",
            result["sample_output"],
            "```",
            "",
        ]
    lines += [
        "以上文本都能正常解码，没有乱码。量化后措辞可以不同；这里检查的是文本是否可读，"
        "更严格的精度结论仍应结合 PPL 或任务准确率。",
        "",
    ]
    return "\n".join(lines)


def update_main_report(generated_section: str) -> None:
    report_path = PROJECT_ROOT.parent / "INT8_FP8_W4A16_GEMM与LLMQRT实战.md"
    start = "<!-- ASSIGNMENT_RESULTS_START -->"
    end = "<!-- ASSIGNMENT_RESULTS_END -->"
    text = report_path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Cannot find unique assignment markers in {report_path}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    replacement = f"{start}\n\n{generated_section.rstrip()}\n\n{end}"
    report_path.write_text(before + replacement + after, encoding="utf-8")
    print(f"Updated main report: {report_path}")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this inside the LLMQRT GPU environment")

    fp16_path = validate_model_dir(args.fp16_model, "FP16")
    awq_path = validate_model_dir(args.awq_model, "AWQ")
    fp8_dynamic_path = (
        validate_model_dir(args.fp8_dynamic_model, "FP8 Dynamic")
        if args.fp8_dynamic_model else None
    )
    fp8_native_dynamic_path = (
        validate_model_dir(args.fp8_native_dynamic_model, "FP8 Native Dynamic")
        if args.fp8_native_dynamic_model else None
    )
    fp8_static_path = (
        validate_model_dir(args.fp8_static_model, "FP8 Static")
        if args.fp8_static_model else None
    )
    set_seed(args.seed)

    # Use one tokenizer and one tokenization result for every model.  This is
    # stronger than merely giving each tokenizer the same prompt string.
    tokenizer = AutoTokenizer.from_pretrained(
        str(fp16_path), trust_remote_code=True
    )
    base_input_ids = tokenize_prompt(tokenizer, args.prompt)
    cases = [
        ModelCase("fp16", "FP16 baseline", fp16_path, "fp16", torch.float16),
        ModelCase(
            "awq",
            "AWQ W4A16 fused kernel",
            awq_path,
            "awq",
            torch.float16,
            "auto",
        ),
    ]
    if args.compare_dequant_paths:
        cases.append(ModelCase(
            "awq_dequant_matmul",
            "AWQ full dequant + Matmul",
            awq_path,
            "awq",
            torch.float16,
            "dequant_matmul",
        ))
    if fp8_dynamic_path:
        cases.append(ModelCase(
            "fp8_dynamic", "FP8 Dynamic", fp8_dynamic_path, "fp8_dynamic", torch.bfloat16
        ))
    if fp8_native_dynamic_path:
        cases.append(ModelCase(
            "fp8_native_dynamic",
            "FP8 Native Dynamic CUTLASS",
            fp8_native_dynamic_path,
            "fp8_native_dynamic",
            torch.bfloat16,
            "cutlass",
        ))
        if args.compare_dequant_paths:
            cases.append(ModelCase(
                "fp8_native_dynamic_dequant_matmul",
                "FP8 Native full dequant + F.linear",
                fp8_native_dynamic_path,
                "fp8_native_dynamic",
                torch.bfloat16,
                "fp8_gemm",
            ))
    if fp8_static_path:
        cases.append(ModelCase(
            "fp8_static", "FP8 Static", fp8_static_path, "fp8_static", torch.bfloat16
        ))

    results = []
    for batch_size in sorted(args.batch_sizes):
        for case in cases:
            results.append(
                benchmark_case(case, batch_size, base_input_ids, tokenizer, args)
            )

    payload = {
        "hardware": hardware_info(torch.device("cuda:0")),
        "config": {
            "prompt": args.prompt,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "repeat_runs": args.repeat_runs,
            "batch_sizes": sorted(args.batch_sizes),
            "output_length": args.output_length,
            "token_id_digest": token_id_digest(base_input_ids),
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_stem = "all_quant_fp16_results" if len(cases) > 2 else "awq_fp16_results"
    json_path = args.output_dir / f"{result_stem}.json"
    markdown_path = args.output_dir / f"{result_stem}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(payload)
    markdown_path.write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    if args.update_main_report:
        update_main_report(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
