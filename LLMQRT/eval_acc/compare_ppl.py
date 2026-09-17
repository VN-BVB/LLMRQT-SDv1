#!/usr/bin/env python3
"""Sequentially evaluate original/AWQ/FP8 models and compare perplexity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class EvalSpec:
    name: str
    display_name: str
    model_path: str
    evaluator: Path
    torch_dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "用同一份数据依次评估 Qwen3-0.6B 原版、AWQ、"
            "FP8 dynamic 和 FP8 static，并汇总 PPL 差异。"
        )
    )
    parser.add_argument(
        "--original-model",
        default="/mnt/e/aiinfra/llmqt_awq/models/Qwen--Qwen3-0.6B",
    )
    parser.add_argument(
        "--awq-model",
        default="/mnt/e/aiinfra/llmqt_awq/quantized/Qwen--Qwen3-0.6B-awq",
    )
    parser.add_argument(
        "--fp8-dynamic-model",
        default="/mnt/e/aiinfra/llmqt_fp8/quantized/Qwen3-0.6B-fp8-dynamic",
    )
    parser.add_argument(
        "--fp8-static-model",
        default="/mnt/e/aiinfra/llmqt_fp8/quantized/Qwen3-0.6B-fp8-static",
    )
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument(
        "--local-text-file",
        type=Path,
        help="使用指定文本评估；不指定时会下载/读取 Hugging Face 数据集一次。",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "eval_results")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="仅评估前 N 个 token，适合快速验证；默认评估全部数据。",
    )
    parser.add_argument(
        "--quant-models",
        nargs="+",
        choices=("awq", "fp8_dynamic", "fp8_static"),
        default=("awq", "fp8_dynamic", "fp8_static"),
        help="要评估的量化模型；原版始终会先评估。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="某个模型失败时立即停止，默认继续评估剩余模型。",
    )
    args = parser.parse_args()
    if args.max_length <= 1:
        parser.error("--max-length 必须大于 1")
    if args.stride <= 0 or args.stride > args.max_length:
        parser.error("--stride 必须在 [1, max-length] 范围内")
    if args.max_tokens is not None and args.max_tokens <= 1:
        parser.error("--max-tokens 必须大于 1")
    return args


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def prepare_shared_text(args: argparse.Namespace) -> Path:
    """Materialize the dataset once so every subprocess sees identical text."""
    if args.local_text_file is not None:
        path = args.local_text_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"本地评估文本不存在: {path}")
        return path

    dataset_dir = args.output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(
        f"{args.dataset_name}_{args.dataset_config}_{args.dataset_split}"
    )
    text_path = dataset_dir / f"{filename}.txt"
    if text_path.is_file() and text_path.stat().st_size > 0:
        print(f"复用已准备的评估文本: {text_path}")
        return text_path.resolve()

    print(
        f"准备共享数据集: "
        f"{args.dataset_name}/{args.dataset_config}/{args.dataset_split}"
    )
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )
    if "text" not in dataset.column_names:
        raise ValueError(f"数据集没有 text 列: {dataset.column_names}")
    text_path.write_text("\n\n".join(dataset["text"]), encoding="utf-8")
    if text_path.stat().st_size == 0:
        raise ValueError("数据集 text 列为空")
    print(f"共享评估文本已保存: {text_path}")
    return text_path.resolve()


def build_specs(args: argparse.Namespace) -> list[EvalSpec]:
    quant_specs = {
        "awq": EvalSpec(
            "awq",
            "AWQ INT4",
            args.awq_model,
            SCRIPT_DIR / "eval_ppl_manual.py",
            "float16",
        ),
        "fp8_dynamic": EvalSpec(
            "fp8_dynamic",
            "FP8 Dynamic",
            args.fp8_dynamic_model,
            SCRIPT_DIR / "eval_ppl_manual.py",
            "bfloat16",
        ),
        "fp8_static": EvalSpec(
            "fp8_static",
            "FP8 Static",
            args.fp8_static_model,
            SCRIPT_DIR / "eval_ppl_manual.py",
            "bfloat16",
        ),
    }
    specs = [
        EvalSpec(
            "original",
            "Original BF16",
            args.original_model,
            SCRIPT_DIR / "eval_ppl_original.py",
            "bfloat16",
        )
    ]
    specs.extend(quant_specs[name] for name in args.quant_models)
    return specs


def validate_specs(specs: list[EvalSpec]) -> None:
    for spec in specs:
        if not spec.evaluator.is_file():
            raise FileNotFoundError(f"评估脚本不存在: {spec.evaluator}")
        model_path = Path(spec.model_path).expanduser()
        if model_path.is_absolute() and not model_path.is_dir():
            raise FileNotFoundError(f"{spec.display_name} 模型目录不存在: {model_path}")


def run_one(
    spec: EvalSpec,
    args: argparse.Namespace,
    shared_text: Path,
) -> dict[str, Any]:
    result_path = (args.output_dir / f"{spec.name}_ppl.json").resolve()
    log_path = (args.output_dir / f"{spec.name}_ppl.log").resolve()
    command = [
        sys.executable,
        "-u",
        str(spec.evaluator),
        "--model-path",
        spec.model_path,
        "--output-file",
        str(result_path),
        "--dataset-name",
        args.dataset_name,
        "--dataset-config",
        args.dataset_config,
        "--dataset-split",
        args.dataset_split,
        "--local-text-file",
        str(shared_text),
        "--max-length",
        str(args.max_length),
        "--stride",
        str(args.stride),
        "--torch-dtype",
        spec.torch_dtype,
    ]
    if args.max_tokens is not None:
        command.extend(("--max-tokens", str(args.max_tokens)))

    print("\n" + "=" * 78)
    print(f"开始评估: {spec.display_name}")
    print(f"模型: {spec.model_path}")
    print(f"日志: {log_path}")
    print("=" * 78)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{old_pythonpath}"
        if old_pythonpath
        else str(PROJECT_ROOT)
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        return {
            "name": spec.name,
            "display_name": spec.display_name,
            "model": spec.model_path,
            "status": "failed",
            "return_code": return_code,
            "result_file": str(result_path),
            "log_file": str(log_path),
        }
    if not result_path.is_file():
        return {
            "name": spec.name,
            "display_name": spec.display_name,
            "model": spec.model_path,
            "status": "failed",
            "error": "评估进程成功退出，但未生成结果文件",
            "result_file": str(result_path),
            "log_file": str(log_path),
        }

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "model": spec.model_path,
        "status": "ok",
        "perplexity": float(payload["perplexity"]),
        "tokens_evaluated": int(payload["tokens_evaluated"]),
        "result_file": str(result_path),
        "log_file": str(log_path),
    }


def add_comparisons(rows: list[dict[str, Any]]) -> None:
    original = next(
        (row for row in rows if row["name"] == "original" and row["status"] == "ok"),
        None,
    )
    if original is None:
        return
    baseline = original["perplexity"]
    for row in rows:
        if row["status"] != "ok":
            continue
        delta = row["perplexity"] - baseline
        row["ppl_delta_vs_original"] = delta
        row["ppl_delta_percent_vs_original"] = (
            delta / baseline * 100.0 if baseline != 0 else None
        )


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\nPPL 对比（越低越好）")
    print("-" * 78)
    print(f"{'Model':<20} {'Status':<8} {'PPL':>12} {'Delta':>12} {'Delta %':>12}")
    print("-" * 78)
    for row in rows:
        if row["status"] == "ok":
            ppl = f"{row['perplexity']:.6f}"
            if "ppl_delta_vs_original" in row:
                delta = f"{row['ppl_delta_vs_original']:+.6f}"
                percent = f"{row['ppl_delta_percent_vs_original']:+.3f}%"
            else:
                delta = percent = "-"
        else:
            ppl = delta = percent = "-"
        print(
            f"{row['display_name']:<20} {row['status']:<8} "
            f"{ppl:>12} {delta:>12} {percent:>12}"
        )
    print("-" * 78)


def write_summary(rows: list[dict[str, Any]], args: argparse.Namespace, shared_text: Path) -> None:
    summary_path = (args.output_dir / "ppl_comparison.json").resolve()
    csv_path = (args.output_dir / "ppl_comparison.csv").resolve()
    summary = {
        "dataset": f"{args.dataset_name}/{args.dataset_config}/{args.dataset_split}",
        "shared_text_file": str(shared_text),
        "max_length": args.max_length,
        "stride": args.stride,
        "max_tokens": args.max_tokens,
        "results": rows,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fields = [
        "name",
        "display_name",
        "status",
        "model",
        "perplexity",
        "ppl_delta_vs_original",
        "ppl_delta_percent_vs_original",
        "tokens_evaluated",
        "result_file",
        "log_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"汇总 JSON: {summary_path}")
    print(f"汇总 CSV:  {csv_path}")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    validate_specs(specs)
    shared_text = prepare_shared_text(args)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_one(spec, args, shared_text)
        rows.append(row)
        if row["status"] != "ok" and args.fail_fast:
            break

    add_comparisons(rows)
    print_table(rows)
    write_summary(rows, args, shared_text)
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
