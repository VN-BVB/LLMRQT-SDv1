#!/home/lyc/workspace/vllm/.venv/bin/python
"""Run batch-, step-, and tree-scaling comparisons for EAGLE3Pro and vLLM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess


@dataclass(frozen=True)
class Case:
    axis: str
    name: str
    num_prompts: int
    vllm_max_num_seqs: int
    topk: int
    steps: int
    tree_nodes: int
    pro_graph: bool
    vllm_graph: bool
    comparable: bool


def _csv_ints(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _tree_specs(value: str) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    try:
        for item in value.split(","):
            topk, steps, nodes = (int(part) for part in item.split(":"))
            if min(topk, steps, nodes) < 1:
                raise ValueError
            specs.append((topk, steps, nodes))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected TOPK:STEPS:NODES entries separated by commas"
        ) from exc
    return specs


def _run(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    print(f"[RUN] {log_path.stem}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=env["BENCH_PROJECT"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(f"benchmark failed: {' '.join(command)}\n" + "\n".join(tail))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pro_metrics(result: dict) -> dict:
    return {
        "tps": float(result["eagle_tps"]),
        "latency": float(result["eagle_seconds"]),
        "acceptance_length": float(result["acceptance_length"]),
        "rounds": int(result["rounds"]),
        "matched": int(result["matched_draft_tokens"]),
        "proposed": int(result["proposed_draft_tokens"]),
        "acceptance_rate": float(result["draft_acceptance_rate"]),
        "tree_corrections": int(result.get("tree_causal_corrections", 0)),
        "matched_per_round": int(result["matched_draft_tokens"])
        / max(1, int(result["rounds"])),
        "peak_mib": float(result["peak_memory_allocated_mib"]),
        "token_ids": result["eagle_token_ids"],
        "exact_vs_greedy": all(result["exact_match"]),
        "tps_runs": result.get("eagle_trial_tokens_per_second", [result["eagle_tps"]]),
        "latency_runs": result.get("eagle_trial_seconds", [result["eagle_seconds"]]),
    }


def _vllm_metrics(result: dict) -> dict:
    spec = result["spec_metrics"]
    drafts = int(spec["vllm:spec_decode_num_drafts"])
    matched = int(spec["vllm:spec_decode_num_accepted_tokens"])
    return {
        "tps": float(result["tokens_per_second"]),
        "latency": float(result["elapsed_seconds"]),
        "acceptance_length": float(result["acceptance_length"]),
        "rounds": drafts,
        "matched": matched,
        "proposed": int(spec["vllm:spec_decode_num_draft_tokens"]),
        "acceptance_rate": float(result["acceptance_rate"]),
        "matched_per_round": matched / max(1, drafts),
        "peak_mib": result.get("peak_memory_allocated_mib"),
        "token_ids": result["token_ids"],
        "exact_vs_greedy": result.get("exact_match", True),
    }


def _median_run(runs: list[dict]) -> dict:
    chosen = min(runs, key=lambda item: abs(item["tps"] - statistics.median(x["tps"] for x in runs)))
    result = dict(chosen)
    result["tps_runs"] = [item["tps"] for item in runs]
    result["latency_runs"] = [item["latency"] for item in runs]
    result["tps"] = statistics.median(result["tps_runs"])
    result["latency"] = statistics.median(result["latency_runs"])
    result["exact_vs_greedy"] = all(item["exact_vs_greedy"] for item in runs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("all", "batch", "step", "tree"), default="all")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints("1,2,4"))
    parser.add_argument(
        "--batch-prompts",
        type=int,
        default=None,
        help="fixed request count for batch scaling; defaults to max(batch-sizes)",
    )
    parser.add_argument("--steps", type=_csv_ints, default=_csv_ints("1,2,4,8"))
    parser.add_argument("--step-prompts", type=int, default=4)
    parser.add_argument(
        "--trees",
        type=_tree_specs,
        default=_tree_specs("1:4:4,2:4:8,4:4:32,8:4:64"),
        help="comma-separated TOPK:STEPS:MAX_NODES",
    )
    parser.add_argument("--tree-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    project = base / "spec-decoding-main"
    python = Path("/home/lyc/workspace/vllm/.venv/bin/python")
    output_dir = args.output_dir or (
        project / "spec_decoding/eagle3pro/benchmark_results/matrix"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "BENCH_PROJECT": str(project),
            "PYTHONPATH": str(project),
            "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    pro_module = "spec_decoding.eagle3pro.example.benchmark_hf_qwen3_eagle3"
    vllm_module = "spec_decoding.eagle3pro.example.benchmark_vllm_qwen3_eagle3"

    cases: list[Case] = []
    if args.axis in ("all", "batch"):
        batch_prompts = args.batch_prompts or max(args.batch_sizes)
        if batch_prompts < max(args.batch_sizes):
            parser.error("--batch-prompts must be >= every --batch-sizes value")
        cases.extend(
            Case(
                "batch", f"batch-{batch}", batch_prompts, batch,
                1, 1, 1, True, True, True,
            )
            for batch in args.batch_sizes
        )
    if args.axis in ("all", "step"):
        cases.extend(
            Case(
                "step", f"step-{steps}", args.step_prompts, 1,
                1, steps, steps, True, True, True,
            )
            for steps in args.steps
        )
    if args.axis in ("all", "tree"):
        cases.extend(
            Case(
                "tree",
                f"tree-k{topk}-s{steps}-n{nodes}",
                args.tree_prompts,
                1,
                topk,
                steps,
                nodes,
                False,
                False,
                topk == 1,
            )
            for topk, steps, nodes in args.trees
        )

    rows: list[dict] = []
    pro_cache: dict[tuple[int, int, int, int, bool], dict] = {}
    vllm_cache: dict[tuple[int, int, int, bool], dict] = {}
    for case_index, case in enumerate(cases, 1):
        slug = f"{case_index:02d}_{case.name}"
        pro_key = (
            case.num_prompts,
            case.topk,
            case.steps,
            case.tree_nodes,
            case.pro_graph,
        )
        if pro_key not in pro_cache:
            output = output_dir / f"{slug}_pro.json"
            command = [
                str(python),
                "-m",
                pro_module,
                "--target-optimization",
                "packed-vllm-flashkv",
                "--num-prompts",
                str(case.num_prompts),
                "--topk",
                str(case.topk),
                "--num-steps",
                str(case.steps),
                "--max-tree-nodes",
                str(case.tree_nodes),
                "--max-new-tokens",
                str(args.max_tokens),
                "--warmup-tokens",
                str(args.warmup_tokens),
                "--repetitions",
                str(args.repetitions),
                "--benchmark-order",
                "eagle-first",
                "--output",
                str(output),
            ]
            if case.pro_graph:
                command.append("--cuda-graph")
            _run(command, env=env, log_path=output_dir / f"{slug}_pro.log")
            pro_cache[pro_key] = _pro_metrics(_load(output))
        pro = pro_cache[pro_key]

        vllm_key = (
            case.num_prompts,
            case.vllm_max_num_seqs,
            case.steps,
            case.vllm_graph,
        )
        if vllm_key not in vllm_cache:
            output = output_dir / f"{slug}_vllm.json"
            command = [
                str(python),
                "-m",
                vllm_module,
                "--mode",
                "eagle3",
                "--num-speculative-tokens",
                str(case.steps),
                "--num-prompts",
                str(case.num_prompts),
                "--max-num-seqs",
                str(case.vllm_max_num_seqs),
                "--max-tokens",
                str(args.max_tokens),
                "--warmup-tokens",
                str(args.warmup_tokens),
                "--warmup-all-prompts",
                "--repetitions",
                str(args.repetitions),
                "--output",
                str(output),
            ]
            if case.vllm_graph:
                command.append("--cuda-graph")
            _run(command, env=env, log_path=output_dir / f"{slug}_vllm.log")
            raw_vllm = _load(output)
            vllm = _vllm_metrics(raw_vllm)
            vllm["tps_runs"] = raw_vllm["trial_tokens_per_second"]
            vllm["latency_runs"] = raw_vllm["trial_seconds"]
            vllm_cache[vllm_key] = vllm
        vllm = vllm_cache[vllm_key]

        exact = pro["token_ids"] == vllm["token_ids"]
        rows.append(
            {
                "axis": case.axis,
                "case": case.name,
                "num_prompts": case.num_prompts,
                "pro_execution_batch": 1,
                "vllm_max_num_seqs": case.vllm_max_num_seqs,
                "topk": case.topk,
                "steps": case.steps,
                "tree_nodes": case.tree_nodes,
                "comparable_algorithm": case.comparable,
                "pro": {key: value for key, value in pro.items() if key != "token_ids"},
                "vllm": {key: value for key, value in vllm.items() if key != "token_ids"},
                "throughput_delta_percent": (pro["tps"] / vllm["tps"] - 1.0) * 100.0,
                "exact_token_ids": exact,
            }
        )

    summary = {
        "max_tokens_per_prompt": args.max_tokens,
        "repetitions": args.repetitions,
        "notes": {
            "batch": "The request set stays fixed. Pro executes it serially at batch 1; only vLLM max_num_seqs changes.",
            "tree": "vLLM EAGLE3 is a linear chain in this benchmark; topk>1 tree rows are diagnostic, not algorithm-identical comparisons.",
        },
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# EAGLE3Pro / vLLM 扩展矩阵",
        "",
        f"每个 prompt 生成 {args.max_tokens} token；正式结果取 {args.repetitions} 次中位数。",
        "batch 轴保持相同 request 集：Pro 逐请求 batch 1 串行，只改变 vLLM `max_num_seqs`；",
        "tree 的 `topk>1` 行与 vLLM linear chain 不属于算法同构对比。",
        "",
        "| 轴 | 配置 | prompts | vLLM max seqs | Pro tok/s | vLLM tok/s | Pro 相对值 | Pro 接受长度/率/matched每轮 | vLLM 接受长度/率/matched每轮 | tree 校正 | token ids | 同算法 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pro = row["pro"]
        vllm = row["vllm"]
        lines.append(
            f"| {row['axis']} | {row['case']} | {row['num_prompts']} "
            f"| {row['vllm_max_num_seqs']} "
            f"| {pro['tps']:.2f} | {vllm['tps']:.2f} "
            f"| {row['throughput_delta_percent']:+.2f}% "
            f"| {pro['acceptance_length']:.4f} / {pro['acceptance_rate']:.2%} / {pro['matched_per_round']:.4f} "
            f"| {vllm['acceptance_length']:.4f} / {vllm['acceptance_rate']:.2%} / {vllm['matched_per_round']:.4f} "
            f"| {pro['tree_corrections']} "
            f"| {'OK' if row['exact_token_ids'] else 'FAIL'} "
            f"| {'yes' if row['comparable_algorithm'] else 'no'} |"
        )
    (output_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
