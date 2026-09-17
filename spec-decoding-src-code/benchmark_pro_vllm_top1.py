#!/home/lyc/workspace/vllm/.venv/bin/python
"""直接运行 Pro/vLLM top-1 eager 与 CUDA Graph 三次对比。"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from pathlib import Path


def _run(name: str, command: list[str], env: dict[str, str], log: Path) -> None:
    print(f"[RUN] {name}，日志：{log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=env["BENCH_PROJECT"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        raise RuntimeError(f"{name} 失败：\n" + "\n".join(tail))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{value:+.2f}%"


def main() -> None:
    # 所有实验参数集中写在这里；直接运行本文件，不需要传命令行参数。
    base = Path(__file__).resolve().parent
    project = base / "spec-decoding-main"
    python = Path("/home/lyc/workspace/vllm/.venv/bin/python")
    result_dir = project / "spec_decoding/eagle3pro/benchmark_results/direct_compare"
    result_dir.mkdir(parents=True, exist_ok=True)

    repetitions = 3
    max_tokens = 128
    num_steps = 4
    warmup_pro = 32
    warmup_vllm = 128
    graph_max_seq_len = 1024

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

    pro_eager: list[dict] = []
    pro_graph: list[dict] = []
    for run in range(1, repetitions + 1):
        eager_json = result_dir / f"pro_eager_{run}.json"
        graph_json = result_dir / f"pro_graph_{run}.json"
        common = [
            str(python), "-m", pro_module,
            "--target-optimization", "packed-vllm-flashkv",
            "--num-steps", str(num_steps),
            "--warmup-tokens", str(warmup_pro),
            "--max-new-tokens", str(max_tokens),
            "--benchmark-order", "eagle-first",
        ]
        _run(
            f"Pro eager {run}/{repetitions}",
            common + ["--output", str(eager_json)],
            env,
            result_dir / f"pro_eager_{run}.log",
        )
        _run(
            f"Pro CUDA Graph {run}/{repetitions}",
            common
            + [
                "--cuda-graph",
                "--cuda-graph-max-seq-len", str(graph_max_seq_len),
                "--compare-json", str(eager_json),
                "--output", str(graph_json),
            ],
            env,
            result_dir / f"pro_graph_{run}.log",
        )
        pro_eager.append(_load(eager_json))
        pro_graph.append(_load(graph_json))

    vllm_eager_json = result_dir / "vllm_eager.json"
    vllm_graph_json = result_dir / "vllm_graph.json"
    vllm_common = [
        str(python), "-m", vllm_module,
        "--mode", "eagle3",
        "--num-speculative-tokens", str(num_steps),
        "--repetitions", str(repetitions),
        "--max-tokens", str(max_tokens),
        "--warmup-tokens", str(warmup_vllm),
        "--warmup-all-prompts",
    ]
    _run(
        "vLLM EAGLE3 eager",
        vllm_common + ["--output", str(vllm_eager_json)],
        env,
        result_dir / "vllm_eager.log",
    )
    _run(
        "vLLM EAGLE3 Graph/compile",
        vllm_common
        + [
            "--cuda-graph",
            "--compare-json", str(vllm_eager_json),
            "--output", str(vllm_graph_json),
        ],
        env,
        result_dir / "vllm_graph.log",
    )
    vllm_eager = _load(vllm_eager_json)
    vllm_graph = _load(vllm_graph_json)

    pe_runs = [x["eagle_tps"] for x in pro_eager]
    pg_runs = [x["eagle_tps"] for x in pro_graph]
    ve_runs = vllm_eager["trial_tokens_per_second"]
    vg_runs = vllm_graph["trial_tokens_per_second"]
    pe, pg = statistics.median(pe_runs), statistics.median(pg_runs)
    ve, vg = statistics.median(ve_runs), statistics.median(vg_runs)

    canonical = pro_eager[0]["eagle_token_ids"]
    exact = {
        "vLLM EAGLE3 eager": vllm_eager["token_ids"] == canonical,
        "vLLM EAGLE3 Graph/compile": vllm_graph["token_ids"] == canonical,
        "Pro eager": all(x["eagle_token_ids"] == canonical for x in pro_eager),
        "Pro CUDA Graph": all(x["eagle_token_ids"] == canonical for x in pro_graph),
    }

    def pro_accept(x: dict) -> tuple[float, int, int, float]:
        return (
            x["acceptance_length"],
            x["matched_draft_tokens"],
            x["proposed_draft_tokens"],
            x["draft_acceptance_rate"],
        )

    def vllm_accept(x: dict) -> tuple[float, int, int, float]:
        metrics = x["spec_metrics"]
        return (
            x["acceptance_length"],
            int(metrics["vllm:spec_decode_num_accepted_tokens"]),
            int(metrics["vllm:spec_decode_num_draft_tokens"]),
            x["acceptance_rate"],
        )

    rows = [
        ("vLLM EAGLE3 eager", ve_runs, ve, "—", "基准", vllm_accept(vllm_eager)),
        ("vLLM EAGLE3 Graph/compile", vg_runs, vg, _pct((vg / ve - 1) * 100), "基准", vllm_accept(vllm_graph)),
        ("Pro eager", pe_runs, pe, "—", _pct((pe / ve - 1) * 100) + " vs vLLM eager", pro_accept(pro_eager[0])),
        ("Pro CUDA Graph", pg_runs, pg, _pct((pg / pe - 1) * 100), _pct((pg / vg - 1) * 100) + " vs vLLM Graph", pro_accept(pro_graph[0])),
    ]

    lines = [
        "# Pro 与 vLLM top-1 直接对比",
        "",
        f"固定口径：FP16、batch=1、top-1/step-{num_steps}、3 个 prompt × {max_tokens} token；正式段三次取中位数。",
        "",
        "| 实现 | 三次 tok/s | 中位数 tok/s | 相对本实现 eager | 相对 vLLM 同模式 | 接受长度 | matched/proposed | draft 接受率 | 输出 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, trials, median, own, vllm_rel, accept in rows:
        length, matched, proposed, rate = accept
        output = "384/384" if exact[name] else "FAIL"
        lines.append(
            f"| {name} | {' / '.join(f'{x:.2f}' for x in trials)} | {median:.2f} "
            f"| {own} | {vllm_rel} | {length:.4f} | {matched}/{proposed} "
            f"| {rate:.2%} | {output} |"
        )
    report = "\n".join(lines) + "\n"
    report_path = base / "PRO_VLLM_TOP1_RESULT.md"
    report_path.write_text(report, encoding="utf-8")
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "pro_eager_tps": pe_runs,
                "pro_graph_tps": pg_runs,
                "vllm_eager_tps": ve_runs,
                "vllm_graph_tps": vg_runs,
                "num_steps": num_steps,
                "exact": exact,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + report, end="")
    print(f"结果已保存：{report_path}")


if __name__ == "__main__":
    main()
