"""评测 Dense、BM25、RRF 与可选 Reranker，不启动生成模型。"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from bm25_retriever import BM25Retriever
from hybrid_retriever import reciprocal_rank_fusion_many
from rag_core import FaissRetriever, read_jsonl, write_json, write_jsonl
from reranker import BGEReranker, DEFAULT_RERANKER_MODEL


ROOT = Path(__file__).resolve().parent
METRIC_NAMES = ("hit", "recall", "mrr", "ndcg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/ohr_bench/questions.jsonl")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--limit", type=int, default=0, help="0 表示评测全部问题")
    parser.add_argument("--hybrid", action="store_true", help="启用 Dense + BM25 + RRF")
    parser.add_argument("--bm25-index", type=Path, help="默认 INDEX_DIR/bm25/")
    parser.add_argument("--dense-k", type=int, default=50, help="Hybrid 的 Dense 召回 Chunk 数")
    parser.add_argument("--bm25-k", type=int, default=50, help="Hybrid 的 BM25 召回 Chunk 数")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 排名平滑常数")
    parser.add_argument("--reranker", action="store_true", help="对 Dense 或 RRF 候选继续重排")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--candidate-k", type=int, default=50, help="送入 Reranker 的候选 Chunk 数")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=20,
        help="纯 Dense 且无 Reranker 时，多召回 Chunk 后按页面去重",
    )
    parser.add_argument("--output-dir", type=Path, help="默认根据启用组件自动选择结果目录")
    return parser.parse_args()


def metrics_at_k(retrieved_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> dict[str, float]:
    if not gold_doc_ids:
        raise ValueError("gold_doc_ids 不能为空")
    top = retrieved_doc_ids[:k]
    relevant_count = len(set(top) & gold_doc_ids)
    hit = float(relevant_count > 0)
    recall = relevant_count / len(gold_doc_ids)
    reciprocal_rank = next(
        (1.0 / rank for rank, doc_id in enumerate(top, start=1) if doc_id in gold_doc_ids),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top, start=1)
        if doc_id in gold_doc_ids
    )
    ideal_count = min(len(gold_doc_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "hit": hit,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def unique_source_ids(retrieved: list[dict[str, Any]]) -> list[str]:
    source_ids = []
    seen = set()
    for item in retrieved:
        source_id = item["source_id"]
        if source_id not in seen:
            seen.add(source_id)
            source_ids.append(source_id)
    return source_ids


def compact_chunks(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    keys = (
        "id", "source_id", "rank", "score",
        "dense_rank", "dense_score", "bm25_rank", "bm25_score",
        "fusion_rank", "fusion_score", "rrf_score",
        "retrieval_rank", "retrieval_score", "rerank_score",
    )
    return [{key: row[key] for key in keys if key in row} for row in rows[:limit]]


def evaluate_stages(
    questions: list[dict[str, Any]],
    stages: dict[str, list[list[dict[str, Any]]]],
    ks: list[int],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    totals = {
        stage: {k: {metric: 0.0 for metric in METRIC_NAMES} for k in ks}
        for stage in stages
    }
    candidate_totals = {stage: {"hit": 0.0, "recall": 0.0} for stage in stages}
    predictions = []

    for question_index, question in enumerate(questions):
        gold = set(question["gold_doc_ids"])
        prediction: dict[str, Any] = {
            "id": question["id"],
            "question": question["question"],
            "gold_doc_ids": sorted(gold),
            "stages": {},
        }
        for stage, all_rows in stages.items():
            rows = all_rows[question_index]
            doc_ids = unique_source_ids(rows)
            candidate_relevant = len(set(doc_ids) & gold)
            candidate = {
                "hit": float(candidate_relevant > 0),
                "recall": candidate_relevant / len(gold),
            }
            for metric, value in candidate.items():
                candidate_totals[stage][metric] += value

            per_k = {}
            for k in ks:
                values = metrics_at_k(doc_ids, gold, k)
                per_k[str(k)] = values
                for metric, value in values.items():
                    totals[stage][k][metric] += value
            prediction["stages"][stage] = {
                "candidate_metrics": candidate,
                "doc_ids": doc_ids[: max(ks)],
                "chunks": compact_chunks(rows, max(ks)),
                "metrics": per_k,
            }
        predictions.append(prediction)

    count = len(questions)
    reports = {}
    for stage in stages:
        raw_metrics = {
            str(k): {
                metric: totals[stage][k][metric] / count for metric in METRIC_NAMES
            }
            for k in ks
        }
        reports[stage] = {
            "candidate_metrics": {
                metric: candidate_totals[stage][metric] / count
                for metric in ("hit", "recall")
            },
            "metrics": raw_metrics,
            "metrics_percent": {
                str(k): {metric: value * 100 for metric, value in raw_metrics[str(k)].items()}
                for k in ks
            },
        }
    return reports, predictions


def metric_deltas(
    baseline: dict[str, dict[str, float]],
    experiment: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    deltas = {}
    for k, experiment_values in experiment.items():
        deltas[k] = {}
        for metric, value in experiment_values.items():
            base = baseline[k][metric]
            absolute = value - base
            deltas[k][metric] = {
                "absolute": absolute,
                "percentage_points": absolute * 100,
                "relative_percent": absolute / base * 100 if base else None,
            }
    return deltas


def format_metric(value: float) -> str:
    return f"{value:.4f} ({value * 100:.2f}%)"


def print_stage_metrics(title: str, report: dict[str, Any], ks: list[int]) -> None:
    print(f"\n{title}")
    print(" K | Hit@K              | Recall@K           | MRR@K              | nDCG@K")
    print("---|--------------------|--------------------|--------------------|-------------------")
    for k in ks:
        values = report["metrics"][str(k)]
        print(
            f"{k:2d} | {format_metric(values['hit']):18s} | "
            f"{format_metric(values['recall']):18s} | "
            f"{format_metric(values['mrr']):18s} | {format_metric(values['ndcg'])}"
        )


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.hybrid and args.reranker:
        name = "ohr_bench_retrieval_hybrid_reranker"
    elif args.hybrid:
        name = "ohr_bench_retrieval_hybrid"
    elif args.reranker:
        name = "ohr_bench_retrieval_reranker"
    else:
        name = "ohr_bench_retrieval_dense"
    return ROOT / "results" / name


def main() -> None:
    args = parse_args()
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("评测问题为空")

    ks = sorted(set(args.k))
    if not ks or min(ks) <= 0:
        raise ValueError("所有 k 必须大于 0")
    for name in ("candidate_k", "dense_k", "bm25_k", "rrf_k", "candidate_multiplier"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} 必须大于 0")
    if (args.reranker or args.hybrid) and args.candidate_k < max(ks):
        raise ValueError("candidate-k 必须大于等于最大的 k")
    if args.reranker_batch_size <= 0:
        raise ValueError("reranker-batch-size 必须大于 0")
    args.output_dir = args.output_dir or default_output_dir(args)

    query_texts = [item["question"] for item in questions]
    retriever = FaissRetriever(args.index_dir, device=args.embedding_device)
    if args.hybrid:
        dense_fetch_k = args.dense_k
    elif args.reranker:
        dense_fetch_k = args.candidate_k
    else:
        dense_fetch_k = max(ks) * args.candidate_multiplier
    dense_fetch_k = min(dense_fetch_k, len(retriever.chunks))

    dense_started = time.perf_counter()
    dense_rows = retriever.search_many(query_texts, top_k=dense_fetch_k)
    dense_seconds = time.perf_counter() - dense_started
    stages: dict[str, list[list[dict[str, Any]]]] = {"dense": dense_rows}
    timings = {
        "dense_seconds": dense_seconds,
        "dense_ms_per_query": dense_seconds * 1000 / len(questions),
    }

    candidate_rows = dense_rows
    if args.hybrid:
        bm25 = BM25Retriever(
            args.index_dir,
            bm25_index=args.bm25_index,
            chunks=retriever.chunks,
        )
        bm25_started = time.perf_counter()
        bm25_rows = bm25.search_many(query_texts, top_k=args.bm25_k)
        bm25_seconds = time.perf_counter() - bm25_started
        timings.update({
            "bm25_seconds": bm25_seconds,
            "bm25_ms_per_query": bm25_seconds * 1000 / len(questions),
        })
        stages["bm25"] = bm25_rows

        rrf_started = time.perf_counter()
        rrf_rows = reciprocal_rank_fusion_many(
            dense_rows, bm25_rows, rrf_k=args.rrf_k, top_k=args.candidate_k
        )
        rrf_seconds = time.perf_counter() - rrf_started
        timings.update({
            "rrf_seconds": rrf_seconds,
            "rrf_ms_per_query": rrf_seconds * 1000 / len(questions),
        })
        stages["rrf"] = rrf_rows
        candidate_rows = rrf_rows

    if args.reranker:
        print(f"正在加载 Reranker：{args.reranker_model} ({args.reranker_device})")
        reranker = BGEReranker(
            model_name=args.reranker_model,
            device=args.reranker_device,
            max_length=args.reranker_max_length,
        )
        reranker_started = time.perf_counter()
        reranked_rows = reranker.rerank_many(
            query_texts, candidate_rows, top_k=None, batch_size=args.reranker_batch_size
        )
        reranker_seconds = time.perf_counter() - reranker_started
        timings.update({
            "reranker_seconds": reranker_seconds,
            "reranker_ms_per_query": reranker_seconds * 1000 / len(questions),
        })
        reranker.unload()
        stages["reranker"] = reranked_rows

    reports, predictions = evaluate_stages(questions, stages, ks)
    selected_stage = "reranker" if args.reranker else ("rrf" if args.hybrid else "dense")
    deltas = {
        stage: metric_deltas(reports["dense"]["metrics"], report["metrics"])
        for stage, report in reports.items()
        if stage != "dense"
    }
    pipeline_seconds = sum(
        value for key, value in timings.items() if key.endswith("_seconds")
    )
    timings.update({
        "retrieval_pipeline_seconds": pipeline_seconds,
        "retrieval_pipeline_ms_per_query": pipeline_seconds * 1000 / len(questions),
    })

    summary = {
        "question_count": len(questions),
        "mode": selected_stage,
        "parameters": {
            "dense_k": dense_fetch_k,
            "bm25_k": args.bm25_k if args.hybrid else None,
            "rrf_k": args.rrf_k if args.hybrid else None,
            "candidate_k": args.candidate_k if (args.hybrid or args.reranker) else dense_fetch_k,
            "k": ks,
        },
        "timing": timings,
        "stages": reports,
        "deltas_vs_dense": deltas,
        "candidate_metrics": reports[selected_stage]["candidate_metrics"],
        "dense_metrics": reports["dense"]["metrics"],
        "metrics": reports[selected_stage]["metrics"],
    }
    if args.reranker:
        summary["reranker"] = {
            "model": args.reranker_model,
            "device": args.reranker_device,
            "batch_size": args.reranker_batch_size,
            "max_length": args.reranker_max_length,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)

    print(f"\n评测问题数：{len(questions)}")
    for stage, report in reports.items():
        candidate = report["candidate_metrics"]
        print(
            f"{stage.upper()} 候选集：Hit={format_metric(candidate['hit'])}, "
            f"Recall={format_metric(candidate['recall'])}"
        )
        print_stage_metrics(stage.upper(), report, ks)

    print("\n相对 Dense 的变化（百分点 / 相对百分比）")
    for stage, stage_deltas in deltas.items():
        print(f"  {stage.upper()}:")
        for k in ks:
            values = stage_deltas[str(k)]
            hit = values["hit"]
            ndcg = values["ndcg"]
            hit_relative = "N/A" if hit["relative_percent"] is None else f"{hit['relative_percent']:+.2f}%"
            ndcg_relative = "N/A" if ndcg["relative_percent"] is None else f"{ndcg['relative_percent']:+.2f}%"
            print(
                f"    K={k}: Hit {hit['percentage_points']:+.2f}pp ({hit_relative}), "
                f"nDCG {ndcg['percentage_points']:+.2f}pp ({ndcg_relative})"
            )

    print("\n延迟")
    for key, value in timings.items():
        if key.endswith("_ms_per_query"):
            print(f"  {key}: {value:.2f} ms/query")
    print(f"详细结果：{args.output_dir}")


if __name__ == "__main__":
    main()
