"""用原问题及其查询变体做 Dense 或 Dense+BM25+RRF，再以原问题执行 Reranker。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from bm25_retriever import BM25Retriever
from hybrid_retriever import reciprocal_rank_fusion_routes
from prepare_hierarchical_retrieval import aggregate_rank_metrics, unique_source_ids
from query_transform import selected_queries
from rag_core import FaissRetriever, read_jsonl, write_json, write_jsonl
from reranker import BGEReranker, DEFAULT_RERANKER_MODEL


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/ohr_bench/questions.jsonl")
    parser.add_argument("--variants", type=Path, default=ROOT / "data/ohr_bench/query_variants.jsonl")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument("--embedding-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--retrieval-routes",
        choices=["dense", "dense-bm25"],
        default="dense-bm25",
        help="dense 是最小基线；dense-bm25 以 RRF 融合两条召回路由",
    )
    parser.add_argument(
        "--variant-mode",
        choices=["original", "rewrite", "step_back", "decompose", "all"],
        required=True,
        help="original 是不做查询变换的严格对照组",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dense-k", type=int, default=50)
    parser.add_argument("--bm25-k", type=int, default=50)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    variants = read_jsonl(args.variants)
    if args.limit:
        questions, variants = questions[: args.limit], variants[: args.limit]
    if len(questions) != len(variants) or any(
        question["id"] != variant["id"] for question, variant in zip(questions, variants)
    ):
        raise ValueError("questions 与 query variants 的数量或顺序不一致")

    grouped_queries = [selected_queries(row, args.variant_mode) for row in variants]
    flat_queries = [query for group in grouped_queries for query in group]
    group_offsets = []
    cursor = 0
    for group in grouped_queries:
        group_offsets.append((cursor, cursor + len(group)))
        cursor += len(group)

    retriever = FaissRetriever(args.index_dir, device=args.embedding_device)
    started = time.perf_counter()
    flat_dense = retriever.search_many(flat_queries, top_k=args.dense_k)
    dense_seconds = time.perf_counter() - started
    flat_bm25: list[list[dict[str, Any]]] | None = None
    bm25_seconds = 0.0
    if args.retrieval_routes == "dense-bm25":
        bm25 = BM25Retriever(args.index_dir, chunks=retriever.chunks)
        started = time.perf_counter()
        flat_bm25 = bm25.search_many(flat_queries, top_k=args.bm25_k)
        bm25_seconds = time.perf_counter() - started

    started = time.perf_counter()
    candidates: list[list[dict[str, Any]]] = []
    for begin, end in group_offsets:
        if args.retrieval_routes == "dense" and end - begin == 1:
            candidates.append(flat_dense[begin][: args.candidate_k])
            continue
        routes = []
        for offset, flat_index in enumerate(range(begin, end)):
            routes.append((f"dense_q{offset}", flat_dense[flat_index]))
            if flat_bm25 is not None:
                routes.append((f"bm25_q{offset}", flat_bm25[flat_index]))
        candidates.append(
            reciprocal_rank_fusion_routes(
                routes, rrf_k=args.rrf_k, top_k=args.candidate_k
            )
        )
    rrf_seconds = time.perf_counter() - started

    retriever.embedder.unload()
    reranker = BGEReranker(
        model_name=args.reranker_model,
        device=args.reranker_device,
        max_length=args.reranker_max_length,
    )
    originals = [row["question"] for row in questions]
    started = time.perf_counter()
    retrieved = reranker.rerank_many(
        originals,
        candidates,
        top_k=args.top_k,
        batch_size=args.reranker_batch_size,
    )
    reranker_seconds = time.perf_counter() - started
    reranker.unload()

    # 严格对照组在线时不会调用查询变换 LLM，因此不能把离线生成变体的耗时
    # 计入 original 的检索流水线。
    transform_seconds = (
        0.0
        if args.variant_mode == "original"
        else sum(float(row.get("transform_latency_seconds", 0.0)) for row in variants)
    )
    retrieval_only_seconds = dense_seconds + bm25_seconds + rrf_seconds + reranker_seconds
    total_seconds = transform_seconds + retrieval_only_seconds
    candidate_quality = aggregate_rank_metrics(questions, candidates, args.candidate_k)
    final_quality = aggregate_rank_metrics(questions, retrieved, args.top_k)
    timing = {
        "query_transform_seconds": transform_seconds,
        "query_transform_ms_per_query": transform_seconds * 1000 / len(questions),
        "dense_seconds": dense_seconds,
        "dense_ms_per_query": dense_seconds * 1000 / len(questions),
        "bm25_seconds": bm25_seconds,
        "bm25_ms_per_query": bm25_seconds * 1000 / len(questions),
        "rrf_seconds": rrf_seconds,
        "rrf_ms_per_query": rrf_seconds * 1000 / len(questions),
        "reranker_seconds": reranker_seconds,
        "reranker_ms_per_query": reranker_seconds * 1000 / len(questions),
        "retrieval_without_transform_seconds": retrieval_only_seconds,
        "retrieval_without_transform_ms_per_query": retrieval_only_seconds * 1000 / len(questions),
        "retrieval_pipeline_seconds": total_seconds,
        "retrieval_pipeline_ms_per_query": total_seconds * 1000 / len(questions),
    }
    parameters = {
        "limit": len(questions),
        "variant_mode": args.variant_mode,
        "retrieval_routes": args.retrieval_routes,
        "average_queries_per_question": sum(map(len, grouped_queries)) / len(grouped_queries),
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "embedding_device": args.embedding_device,
        "reranker_model": args.reranker_model,
        "reranker_device": args.reranker_device,
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_max_length": args.reranker_max_length,
    }
    route_label = (
        "dense" if args.retrieval_routes == "dense" else "dense_bm25_rrf"
    )
    write_json(
        args.output_dir / "retrieval_meta.json",
        {
            "mode": (
                f"{args.variant_mode}_{route_label}_reranker"
                if args.variant_mode == "original"
                else (
                    f"query_transform_{args.variant_mode}_"
                    f"{route_label}_reranker"
                )
            ),
            "question_count": len(questions),
            "parameters": parameters,
            "timing": timing,
            "retrieval_quality": {
                "candidate": candidate_quality,
                "final": final_quality,
            },
        },
    )
    rows = []
    for question, (begin, _), candidate, final in zip(
        questions, group_offsets, candidates, retrieved
    ):
        rows.append(
            {
                "id": question["id"],
                "dense_doc_ids": unique_source_ids(flat_dense[begin][: args.top_k]),
                "candidate_doc_ids": unique_source_ids(candidate),
                "retrieved": final,
            }
        )
    write_jsonl(args.output_dir / "retrieval_rows.jsonl", rows)
    print(
        f"{args.variant_mode}: Final Hit@{args.top_k}={final_quality['hit_at_k']:.4f}, "
        f"Precision={final_quality['precision_at_k']:.4f}, "
        f"Recall={final_quality['recall_at_k']:.4f}, "
        f"MRR={final_quality['mrr_at_k']:.4f}, "
        f"nDCG={final_quality['ndcg_at_k']:.4f}, "
        f"total={timing['retrieval_pipeline_ms_per_query']:.1f} ms/query"
    )


if __name__ == "__main__":
    main()
