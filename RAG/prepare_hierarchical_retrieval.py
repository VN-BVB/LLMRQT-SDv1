"""运行 Parent→Child 两层混合检索并缓存重排后的 Top-K。"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

from bm25_retriever import BM25Retriever
from hybrid_retriever import reciprocal_rank_fusion_route_batches
from question_retriever import QuestionFaissRetriever
from rag_core import FaissRetriever, read_jsonl, write_json, write_jsonl
from reranker import BGEReranker, DEFAULT_RERANKER_MODEL


ROOT = Path(__file__).resolve().parent


def unique_source_ids(rows: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen = set()
    for row in rows:
        source_id = row["source_id"]
        if source_id not in seen:
            seen.add(source_id)
            result.append(source_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/ohr_bench/questions.jsonl")
    parser.add_argument(
        "--hierarchy-dir",
        type=Path,
        default=ROOT / "storage/ohr_bench_hierarchical_questions",
    )
    parser.add_argument("--embedding-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--parent-route-k", type=int, default=50)
    parser.add_argument("--parent-candidate-k", type=int, default=50)
    parser.add_argument("--parent-k", type=int, default=8)
    parser.add_argument("--parent-question-fetch-k", type=int, default=300)
    parser.add_argument("--child-route-k", type=int, default=50)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ohr_hierarchical_retrieval_cache",
    )
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    query_texts = [row["question"] for row in questions]
    parent_dir = args.hierarchy_dir / "parent"
    child_dir = args.hierarchy_dir / "child"

    parent_text = FaissRetriever(parent_dir, device=args.embedding_device)
    parent_question = QuestionFaissRetriever(
        parent_dir,
        chunks=parent_text.chunks,
        embedder=parent_text.embedder,
    )
    parent_bm25 = BM25Retriever(
        parent_dir,
        bm25_index=parent_dir / "bm25_questions",
        chunks=parent_text.chunks,
    )
    started = time.perf_counter()
    query_vectors = parent_text.embedder.encode(query_texts, is_query=True)
    parent_text_rows = parent_text.search_many(
        query_texts,
        top_k=args.parent_route_k,
        query_vectors=query_vectors,
    )
    parent_text_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parent_question_rows = parent_question.search_many(
        query_texts,
        top_k=args.parent_route_k,
        fetch_k=args.parent_question_fetch_k,
        query_vectors=query_vectors,
    )
    parent_question_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parent_bm25_rows = parent_bm25.search_many(query_texts, top_k=args.parent_route_k)
    parent_bm25_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parent_candidates = reciprocal_rank_fusion_route_batches(
        [
            ("summary_dense", parent_text_rows),
            ("summary_question", parent_question_rows),
            ("summary_bm25", parent_bm25_rows),
        ],
        rrf_k=args.rrf_k,
        top_k=args.parent_candidate_k,
    )
    parent_rrf_seconds = time.perf_counter() - started

    reranker = BGEReranker(
        model_name=args.reranker_model,
        device=args.reranker_device,
        max_length=args.reranker_max_length,
    )
    started = time.perf_counter()
    selected_parents = reranker.rerank_many(
        query_texts,
        parent_candidates,
        top_k=args.parent_k,
        batch_size=args.reranker_batch_size,
    )
    parent_reranker_seconds = time.perf_counter() - started
    allowed_parent_ids = [
        {parent["source_id"] for parent in parents} for parents in selected_parents
    ]

    # Child 文本和问题共用 Parent 已加载的 BGE，避免重复占用内存。
    child_text = FaissRetriever(
        child_dir,
        device=args.embedding_device,
        embedder=parent_text.embedder,
    )
    child_question = QuestionFaissRetriever(
        child_dir,
        chunks=child_text.chunks,
        embedder=child_text.embedder,
    )
    child_bm25 = BM25Retriever(
        child_dir,
        bm25_index=child_dir / "bm25_questions",
        chunks=child_text.chunks,
    )

    started = time.perf_counter()
    child_text_rows = child_text.search_many_filtered(
        query_texts,
        allowed_parent_ids,
        top_k=args.child_route_k,
        query_vectors=query_vectors,
    )
    child_text_seconds = time.perf_counter() - started

    started = time.perf_counter()
    child_question_rows = child_question.search_many_filtered(
        query_texts,
        allowed_parent_ids,
        top_k=args.child_route_k,
        query_vectors=query_vectors,
    )
    child_question_seconds = time.perf_counter() - started

    started = time.perf_counter()
    child_bm25_rows = child_bm25.search_many_filtered(
        query_texts,
        allowed_parent_ids,
        top_k=args.child_route_k,
    )
    child_bm25_seconds = time.perf_counter() - started

    started = time.perf_counter()
    child_candidates = reciprocal_rank_fusion_route_batches(
        [
            ("dense", child_text_rows),
            ("question_dense", child_question_rows),
            ("question_bm25", child_bm25_rows),
        ],
        rrf_k=args.rrf_k,
        top_k=args.candidate_k,
    )
    child_rrf_seconds = time.perf_counter() - started
    started = time.perf_counter()
    retrieved_rows = reranker.rerank_many(
        query_texts,
        child_candidates,
        top_k=args.top_k,
        batch_size=args.reranker_batch_size,
    )
    child_reranker_seconds = time.perf_counter() - started
    reranker.unload()

    parent_hit = sum(
        bool({row["source_id"] for row in parents} & set(question["gold_doc_ids"]))
        for question, parents in zip(questions, selected_parents)
    ) / len(questions)
    candidate_hit = sum(
        bool({row["source_id"] for row in rows} & set(question["gold_doc_ids"]))
        for question, rows in zip(questions, child_candidates)
    ) / len(questions)
    final_hit = sum(
        bool({row["source_id"] for row in rows} & set(question["gold_doc_ids"]))
        for question, rows in zip(questions, retrieved_rows)
    ) / len(questions)

    retrieval_seconds = sum(
        [
            parent_text_seconds,
            parent_question_seconds,
            parent_bm25_seconds,
            parent_rrf_seconds,
            parent_reranker_seconds,
            child_text_seconds,
            child_question_seconds,
            child_bm25_seconds,
            child_rrf_seconds,
            child_reranker_seconds,
        ]
    )
    timing = {
        "parent_text_dense_seconds": parent_text_seconds,
        "parent_question_dense_seconds": parent_question_seconds,
        "parent_bm25_seconds": parent_bm25_seconds,
        "parent_rrf_seconds": parent_rrf_seconds,
        "parent_reranker_seconds": parent_reranker_seconds,
        "dense_seconds": child_text_seconds,
        "question_dense_seconds": child_question_seconds,
        "expanded_bm25_seconds": child_bm25_seconds,
        "rrf_seconds": parent_rrf_seconds + child_rrf_seconds,
        "reranker_seconds": parent_reranker_seconds + child_reranker_seconds,
        "child_rrf_seconds": child_rrf_seconds,
        "child_reranker_seconds": child_reranker_seconds,
        "retrieval_pipeline_seconds": retrieval_seconds,
        "retrieval_pipeline_ms_per_query": retrieval_seconds * 1000 / len(questions),
    }
    parameters = {
        "limit": len(questions),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "dense_k": args.child_route_k,
        "parent_route_k": args.parent_route_k,
        "parent_candidate_k": args.parent_candidate_k,
        "parent_k": args.parent_k,
        "child_route_k": args.child_route_k,
        "child_scope": "exact_parent_filtered_search",
        "rrf_k": args.rrf_k,
        "embedding_device": args.embedding_device,
        "reranker_model": args.reranker_model,
        "reranker_device": args.reranker_device,
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_max_length": args.reranker_max_length,
    }
    write_json(
        args.output_dir / "retrieval_meta.json",
        {
            "mode": "hierarchical_parent_child_text_question_bm25_rrf",
            "question_count": len(questions),
            "parameters": parameters,
            "timing": timing,
            "hierarchical_quality": {
                "parent_hit_at_k": parent_hit,
                "candidate_hit_at_k": candidate_hit,
                "final_hit_at_k": final_hit,
            },
        },
    )
    output_rows = []
    for question, dense, parents, candidates, retrieved in zip(
        questions, child_text_rows, selected_parents, child_candidates, retrieved_rows
    ):
        output_rows.append(
            {
                "id": question["id"],
                "dense_doc_ids": unique_source_ids(dense[: args.top_k]),
                "parent_doc_ids": unique_source_ids(parents),
                "candidate_doc_ids": unique_source_ids(candidates),
                "retrieved": retrieved,
            }
        )
    write_jsonl(args.output_dir / "retrieval_rows.jsonl", output_rows)
    print(
        f"Parent Hit@{args.parent_k}={parent_hit:.4f}, "
        f"Candidate Hit@{args.candidate_k}={candidate_hit:.4f}, "
        f"Final Hit@{args.top_k}={final_hit:.4f}"
    )
    print(f"两层检索缓存：{args.output_dir}")


if __name__ == "__main__":
    main()
