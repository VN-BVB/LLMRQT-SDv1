"""Prepare four OHR retrieval variants and cache reranked Top-K for generation."""

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
    result = []
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
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--question-index", type=Path)
    parser.add_argument("--bm25-index", type=Path)
    parser.add_argument("--expanded-bm25-index", type=Path)
    parser.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-k", type=int, default=50)
    parser.add_argument("--bm25-k", type=int, default=50)
    parser.add_argument("--question-k", type=int, default=50)
    parser.add_argument("--question-fetch-k", type=int, default=300)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ohr_question_experiment_retrieval_cache",
    )
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    query_texts = [row["question"] for row in questions]
    retriever = FaissRetriever(args.index_dir, device=args.embedding_device)

    started = time.perf_counter()
    query_vectors = retriever.embedder.encode(query_texts, is_query=True)
    query_embedding_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dense_rows = retriever.search_many(
        query_texts,
        top_k=args.dense_k,
        query_vectors=query_vectors,
    )
    dense_index_seconds = time.perf_counter() - started

    bm25_seconds = 0.0

    expanded_bm25 = BM25Retriever(
        args.index_dir,
        bm25_index=args.expanded_bm25_index or args.index_dir / "bm25_questions",
        chunks=retriever.chunks,
    )
    started = time.perf_counter()
    expanded_bm25_rows = expanded_bm25.search_many(query_texts, top_k=args.bm25_k)
    expanded_bm25_seconds = time.perf_counter() - started

    question_retriever = QuestionFaissRetriever(
        args.index_dir,
        question_index=args.question_index or args.index_dir / "questions",
        chunks=retriever.chunks,
        embedder=retriever.embedder,
    )
    started = time.perf_counter()
    question_rows = question_retriever.search_many(
        query_texts,
        top_k=args.question_k,
        fetch_k=args.question_fetch_k,
        query_vectors=query_vectors,
    )
    question_seconds = time.perf_counter() - started

    candidates: dict[str, list[list[dict[str, Any]]]] = {}
    rrf_timings: dict[str, float] = {}

    # 四组严格消融：文本 Dense、问题 Dense、文本+问题 BM25、三路 RRF。
    candidates["no_questions"] = [rows[: args.candidate_k] for rows in dense_rows]
    rrf_timings["no_questions"] = 0.0

    candidates["question_bge"] = [rows[: args.candidate_k] for rows in question_rows]
    rrf_timings["question_bge"] = 0.0

    candidates["question_bm25"] = [
        rows[: args.candidate_k] for rows in expanded_bm25_rows
    ]
    rrf_timings["question_bm25"] = 0.0

    started = time.perf_counter()
    candidates["triple_rrf"] = reciprocal_rank_fusion_route_batches(
        [
            ("dense", dense_rows),
            ("question_bm25", expanded_bm25_rows),
            ("question_dense", question_rows),
        ],
        rrf_k=args.rrf_k,
        top_k=args.candidate_k,
    )
    rrf_timings["triple_rrf"] = time.perf_counter() - started

    # BGE 只在 CPU/GPU 编码阶段使用；重排前释放，避免模型并存。
    del question_retriever, retriever, expanded_bm25
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    reranker = BGEReranker(
        model_name=args.reranker_model,
        device=args.reranker_device,
        max_length=args.reranker_max_length,
    )
    reranked: dict[str, list[list[dict[str, Any]]]] = {}
    reranker_timings: dict[str, float] = {}
    for mode, rows in candidates.items():
        print(f"正在重排：{mode}")
        started = time.perf_counter()
        reranked[mode] = reranker.rerank_many(
            query_texts,
            rows,
            top_k=args.top_k,
            batch_size=args.reranker_batch_size,
        )
        reranker_timings[mode] = time.perf_counter() - started
    reranker.unload()

    component_seconds = {
        "query_embedding": query_embedding_seconds,
        "dense": dense_index_seconds,
        "bm25": bm25_seconds,
        "expanded_bm25": expanded_bm25_seconds,
        "question_dense": question_seconds,
    }
    mode_components = {
        "no_questions": ["query_embedding", "dense"],
        "question_bge": ["query_embedding", "question_dense"],
        "question_bm25": ["expanded_bm25"],
        "triple_rrf": [
            "query_embedding",
            "dense",
            "expanded_bm25",
            "question_dense",
        ],
    }
    common_parameters = {
        "limit": len(questions),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "question_k": args.question_k,
        "question_fetch_k": args.question_fetch_k,
        "rrf_k": args.rrf_k,
        "embedding_device": args.embedding_device,
        "reranker_model": args.reranker_model,
        "reranker_device": args.reranker_device,
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_max_length": args.reranker_max_length,
    }

    for mode, retrieved_rows in reranked.items():
        output = args.output_dir / mode
        retrieval_seconds = sum(component_seconds[name] for name in mode_components[mode])
        retrieval_seconds += rrf_timings[mode] + reranker_timings[mode]
        timing = {
            "query_embedding_seconds": (
                query_embedding_seconds
                if "query_embedding" in mode_components[mode]
                else 0.0
            ),
            "query_embedding_ms_per_query": (
                query_embedding_seconds * 1000 / len(questions)
                if "query_embedding" in mode_components[mode]
                else 0.0
            ),
            "dense_seconds": (
                query_embedding_seconds + dense_index_seconds
                if "dense" in mode_components[mode]
                else 0.0
            ),
            "dense_ms_per_query": (
                (query_embedding_seconds + dense_index_seconds) * 1000 / len(questions)
                if "dense" in mode_components[mode]
                else 0.0
            ),
            "bm25_seconds": (
                bm25_seconds if "bm25" in mode_components[mode] else 0.0
            ),
            "bm25_ms_per_query": (
                bm25_seconds * 1000 / len(questions)
                if "bm25" in mode_components[mode]
                else 0.0
            ),
            "expanded_bm25_seconds": (
                expanded_bm25_seconds if "expanded_bm25" in mode_components[mode] else 0.0
            ),
            "expanded_bm25_ms_per_query": (
                expanded_bm25_seconds * 1000 / len(questions)
                if "expanded_bm25" in mode_components[mode]
                else 0.0
            ),
            "question_dense_seconds": (
                (
                    question_seconds
                    + (query_embedding_seconds if "dense" not in mode_components[mode] else 0.0)
                )
                if "question_dense" in mode_components[mode]
                else 0.0
            ),
            "question_dense_ms_per_query": (
                (
                    question_seconds
                    + (query_embedding_seconds if "dense" not in mode_components[mode] else 0.0)
                )
                * 1000
                / len(questions)
                if "question_dense" in mode_components[mode]
                else 0.0
            ),
            "rrf_seconds": rrf_timings[mode],
            "rrf_ms_per_query": rrf_timings[mode] * 1000 / len(questions),
            "reranker_seconds": reranker_timings[mode],
            "reranker_ms_per_query": reranker_timings[mode] * 1000 / len(questions),
            "retrieval_pipeline_seconds": retrieval_seconds,
            "retrieval_pipeline_ms_per_query": retrieval_seconds * 1000 / len(questions),
        }
        write_json(
            output / "retrieval_meta.json",
            {
                "mode": mode,
                "question_count": len(questions),
                "parameters": common_parameters,
                "timing": timing,
            },
        )
        rows_to_save = []
        for question, dense, candidate, retrieved in zip(
            questions, dense_rows, candidates[mode], retrieved_rows
        ):
            rows_to_save.append(
                {
                    "id": question["id"],
                    "dense_doc_ids": unique_source_ids(dense[: args.top_k]),
                    "candidate_doc_ids": unique_source_ids(candidate),
                    "retrieved": retrieved,
                }
            )
        write_jsonl(output / "retrieval_rows.jsonl", rows_to_save)
        print(f"已保存 {mode}：{output}")


if __name__ == "__main__":
    main()
