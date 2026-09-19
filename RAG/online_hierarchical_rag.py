"""在线两层 RAG：Parent Summary 定位页面，再在页内检索精细 Child。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from bm25_retriever import BM25Retriever
from generation_backends import DEFAULT_AWQ_MODEL, DEFAULT_LLMQRT_ROOT, create_backend
from hybrid_retriever import reciprocal_rank_fusion_routes
from online_rag import build_rag_messages
from question_retriever import QuestionFaissRetriever
from rag_core import FaissRetriever
from reranker import BGEReranker, DEFAULT_RERANKER_MODEL


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchy-dir",
        type=Path,
        default=ROOT / "storage/ohr_bench_hierarchical_questions",
    )
    parser.add_argument("--question", help="不填则进入连续提问")
    parser.add_argument("--embedding-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--parent-route-k", type=int, default=50)
    parser.add_argument("--parent-candidate-k", type=int, default=50)
    parser.add_argument("--parent-k", type=int, default=8)
    parser.add_argument("--child-route-k", type=int, default=50)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--backend", choices=["vllm", "local-awq"], default="vllm")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--awq-model", type=Path, default=DEFAULT_AWQ_MODEL)
    parser.add_argument("--llmqrt-root", type=Path, default=DEFAULT_LLMQRT_ROOT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    parent_dir = args.hierarchy_dir / "parent"
    child_dir = args.hierarchy_dir / "child"
    print("加载 Parent/Child 文本 FAISS、问题 FAISS 与 BM25……")
    parent_text = FaissRetriever(parent_dir, device=args.embedding_device)
    parent_question = QuestionFaissRetriever(
        parent_dir, chunks=parent_text.chunks, embedder=parent_text.embedder
    )
    parent_bm25 = BM25Retriever(
        parent_dir, bm25_index=parent_dir / "bm25_questions", chunks=parent_text.chunks
    )
    child_text = FaissRetriever(child_dir, embedder=parent_text.embedder)
    child_question = QuestionFaissRetriever(
        child_dir, chunks=child_text.chunks, embedder=parent_text.embedder
    )
    child_bm25 = BM25Retriever(
        child_dir, bm25_index=child_dir / "bm25_questions", chunks=child_text.chunks
    )
    reranker = BGEReranker(
        model_name=args.reranker_model,
        device=args.reranker_device,
        max_length=args.reranker_max_length,
    )
    client = None
    if not args.retrieve_only:
        client = create_backend(
            args.backend,
            api_base=args.api_base,
            model=args.model,
            awq_model=args.awq_model,
            llmqrt_root=args.llmqrt_root,
        )

    def answer(question: str) -> None:
        started = time.perf_counter()
        query_vectors = parent_text.embedder.encode([question], is_query=True)
        parent_candidates = reciprocal_rank_fusion_routes(
            [
                (
                    "summary_dense",
                    parent_text.search_many(
                        [question],
                        args.parent_route_k,
                        query_vectors=query_vectors,
                    )[0],
                ),
                (
                    "summary_question",
                    parent_question.search_many(
                        [question],
                        top_k=args.parent_route_k,
                        query_vectors=query_vectors,
                    )[0],
                ),
                ("summary_bm25", parent_bm25.search(question, args.parent_route_k)),
            ],
            rrf_k=args.rrf_k,
            top_k=args.parent_candidate_k,
        )
        parents = reranker.rerank(
            question,
            parent_candidates,
            top_k=args.parent_k,
            batch_size=args.reranker_batch_size,
        )
        allowed = [{row["source_id"] for row in parents}]
        child_candidates = reciprocal_rank_fusion_routes(
            [
                (
                    "dense",
                    child_text.search_many_filtered(
                        [question],
                        allowed,
                        top_k=args.child_route_k,
                        query_vectors=query_vectors,
                    )[0],
                ),
                (
                    "question_dense",
                    child_question.search_many_filtered(
                        [question],
                        allowed,
                        top_k=args.child_route_k,
                        query_vectors=query_vectors,
                    )[0],
                ),
                (
                    "question_bm25",
                    child_bm25.search_many_filtered(
                        [question], allowed, top_k=args.child_route_k
                    )[0],
                ),
            ],
            rrf_k=args.rrf_k,
            top_k=args.candidate_k,
        )
        retrieved = reranker.rerank(
            question,
            child_candidates,
            top_k=args.top_k,
            batch_size=args.reranker_batch_size,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        if args.show_context or args.retrieve_only:
            print("\nParent 页：", [row["source_id"] for row in parents])
            for row in retrieved:
                print(
                    f"\n[{row['id']}] page={row.get('page')} "
                    f"rerank={row.get('rerank_score', 0):.4f}\n{row['text']}"
                )
        if client is not None:
            generation_started = time.perf_counter()
            response = client.chat(
                build_rag_messages(question, retrieved),
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            generation_ms = (time.perf_counter() - generation_started) * 1000
            print(f"\n{response}")
            print(
                f"\n延迟：检索 {retrieval_ms:.1f} ms，生成 {generation_ms:.1f} ms，"
                f"合计 {retrieval_ms + generation_ms:.1f} ms"
            )
        else:
            print(f"\n检索延迟：{retrieval_ms:.1f} ms")

    if args.question:
        answer(args.question)
    else:
        while True:
            question = input("\n问题（输入 exit 退出）：").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                answer(question)


if __name__ == "__main__":
    main()
