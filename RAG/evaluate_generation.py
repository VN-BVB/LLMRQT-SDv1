"""端到端评测：评测 RAG 的 EM/F1、nDCG 和延迟，可选无 RAG 对照。"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import time
from collections import Counter
from pathlib import Path

from bm25_retriever import BM25Retriever
from generation_backends import DEFAULT_AWQ_MODEL, DEFAULT_LLMQRT_ROOT, create_backend
from hybrid_retriever import reciprocal_rank_fusion_many
from online_rag import build_eval_no_rag_messages, build_eval_rag_messages
from rag_core import FaissRetriever, read_jsonl, write_json, write_jsonl
from reranker import BGEReranker, DEFAULT_RERANKER_MODEL


ROOT = Path(__file__).resolve().parent
CHINESE_PUNCTUATION = "，。！？；：、“”‘’（）《》【】…—·"


def normalize_answer(text: str) -> str:
    """用于自动评分：小写，并去掉空格与中英文标点。"""

    punctuation = set(string.punctuation + CHINESE_PUNCTUATION)
    return "".join(char for char in text.lower() if not char.isspace() and char not in punctuation)


def exact_match(prediction: str, answers: list[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(answer) for answer in answers))


def character_f1(prediction: str, answers: list[str]) -> float:
    """中文常用字级 F1；取多个标准答案中的最高分。"""

    predicted = list(normalize_answer(prediction))
    best = 0.0
    for answer in answers:
        gold = list(normalize_answer(answer))
        common = Counter(predicted) & Counter(gold)
        same = sum(common.values())
        if not predicted or not gold:
            score = float(predicted == gold)
        elif same == 0:
            score = 0.0
        else:
            precision = same / len(predicted)
            recall = same / len(gold)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def clean_model_answer(text: str) -> str:
    """评分前去掉模型按要求产生的 [资料1] 引用标记。"""

    return re.sub(r"\[资料\s*\d+\]", "", text).strip()


def ndcg_at_k(retrieved_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    """计算最终 K 个 chunk 的页面级二元 nDCG；同页重复只计第一次。"""

    if not gold_doc_ids:
        raise ValueError("gold_doc_ids 不能为空")
    seen = set()
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in gold_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_count = min(len(gold_doc_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/ohr_bench/questions.jsonl")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hybrid", action="store_true", help="使用 Dense + BM25 + RRF")
    parser.add_argument("--bm25-index", type=Path, help="默认 INDEX_DIR/bm25/")
    parser.add_argument("--dense-k", type=int, default=50)
    parser.add_argument("--bm25-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker", action="store_true", help="使用 Dense 候选 + Cross-Encoder")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=50, help="先用 50 条快速观察；0 表示全部")
    parser.add_argument("--backend", choices=["vllm", "local-awq"], default="vllm")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", help="vLLM served model name；默认自动查询")
    parser.add_argument("--awq-model", type=Path, default=DEFAULT_AWQ_MODEL)
    parser.add_argument("--llmqrt-root", type=Path, default=DEFAULT_LLMQRT_ROOT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-no-rag",
        action="store_true",
        help="跳过无 RAG 对照生成；用于只比较 RAG 与 RAG+Reranker",
    )
    parser.add_argument("--output-dir", type=Path, help="默认按 Dense/Reranker 分目录，避免覆盖")
    parser.add_argument(
        "--retrieval-cache",
        type=Path,
        help="读取 prepare_question_retrievals.py 生成的缓存，只并行评测答案生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("评测问题为空")
    if args.top_k <= 0:
        raise ValueError("top-k 必须大于 0")
    if (args.reranker or args.hybrid) and args.candidate_k < args.top_k:
        raise ValueError("启用 Hybrid/Reranker 时 candidate-k 必须大于等于 top-k")
    for name in ("dense_k", "bm25_k", "rrf_k"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} 必须大于 0")
    if args.reranker and args.reranker_batch_size <= 0:
        raise ValueError("reranker-batch-size 必须大于 0")
    cache_meta = None
    if args.retrieval_cache is not None:
        with (args.retrieval_cache / "retrieval_meta.json").open("r", encoding="utf-8") as file:
            cache_meta = json.load(file)
    if args.output_dir is None:
        if cache_meta is not None:
            name = f"ohr_question_generation_{cache_meta['mode']}"
        elif args.hybrid and args.reranker:
            name = "ohr_bench_generation_hybrid_reranker"
        elif args.hybrid:
            name = "ohr_bench_generation_hybrid"
        elif args.reranker:
            name = "ohr_bench_generation_reranker"
        else:
            name = "ohr_bench_generation_dense"
        args.output_dir = ROOT / "results" / name

    query_texts = [item["question"] for item in questions]
    dense_k = args.dense_k if args.hybrid else (args.candidate_k if args.reranker else args.top_k)
    dense_seconds = 0.0
    bm25_seconds = 0.0
    expanded_bm25_seconds = 0.0
    question_dense_seconds = 0.0
    rrf_seconds = 0.0
    reranker_seconds = 0.0
    if cache_meta is not None:
        cached_rows = read_jsonl(args.retrieval_cache / "retrieval_rows.jsonl")
        if len(cached_rows) != len(questions):
            raise ValueError("检索缓存问题数与本次 --limit 不一致")
        if any(cached["id"] != question["id"] for cached, question in zip(cached_rows, questions)):
            raise ValueError("检索缓存的问题顺序与 questions.jsonl 不一致")
        cached_parameters = cache_meta["parameters"]
        if int(cached_parameters["top_k"]) != args.top_k:
            raise ValueError(
                f"检索缓存 top_k={cached_parameters['top_k']}，但命令行 top_k={args.top_k}"
            )
        dense_k = int(cached_parameters["dense_k"])
        args.candidate_k = int(cached_parameters["candidate_k"])
        dense_rows = [
            [{"source_id": source_id} for source_id in row["dense_doc_ids"]]
            for row in cached_rows
        ]
        candidate_rows = [
            [{"source_id": source_id} for source_id in row["candidate_doc_ids"]]
            for row in cached_rows
        ]
        retrieved_rows = [row["retrieved"] for row in cached_rows]
        cached_timing = cache_meta["timing"]
        dense_seconds = float(cached_timing.get("dense_seconds", 0.0))
        bm25_seconds = float(cached_timing.get("bm25_seconds", 0.0))
        expanded_bm25_seconds = float(cached_timing.get("expanded_bm25_seconds", 0.0))
        question_dense_seconds = float(cached_timing.get("question_dense_seconds", 0.0))
        rrf_seconds = float(cached_timing.get("rrf_seconds", 0.0))
        reranker_seconds = float(cached_timing.get("reranker_seconds", 0.0))
        retrieval_pipeline_seconds = float(cached_timing["retrieval_pipeline_seconds"])
        mode = cache_meta["mode"]
    else:
        retriever = FaissRetriever(args.index_dir, device=args.embedding_device)
        dense_started = time.perf_counter()
        dense_rows = retriever.search_many(query_texts, top_k=dense_k)
        dense_seconds = time.perf_counter() - dense_started
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
            rrf_started = time.perf_counter()
            candidate_rows = reciprocal_rank_fusion_many(
                dense_rows,
                bm25_rows,
                rrf_k=args.rrf_k,
                top_k=args.candidate_k,
            )
            rrf_seconds = time.perf_counter() - rrf_started
        retrieved_rows = [rows[: args.top_k] for rows in candidate_rows]

    if args.reranker and cache_meta is None:
        print(f"正在加载 Reranker：{args.reranker_model} ({args.reranker_device})")
        reranker = BGEReranker(
            model_name=args.reranker_model,
            device=args.reranker_device,
            max_length=args.reranker_max_length,
        )
        reranker_started = time.perf_counter()
        retrieved_rows = reranker.rerank_many(
            query_texts,
            candidate_rows,
            top_k=args.top_k,
            batch_size=args.reranker_batch_size,
        )
        reranker_seconds = time.perf_counter() - reranker_started
        # 先完成并固定检索结果，再加载 AWQ，避免两个模型长期争抢显存。
        reranker.unload()
    if cache_meta is None:
        retrieval_pipeline_seconds = dense_seconds + bm25_seconds + rrf_seconds + reranker_seconds
        if args.hybrid and args.reranker:
            mode = "dense_bm25_rrf_then_reranker"
        elif args.hybrid:
            mode = "dense_bm25_rrf"
        elif args.reranker:
            mode = "dense_then_reranker"
        else:
            mode = "dense"

    client = create_backend(
        args.backend,
        api_base=args.api_base,
        model=args.model,
        awq_model=args.awq_model,
        llmqrt_root=args.llmqrt_root,
    )
    reranker_enabled = args.reranker or (
        cache_meta is not None and float(reranker_seconds) > 0.0
    )

    predictions = []
    totals = {
        "rag_em": 0.0,
        "rag_f1": 0.0,
        "candidate_hit": 0.0,
        "candidate_recall": 0.0,
        "hit": 0.0,
        "recall": 0.0,
        "dense_recall": 0.0,
        "dense_ndcg": 0.0,
        "ndcg": 0.0,
    }
    if not args.skip_no_rag:
        totals.update({"no_rag_em": 0.0, "no_rag_f1": 0.0})
    no_rag_generation_seconds = 0.0
    rag_generation_seconds = 0.0

    for number, (item, dense, candidates, retrieved) in enumerate(
        zip(questions, dense_rows, candidate_rows, retrieved_rows), start=1
    ):
        question = item["question"]
        answers = item["answers"]
        no_rag_answer = None
        no_rag_latency = 0.0
        if not args.skip_no_rag:
            no_rag_started = time.perf_counter()
            no_rag_answer = client.chat(
                build_eval_no_rag_messages(question),
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            no_rag_latency = time.perf_counter() - no_rag_started
        rag_started = time.perf_counter()
        rag_answer = client.chat(
            build_eval_rag_messages(question, retrieved), max_tokens=args.max_tokens, temperature=0.0
        )
        rag_latency = time.perf_counter() - rag_started
        no_rag_generation_seconds += no_rag_latency
        rag_generation_seconds += rag_latency

        rag_clean = clean_model_answer(rag_answer)
        retrieved_doc_ids = [row["source_id"] for row in retrieved]
        candidate_doc_ids = [row["source_id"] for row in candidates]
        # nDCG 的 K 严格对应实际送给 LLM 的前 K 个 chunk。
        dense_ranked_doc_ids = [row["source_id"] for row in dense[: args.top_k]]
        reranked_doc_ids = [row["source_id"] for row in retrieved[: args.top_k]]
        gold_doc_ids = set(item["gold_doc_ids"])
        candidate_hit = float(bool(set(candidate_doc_ids) & set(item["gold_doc_ids"])))
        hit = float(bool(set(retrieved_doc_ids) & set(item["gold_doc_ids"])))
        candidate_recall = len(set(candidate_doc_ids) & gold_doc_ids) / len(gold_doc_ids)
        recall = len(set(retrieved_doc_ids) & gold_doc_ids) / len(gold_doc_ids)
        dense_recall = len(set(dense_ranked_doc_ids) & gold_doc_ids) / len(gold_doc_ids)
        scores = {
            "rag_em": exact_match(rag_clean, answers),
            "rag_f1": character_f1(rag_clean, answers),
            "candidate_hit": candidate_hit,
            "candidate_recall": candidate_recall,
            "hit": hit,
            "recall": recall,
            "dense_recall": dense_recall,
            "dense_ndcg": ndcg_at_k(dense_ranked_doc_ids, gold_doc_ids, args.top_k),
            "ndcg": ndcg_at_k(reranked_doc_ids, gold_doc_ids, args.top_k),
        }
        if no_rag_answer is not None:
            no_rag_clean = clean_model_answer(no_rag_answer)
            scores.update(
                {
                    "no_rag_em": exact_match(no_rag_clean, answers),
                    "no_rag_f1": character_f1(no_rag_clean, answers),
                }
            )
        for name, value in scores.items():
            totals[name] += value

        predictions.append(
            {
                "id": item["id"],
                "question": question,
                "answers": answers,
                "gold_doc_ids": item["gold_doc_ids"],
                "candidate_doc_ids": candidate_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_chunks": [
                    {
                        key: row[key]
                        for key in (
                            "id",
                            "source_id",
                            "rank",
                            "score",
                            "dense_rank",
                            "dense_score",
                            "bm25_rank",
                            "bm25_score",
                            "question_bm25_rank",
                            "question_bm25_score",
                            "question_dense_rank",
                            "question_dense_score",
                            "summary_dense_rank",
                            "summary_dense_score",
                            "summary_question_rank",
                            "summary_question_score",
                            "summary_bm25_rank",
                            "summary_bm25_score",
                            "fusion_rank",
                            "fusion_score",
                            "rrf_score",
                            "rerank_score",
                        )
                        if key in row
                    }
                    for row in retrieved
                ],
                "no_rag_answer": no_rag_answer,
                "rag_answer": rag_answer,
                "latency": {
                    "no_rag_generation_seconds": no_rag_latency,
                    "rag_generation_seconds": rag_latency,
                },
                "scores": scores,
            }
        )
        print(f"[{number}/{len(questions)}] RAG F1={scores['rag_f1']:.3f}  {question}")

    averages = {name: value / len(questions) for name, value in totals.items()}
    client.unload()
    pipeline_seconds = retrieval_pipeline_seconds + rag_generation_seconds
    summary = {
        "question_count": len(questions),
        "top_k": args.top_k,
        "candidate_k": (
            args.candidate_k
            if (cache_meta is not None or args.hybrid or args.reranker)
            else dense_k
        ),
        "mode": mode,
        "backend": args.backend,
        "model": client.model,
        "timing": {
            "dense_seconds": dense_seconds,
            "dense_ms_per_query": dense_seconds * 1000 / len(questions),
            "bm25_seconds": bm25_seconds,
            "bm25_ms_per_query": bm25_seconds * 1000 / len(questions),
            "expanded_bm25_seconds": expanded_bm25_seconds,
            "expanded_bm25_ms_per_query": expanded_bm25_seconds * 1000 / len(questions),
            "question_dense_seconds": question_dense_seconds,
            "question_dense_ms_per_query": question_dense_seconds * 1000 / len(questions),
            "rrf_seconds": rrf_seconds,
            "rrf_ms_per_query": rrf_seconds * 1000 / len(questions),
            "reranker_seconds": reranker_seconds,
            "reranker_ms_per_query": reranker_seconds * 1000 / len(questions),
            "retrieval_pipeline_seconds": retrieval_pipeline_seconds,
            "retrieval_pipeline_ms_per_query": (
                retrieval_pipeline_seconds * 1000 / len(questions)
            ),
            "no_rag_generation_seconds": no_rag_generation_seconds,
            "no_rag_generation_ms_per_query": no_rag_generation_seconds * 1000 / len(questions),
            "rag_generation_seconds": rag_generation_seconds,
            "rag_generation_ms_per_query": rag_generation_seconds * 1000 / len(questions),
            "rag_pipeline_seconds": pipeline_seconds,
            "rag_pipeline_ms_per_query": pipeline_seconds * 1000 / len(questions),
            "model_loading_included": False,
            "no_rag_enabled": not args.skip_no_rag,
        },
        "retrieval_quality": {
            "k": args.top_k,
            "dense_ndcg_at_k": averages["dense_ndcg"],
            "ndcg_at_k": averages["ndcg"],
            "final_delta_vs_dense": averages["ndcg"] - averages["dense_ndcg"],
            "dense_recall_at_k": averages["dense_recall"],
            "recall_at_k": averages["recall"],
            "relevance_level": "page/source_id",
        },
        "averages": averages,
        "rag_minus_no_rag": (
            {
                "em": averages["rag_em"] - averages["no_rag_em"],
                "f1": averages["rag_f1"] - averages["no_rag_f1"],
            }
            if not args.skip_no_rag
            else None
        ),
        "note": "生成式回答不一定与短标准答案字面一致，建议人工复查 predictions.jsonl。",
    }
    if reranker_enabled:
        summary["reranker"] = {
            "model": (
                cache_meta["parameters"].get("reranker_model", args.reranker_model)
                if cache_meta is not None
                else args.reranker_model
            ),
            "device": (
                cache_meta["parameters"].get("reranker_device", args.reranker_device)
                if cache_meta is not None
                else args.reranker_device
            ),
            "batch_size": (
                cache_meta["parameters"].get("reranker_batch_size", args.reranker_batch_size)
                if cache_meta is not None
                else args.reranker_batch_size
            ),
            "max_length": (
                cache_meta["parameters"].get("reranker_max_length", args.reranker_max_length)
                if cache_meta is not None
                else args.reranker_max_length
            ),
        }
    if cache_meta is not None:
        summary["retrieval_cache"] = {
            "path": str(args.retrieval_cache.resolve()),
            "parameters": cache_meta["parameters"],
        }
        if "hierarchical_quality" in cache_meta:
            summary["hierarchical_quality"] = cache_meta["hierarchical_quality"]
    if args.hybrid:
        summary["hybrid"] = {
            "dense_k": args.dense_k,
            "bm25_k": args.bm25_k,
            "rrf_k": args.rrf_k,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)

    print("\n端到端结果：")
    if not args.skip_no_rag:
        print(f"  无 RAG: EM={averages['no_rag_em']:.4f}, F1={averages['no_rag_f1']:.4f}")
    print(f"  有 RAG: EM={averages['rag_em']:.4f}, F1={averages['rag_f1']:.4f}")
    if not args.skip_no_rag:
        print(
            f"  提升量: EM={summary['rag_minus_no_rag']['em']:+.4f}, "
            f"F1={summary['rag_minus_no_rag']['f1']:+.4f}"
        )
    candidate_name = mode
    candidate_k = (
        args.candidate_k if (cache_meta is not None or args.hybrid or args.reranker) else dense_k
    )
    print(f"  {candidate_name} Candidate Hit@{candidate_k}={averages['candidate_hit']:.4f}")
    print(f"  检索 Hit@{args.top_k}={averages['hit']:.4f}")
    print(f"  检索 Recall@{args.top_k}={averages['recall']:.4f}")
    print(f"  Dense nDCG@{args.top_k}={averages['dense_ndcg']:.4f}")
    print(f"  最终 nDCG@{args.top_k}={averages['ndcg']:.4f}")
    if reranker_enabled:
        print(
            f"  Reranker nDCG@{args.top_k} 差值="
            f"{averages['ndcg'] - averages['dense_ndcg']:+.4f}"
        )
    print(f"  Dense 检索：{dense_seconds * 1000 / len(questions):.1f} ms/query")
    if args.hybrid or bm25_seconds > 0:
        print(f"  BM25 检索：{bm25_seconds * 1000 / len(questions):.1f} ms/query")
    if expanded_bm25_seconds > 0:
        print(
            "  文本+问题 BM25："
            f"{expanded_bm25_seconds * 1000 / len(questions):.1f} ms/query"
        )
    if question_dense_seconds > 0:
        print(
            "  问题 FAISS："
            f"{question_dense_seconds * 1000 / len(questions):.1f} ms/query"
        )
    if args.hybrid or rrf_seconds > 0:
        print(f"  RRF 融合：{rrf_seconds * 1000 / len(questions):.1f} ms/query")
    if reranker_enabled:
        print(f"  Reranker：{reranker_seconds * 1000 / len(questions):.1f} ms/query")
    print(
        "  完整检索流水线："
        f"{retrieval_pipeline_seconds * 1000 / len(questions):.1f} ms/query"
    )
    print(f"  RAG 生成：{rag_generation_seconds * 1000 / len(questions):.1f} ms/query")
    print(f"  完整 RAG 流水线：{pipeline_seconds * 1000 / len(questions):.1f} ms/query")
    print(f"详细结果：{args.output_dir}")


if __name__ == "__main__":
    main()
