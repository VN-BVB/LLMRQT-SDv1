"""在固定小语料上比较 BGE-M3 多种检索组合，不启动 Reranker 或 LLM。

比较：
1. Dense + Sparse + ColBERT；
2. Dense + BM25；
3. Dense + ColBERT。

所有路线在同一批 Chunk 上评分，并通过 RRF 融合排名，避免直接相加不同量纲的分数。
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from bm25_retriever import chunk_search_text, tokenize_bm25
from rag_core import DEFAULT_EMBEDDING_MODEL, read_jsonl, write_json


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/ohr_bench/questions.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "storage/ohr_bench_bge_m3/chunks.jsonl")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--question-count", type=int, default=30)
    parser.add_argument("--corpus-size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--colbert-score-batch-size", type=int, default=64)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 8, 10])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/bge_m3_modes_small/summary.json"
    )
    return parser.parse_args()


def sample_fixed_corpus(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    question_count: int,
    corpus_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    eligible = [row for row in questions if row.get("gold_doc_ids")]
    if question_count > len(eligible):
        raise ValueError("question-count 超过可评测问题数")
    selected_questions = rng.sample(eligible, question_count)
    gold_sources = {
        str(source_id)
        for question in selected_questions
        for source_id in question["gold_doc_ids"]
    }
    gold_indices = [
        index for index, chunk in enumerate(chunks) if str(chunk["source_id"]) in gold_sources
    ]
    present_sources = {str(chunks[index]["source_id"]) for index in gold_indices}
    missing = gold_sources - present_sources
    if missing:
        raise ValueError(f"Chunk 库缺少 {len(missing)} 个标注证据页")
    if len(gold_indices) > corpus_size:
        raise ValueError(
            f"全部证据页已有 {len(gold_indices)} 个 Chunk，corpus-size 至少需要这么大"
        )
    selected_indices = set(gold_indices)
    distractors = [index for index in range(len(chunks)) if index not in selected_indices]
    selected_indices.update(rng.sample(distractors, corpus_size - len(selected_indices)))
    corpus_indices = list(selected_indices)
    rng.shuffle(corpus_indices)
    return selected_questions, [chunks[index] for index in corpus_indices]


def sparse_scores(
    query_weights: list[dict[str, float]],
    document_weights: list[dict[str, float]],
) -> np.ndarray:
    postings: dict[str, list[tuple[int, float]]] = {}
    for document_index, weights in enumerate(document_weights):
        for token, weight in weights.items():
            postings.setdefault(str(token), []).append((document_index, float(weight)))
    scores = np.zeros((len(query_weights), len(document_weights)), dtype=np.float32)
    for query_index, weights in enumerate(query_weights):
        for token, query_weight in weights.items():
            for document_index, document_weight in postings.get(str(token), []):
                scores[query_index, document_index] += float(query_weight) * document_weight
    return scores


def colbert_scores(
    query_vectors: list[np.ndarray],
    document_vectors: list[np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from tqdm.auto import tqdm

    all_scores = np.empty((len(query_vectors), len(document_vectors)), dtype=np.float32)
    with torch.inference_mode():
        for query_index, query_array in enumerate(
            tqdm(query_vectors, desc="ColBERT 全库晚交互", unit="query", dynamic_ncols=True)
        ):
            query = torch.as_tensor(query_array, device=device, dtype=torch.float16)
            for start in range(0, len(document_vectors), batch_size):
                batch = document_vectors[start : start + batch_size]
                max_tokens = max(len(item) for item in batch)
                padded = torch.zeros(
                    (len(batch), max_tokens, query.shape[1]),
                    device=device,
                    dtype=torch.float16,
                )
                mask = torch.zeros((len(batch), max_tokens), device=device, dtype=torch.bool)
                for offset, array in enumerate(batch):
                    length = len(array)
                    padded[offset, :length] = torch.as_tensor(
                        array, device=device, dtype=torch.float16
                    )
                    mask[offset, :length] = True
                token_scores = torch.einsum("qd,bpd->bqp", query, padded)
                token_scores.masked_fill_(~mask[:, None, :], -torch.inf)
                scores = token_scores.max(dim=2).values.mean(dim=1)
                all_scores[query_index, start : start + len(batch)] = (
                    scores.float().cpu().numpy()
                )
    return all_scores


def score_orders(scores: np.ndarray, *, positive_only: bool = False) -> list[np.ndarray]:
    orders = []
    for row in scores:
        order = np.argsort(-row, kind="stable")
        if positive_only:
            order = order[row[order] > 0]
        orders.append(order)
    return orders


def rrf_orders(routes: list[list[np.ndarray]], document_count: int, rrf_k: int) -> list[np.ndarray]:
    output = []
    for query_index in range(len(routes[0])):
        fused = np.zeros(document_count, dtype=np.float32)
        for route in routes:
            order = route[query_index]
            fused[order] += 1.0 / (rrf_k + np.arange(1, len(order) + 1, dtype=np.float32))
        output.append(np.argsort(-fused, kind="stable"))
    return output


def metrics_at_k(source_ids: list[str], gold: set[str], k: int) -> dict[str, float]:
    top = source_ids[:k]
    relevant = len(set(top) & gold)
    first = next((rank for rank, source in enumerate(top, 1) if source in gold), None)
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, source in enumerate(top, 1)
        if source in gold
    )
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return {
        "hit": float(relevant > 0),
        "precision": relevant / k,
        "recall": relevant / len(gold),
        "mrr": 0.0 if first is None else 1.0 / first,
        "ndcg": 0.0 if ideal == 0 else float(dcg / ideal),
    }


def evaluate(
    orders: list[np.ndarray],
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    ks: list[int],
) -> dict[str, dict[str, float]]:
    values = {k: {name: [] for name in ("hit", "precision", "recall", "mrr", "ndcg")} for k in ks}
    for order, question in zip(orders, questions):
        unique_sources = []
        seen = set()
        for index in order:
            source = str(chunks[int(index)]["source_id"])
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
            if len(unique_sources) >= max(ks):
                break
        gold = {str(item) for item in question["gold_doc_ids"]}
        for k in ks:
            current = metrics_at_k(unique_sources, gold, k)
            for name, value in current.items():
                values[k][name].append(value)
    return {
        str(k): {name: mean(metric_values) for name, metric_values in values[k].items()}
        for k in ks
    }


def main() -> None:
    args = parse_args()
    if min(args.k) <= 0 or args.rrf_k <= 0:
        raise ValueError("k 和 rrf-k 必须大于 0")
    questions, corpus = sample_fixed_corpus(
        read_jsonl(args.questions),
        read_jsonl(args.chunks),
        question_count=args.question_count,
        corpus_size=args.corpus_size,
        seed=args.seed,
    )
    query_texts = [row["question"] for row in questions]
    document_texts = [chunk_search_text(row) for row in corpus]

    try:
        import bm25s
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("需要安装 FlagEmbedding 和 bm25s") from exc

    bm25_started = time.perf_counter()
    tokenized_corpus = [tokenize_bm25(text) for text in document_texts]
    bm25 = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    bm25.index(tokenized_corpus, show_progress=False)
    bm25_offline_seconds = time.perf_counter() - bm25_started

    print(f"加载 {args.model}，语料 {len(corpus)} Chunk，问题 {len(questions)} 条")
    model = BGEM3FlagModel(
        args.model,
        use_fp16=args.device.startswith("cuda"),
        devices=args.device,
        batch_size=args.batch_size,
        query_max_length=args.max_length,
        passage_max_length=args.max_length,
    )
    document_started = time.perf_counter()
    document_output = model.encode(
        document_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    document_encode_seconds = time.perf_counter() - document_started

    dense_query_started = time.perf_counter()
    dense_query_output = model.encode(
        query_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_query_seconds = time.perf_counter() - dense_query_started

    multimode_query_started = time.perf_counter()
    query_output = model.encode(
        query_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    multimode_query_seconds = time.perf_counter() - multimode_query_started

    dense_started = time.perf_counter()
    dense_score_matrix = np.asarray(query_output["dense_vecs"], dtype=np.float32) @ np.asarray(
        document_output["dense_vecs"], dtype=np.float32
    ).T
    dense_seconds = time.perf_counter() - dense_started

    sparse_started = time.perf_counter()
    sparse_score_matrix = sparse_scores(
        query_output["lexical_weights"], document_output["lexical_weights"]
    )
    sparse_seconds = time.perf_counter() - sparse_started

    bm25_started = time.perf_counter()
    bm25_score_matrix = np.stack(
        [bm25.get_scores(tokenize_bm25(text)) for text in query_texts]
    ).astype(np.float32)
    bm25_seconds = time.perf_counter() - bm25_started

    colbert_started = time.perf_counter()
    colbert_score_matrix = colbert_scores(
        query_output["colbert_vecs"],
        document_output["colbert_vecs"],
        device=args.device,
        batch_size=args.colbert_score_batch_size,
    )
    colbert_seconds = time.perf_counter() - colbert_started

    dense_orders = score_orders(dense_score_matrix)
    sparse_orders = score_orders(sparse_score_matrix, positive_only=True)
    bm25_orders = score_orders(bm25_score_matrix, positive_only=True)
    colbert_orders = score_orders(colbert_score_matrix)

    fusion_started = time.perf_counter()
    experiment_orders = {
        "dense_sparse_colbert": rrf_orders(
            [dense_orders, sparse_orders, colbert_orders], len(corpus), args.rrf_k
        ),
        "dense_bm25": rrf_orders([dense_orders, bm25_orders], len(corpus), args.rrf_k),
        "dense_colbert": rrf_orders([dense_orders, colbert_orders], len(corpus), args.rrf_k),
    }
    fusion_seconds = time.perf_counter() - fusion_started
    ks = sorted(set(args.k))
    metrics = {
        name: evaluate(orders, questions, corpus, ks)
        for name, orders in experiment_orders.items()
    }

    query_count = len(questions)
    fusion_per_route = fusion_seconds / len(experiment_orders)
    timing = {
        "offline": {
            "bge_document_encode_seconds": document_encode_seconds,
            "bm25_build_seconds": bm25_offline_seconds,
        },
        "shared_components_ms_per_query": {
            "dense_only_query_encode": dense_query_seconds * 1000 / query_count,
            "multimode_query_encode": multimode_query_seconds * 1000 / query_count,
            "dense_score": dense_seconds * 1000 / query_count,
            "sparse_score": sparse_seconds * 1000 / query_count,
            "bm25_score": bm25_seconds * 1000 / query_count,
            "colbert_score": colbert_seconds * 1000 / query_count,
        },
        "estimated_online_ms_per_query": {
            "dense_sparse_colbert": (
                multimode_query_seconds + dense_seconds + sparse_seconds + colbert_seconds
                + fusion_per_route
            ) * 1000 / query_count,
            "dense_bm25": (
                dense_query_seconds + dense_seconds + bm25_seconds + fusion_per_route
            ) * 1000 / query_count,
            "dense_colbert": (
                multimode_query_seconds + dense_seconds + colbert_seconds + fusion_per_route
            ) * 1000 / query_count,
        },
    }
    result = {
        "experiment": "fixed_small_corpus_full_scan_rrf",
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "question_count": len(questions),
        "corpus_chunk_count": len(corpus),
        "gold_source_count": len(
            {source for row in questions for source in row["gold_doc_ids"]}
        ),
        "max_length": args.max_length,
        "rrf_k": args.rrf_k,
        "selected_question_ids": [row["id"] for row in questions],
        "metrics": metrics,
        "timing": timing,
        "notes": [
            "三组均在同一固定小语料上全库评分，并使用等权 RRF 融合排名。",
            "文档编码和 BM25 建库属于离线成本，不计入在线延迟。",
            "延迟为批量处理总耗时除以问题数，代表吞吐等效值，不是并发服务 P95。",
        ],
    }
    write_json(args.output, result)
    for name in ("dense_sparse_colbert", "dense_bm25", "dense_colbert"):
        row = metrics[name][str(max(ks))]
        print(
            f"{name}: Hit@{max(ks)}={row['hit']:.4f}, "
            f"Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, "
            f"MRR={row['mrr']:.4f}, nDCG={row['ndcg']:.4f}, "
            f"latency={timing['estimated_online_ms_per_query'][name]:.1f} ms/query"
        )
    print(f"结果：{args.output}")

    del model, document_output, query_output, dense_query_output
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
