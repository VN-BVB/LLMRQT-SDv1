"""Dense + BM25 双路召回与 Reciprocal Rank Fusion。"""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion_routes(
    routes: list[tuple[str, list[dict[str, Any]]]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """融合任意数量的 Chunk 排名列表，并按 Chunk ID 去重。"""

    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k 必须大于 0 或为 None")
    if not routes:
        return []

    fused: dict[str, dict[str, Any]] = {}
    for source, results in routes:
        seen_in_route = set()
        for fallback_rank, result in enumerate(results, start=1):
            chunk_id = result["id"]
            if chunk_id in seen_in_route:
                continue
            seen_in_route.add(chunk_id)
            rank = int(result.get("rank", fallback_rank))
            if chunk_id not in fused:
                fused[chunk_id] = dict(result)
                fused[chunk_id]["rrf_score"] = 0.0
            item = fused[chunk_id]
            item["rrf_score"] += 1.0 / (rrf_k + rank)
            item[f"{source}_rank"] = rank
            item[f"{source}_score"] = float(result["score"])

    rows = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
        item["score"] = float(item["rrf_score"])
        item["fusion_rank"] = rank
    return rows


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """按 Chunk ID 去重并融合排名；不直接混合两种不可比分数。"""

    return reciprocal_rank_fusion_routes(
        [("dense", dense_results), ("bm25", bm25_results)],
        rrf_k=rrf_k,
        top_k=top_k,
    )


def reciprocal_rank_fusion_route_batches(
    routes: list[tuple[str, list[list[dict[str, Any]]]]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[list[dict[str, Any]]]:
    """批量版本；所有路线必须包含相同数量的查询。"""

    if not routes:
        return []
    query_count = len(routes[0][1])
    if any(len(rows) != query_count for _, rows in routes):
        raise ValueError("参与 RRF 的路线查询数量不一致")
    return [
        reciprocal_rank_fusion_routes(
            [(name, rows[query_index]) for name, rows in routes],
            rrf_k=rrf_k,
            top_k=top_k,
        )
        for query_index in range(query_count)
    ]


def reciprocal_rank_fusion_many(
    dense_rows: list[list[dict[str, Any]]],
    bm25_rows: list[list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[list[dict[str, Any]]]:
    return reciprocal_rank_fusion_route_batches(
        [("dense", dense_rows), ("bm25", bm25_rows)],
        rrf_k=rrf_k,
        top_k=top_k,
    )
