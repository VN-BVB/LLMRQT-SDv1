"""Generated-question BGE/FAISS auxiliary index build and retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rag_core import BGEEmbedder, read_jsonl, write_json


def build_question_faiss_index(
    index_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    embedding_model: str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Flatten ``generated_questions`` and build one vector per question."""

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("缺少 faiss，请先安装 requirements.txt") from exc

    index_dir = Path(index_dir)
    output_dir = Path(output_dir or index_dir / "questions")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"问题索引已存在：{output_dir}；如需重建请使用 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (index_dir / "index_meta.json").open("r", encoding="utf-8") as file:
        chunk_meta = json.load(file)
    chunks = read_jsonl(index_dir / "chunks.jsonl")
    question_texts: list[str] = []
    chunk_indices: list[int] = []
    for chunk_index, chunk in enumerate(chunks):
        for question in chunk.get("generated_questions") or []:
            if isinstance(question, str) and question.strip():
                question_texts.append(question.strip())
                chunk_indices.append(chunk_index)
    if not question_texts:
        raise ValueError("chunks.jsonl 中没有 generated_questions，无法建立问题索引")

    model_name = embedding_model or chunk_meta["embedding_model"]
    embedding_max_length = int(max_length or chunk_meta.get("embedding_max_length", 1024))
    print(f"正在加载问题 Embedding 模型：{model_name} ({device})")
    embedder = BGEEmbedder(
        model_name=model_name,
        device=device,
        query_prefix=chunk_meta.get("query_prefix", ""),
        max_length=embedding_max_length,
    )
    vectors = embedder.encode(question_texts, is_query=False, batch_size=batch_size)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    np.save(output_dir / "question_chunk_indices.npy", np.asarray(chunk_indices, dtype=np.int64))

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks_file": str((index_dir / "chunks.jsonl").resolve()),
        "chunk_count": len(chunks),
        "question_count": len(question_texts),
        "mapping_order": "chunk_line_then_generated_question_order",
        "embedding_model": model_name,
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_max_length": embedding_max_length,
        "query_prefix": chunk_meta.get("query_prefix", ""),
        "normalized": True,
        "index_type": "IndexFlatIP",
    }
    write_json(output_dir / "index_meta.json", metadata)
    return metadata


class QuestionFaissRetriever:
    """Search generated questions, collapse hits to unique original chunks."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        question_index: str | Path | None = None,
        device: str = "cpu",
        chunks: list[dict[str, Any]] | None = None,
        embedder: BGEEmbedder | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("缺少 faiss，请先安装 requirements.txt") from exc

        index_dir = Path(index_dir)
        question_index = Path(question_index or index_dir / "questions")
        with (question_index / "index_meta.json").open("r", encoding="utf-8") as file:
            self.meta = json.load(file)
        self.chunks = chunks if chunks is not None else read_jsonl(index_dir / "chunks.jsonl")
        self.chunk_indices = np.load(question_index / "question_chunk_indices.npy", mmap_mode="r")
        self.index = faiss.read_index(str(question_index / "index.faiss"))
        if self.index.ntotal != len(self.chunk_indices):
            raise ValueError("问题 FAISS 向量数与 question_chunk_indices.npy 长度不一致")
        if int(self.meta["chunk_count"]) != len(self.chunks):
            raise ValueError("问题索引记录的 Chunk 数与当前 chunks.jsonl 不一致")
        self.embedder = embedder or BGEEmbedder(
            model_name=self.meta["embedding_model"],
            device=device,
            query_prefix=self.meta.get("query_prefix", ""),
            max_length=int(self.meta.get("embedding_max_length", 1024)),
        )

    def search_many(
        self,
        questions: list[str],
        *,
        top_k: int = 50,
        fetch_k: int | None = None,
        query_vectors: Any | None = None,
    ) -> list[list[dict[str, Any]]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not questions:
            return []
        fetch_k = fetch_k or top_k * 6
        fetch_k = min(max(fetch_k, top_k), self.index.ntotal)
        vectors = (
            self.embedder.encode(questions, is_query=True)
            if query_vectors is None
            else query_vectors
        )
        if len(vectors) != len(questions):
            raise ValueError("query_vectors 数量与 questions 不一致")
        scores, indices = self.index.search(vectors, fetch_k)

        all_results: list[list[dict[str, Any]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results = []
            seen_chunks = set()
            for score, question_index in zip(row_scores, row_indices):
                if int(question_index) < 0:
                    continue
                chunk_index = int(self.chunk_indices[int(question_index)])
                if chunk_index in seen_chunks:
                    continue
                seen_chunks.add(chunk_index)
                item = dict(self.chunks[chunk_index])
                rank = len(results) + 1
                item["rank"] = rank
                item["score"] = float(score)
                item["question_dense_rank"] = rank
                item["question_dense_score"] = float(score)
                results.append(item)
                if len(results) >= top_k:
                    break
            all_results.append(results)
        return all_results

    def search(
        self,
        question: str,
        *,
        top_k: int = 50,
        fetch_k: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.search_many([question], top_k=top_k, fetch_k=fetch_k)[0]

    def search_many_filtered(
        self,
        questions: list[str],
        allowed_source_ids: list[set[str]],
        *,
        top_k: int = 50,
        query_vectors: Any | None = None,
    ) -> list[list[dict[str, Any]]]:
        """只在选中 Parent 页所属的生成问题向量中搜索。"""

        if len(questions) != len(allowed_source_ids):
            raise ValueError("questions 与 allowed_source_ids 数量不一致")
        if not questions:
            return []
        import faiss

        if not hasattr(self, "_source_to_question_indices"):
            mapping: dict[str, list[int]] = {}
            for question_index, chunk_index in enumerate(self.chunk_indices):
                source_id = str(self.chunks[int(chunk_index)]["source_id"])
                mapping.setdefault(source_id, []).append(question_index)
            self._source_to_question_indices = mapping
        vectors = (
            self.embedder.encode(questions, is_query=True)
            if query_vectors is None
            else query_vectors
        )
        if len(vectors) != len(questions):
            raise ValueError("query_vectors 数量与 questions 不一致")
        all_results: list[list[dict[str, Any]]] = []
        for vector, allowed in zip(vectors, allowed_source_ids):
            ids = np.asarray(
                [
                    index
                    for source_id in allowed
                    for index in self._source_to_question_indices.get(str(source_id), [])
                ],
                dtype=np.int64,
            )
            if len(ids) == 0:
                all_results.append([])
                continue
            selector = faiss.IDSelectorBatch(ids)
            params = faiss.SearchParameters()
            params.sel = selector
            # 同一 Child 可能有多个问题，多取一些再折叠为唯一 Child。
            fetch_k = min(len(ids), max(top_k * 3, top_k))
            scores, indices = self.index.search(
                np.ascontiguousarray(vector[None, :], dtype="float32"),
                fetch_k,
                params=params,
            )
            results = []
            seen_chunks = set()
            for score, question_index in zip(scores[0], indices[0]):
                if int(question_index) < 0:
                    continue
                chunk_index = int(self.chunk_indices[int(question_index)])
                if chunk_index in seen_chunks:
                    continue
                seen_chunks.add(chunk_index)
                item = dict(self.chunks[chunk_index])
                rank = len(results) + 1
                item["rank"] = rank
                item["score"] = float(score)
                item["question_dense_rank"] = rank
                item["question_dense_score"] = float(score)
                results.append(item)
                if len(results) >= top_k:
                    break
            all_results.append(results)
        return all_results
