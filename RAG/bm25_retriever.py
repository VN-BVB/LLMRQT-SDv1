"""基于 bm25s 的持久化 BM25 Chunk 检索器。"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from rag_core import read_jsonl


TOKENIZER_VERSION = "jieba_cjk_and_latin_words_v1"
_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize_bm25(text: str) -> list[str]:
    """英文和数字按词切分，连续中文使用 jieba 精确模式。"""

    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("缺少 jieba，请先安装 requirements.txt") from exc

    tokens: list[str] = []
    for segment in _SEGMENT_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", segment):
            tokens.extend(token.strip() for token in jieba.lcut(segment) if token.strip())
        else:
            tokens.append(segment)
    return tokens


def chunk_search_text(
    chunk: dict[str, Any],
    *,
    include_generated_questions: bool = False,
) -> str:
    parts = [chunk.get("title", "")]
    if chunk.get("domain"):
        parts.append(str(chunk["domain"]))
    if chunk.get("section_path"):
        parts.append(" > ".join(chunk["section_path"]))
    parts.append(chunk["text"])
    if include_generated_questions:
        parts.extend(chunk.get("generated_questions") or [])
    return "\n".join(str(part) for part in parts if part)


def build_bm25_index(
    chunks_path: str | Path,
    index_path: str | Path,
    *,
    overwrite: bool = False,
    include_generated_questions: bool = False,
) -> dict[str, Any]:
    """将 chunks.jsonl 建成可 mmap 加载的 bm25s 索引。"""

    try:
        import bm25s
    except ImportError as exc:
        raise RuntimeError("缺少 bm25s，请先安装 requirements.txt") from exc

    chunks_path = Path(chunks_path)
    index_path = Path(index_path)
    if index_path.exists() and not overwrite:
        raise FileExistsError(f"BM25 索引已存在：{index_path}；如需重建请使用 --overwrite")
    if index_path.exists():
        shutil.rmtree(index_path)

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"Chunk 为空：{chunks_path}")

    from tqdm.auto import tqdm

    corpus_tokens = [
        tokenize_bm25(
            chunk_search_text(
                chunk,
                include_generated_questions=include_generated_questions,
            )
        )
        for chunk in tqdm(chunks, desc="BM25 分词", unit="chunk", dynamic_ncols=True)
    ]
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(index_path, show_progress=True)

    metadata = {
        "chunks_file": str(chunks_path.resolve()),
        "chunk_count": len(chunks),
        "index_dir": str(index_path.resolve()),
        "engine": "bm25s",
        "bm25s_version": bm25s.__version__,
        "method": "lucene",
        "k1": 1.5,
        "b": 0.75,
        "tokenizer": TOKENIZER_VERSION,
        "include_generated_questions": include_generated_questions,
    }
    with (index_path / "bm25_meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return metadata


class BM25Retriever:
    """读取 bm25s 索引，返回与 FAISS 相同结构的 Chunk。"""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        bm25_index: str | Path | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            import bm25s
        except ImportError as exc:
            raise RuntimeError("缺少 bm25s，请先安装 requirements.txt") from exc

        index_dir = Path(index_dir)
        self.index_path = Path(bm25_index or index_dir / "bm25")
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"找不到 BM25 索引：{self.index_path}。请先运行 build_bm25.py"
            )
        self.chunks = chunks if chunks is not None else read_jsonl(index_dir / "chunks.jsonl")
        meta_path = self.index_path / "bm25_meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            if metadata["chunk_count"] != len(self.chunks):
                raise ValueError(
                    f"BM25 索引有 {metadata['chunk_count']} 条记录，"
                    f"但 chunks.jsonl 有 {len(self.chunks)} 行"
                )
        self.retriever = bm25s.BM25.load(
            self.index_path,
            load_corpus=False,
            mmap=True,
            show_progress=False,
        )

    def search(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.search_many([question], top_k=top_k)[0]

    def search_many(self, questions: list[str], top_k: int = 10) -> list[list[dict[str, Any]]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not questions:
            return []
        query_tokens = [tokenize_bm25(question) for question in questions]
        documents, scores = self.retriever.retrieve(
            query_tokens,
            k=min(top_k, len(self.chunks)),
            show_progress=len(questions) > 1,
        )
        all_results = []
        for row_indices, row_scores in zip(documents, scores):
            results = []
            for index, score in zip(row_indices, row_scores):
                # bm25s 在匹配数量不足时可能补零分文档，不应当纳入候选集。
                if int(index) < 0 or float(score) <= 0:
                    continue
                item = dict(self.chunks[int(index)])
                rank = len(results) + 1
                item["rank"] = rank
                item["score"] = float(score)
                item["bm25_rank"] = rank
                item["bm25_score"] = float(score)
                results.append(item)
            all_results.append(results)
        return all_results

    def search_many_filtered(
        self,
        questions: list[str],
        allowed_source_ids: list[set[str]],
        *,
        top_k: int = 50,
    ) -> list[list[dict[str, Any]]]:
        """计算全局 BM25 分数，但只在选中的 Parent 页面内排序。"""

        if len(questions) != len(allowed_source_ids):
            raise ValueError("questions 与 allowed_source_ids 数量不一致")
        if not questions:
            return []
        import numpy as np

        if not hasattr(self, "_source_to_indices"):
            mapping: dict[str, list[int]] = {}
            for index, chunk in enumerate(self.chunks):
                mapping.setdefault(str(chunk["source_id"]), []).append(index)
            self._source_to_indices = mapping
        all_results = []
        for question, allowed in zip(questions, allowed_source_ids):
            candidate_indices = np.asarray(
                [
                    index
                    for source_id in allowed
                    for index in self._source_to_indices.get(str(source_id), [])
                ],
                dtype=np.int64,
            )
            if len(candidate_indices) == 0:
                all_results.append([])
                continue
            scores = self.retriever.get_scores(tokenize_bm25(question))
            local_scores = scores[candidate_indices]
            order = np.argsort(-local_scores)[: min(top_k, len(candidate_indices))]
            results = []
            for offset in order:
                score = float(local_scores[int(offset)])
                if score <= 0:
                    continue
                index = int(candidate_indices[int(offset)])
                item = dict(self.chunks[index])
                rank = len(results) + 1
                item["rank"] = rank
                item["score"] = score
                item["bm25_rank"] = rank
                item["bm25_score"] = score
                results.append(item)
            all_results.append(results)
        return all_results
