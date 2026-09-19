"""最小但完整的 RAG 公共组件。

这个文件只做三件事：
1. 把长文档切成有重叠的小块（chunk）；
2. 用专门的 Embedding 模型把文字变成向量；
3. 从 FAISS 中找出和问题最相似的 chunk。

这里故意不使用 LangChain，方便初学者看清 RAG 的每一个步骤。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


# BGE-M3 的 dense 检索不要求给查询添加 instruction 前缀。
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_QUERY_PREFIX = ""


def detect_heading(line: str) -> tuple[int, str] | None:
    """识别常见 Markdown、Chapter/Section 和数字编号标题。"""

    line = line.strip()
    if not line or len(line) > 180:
        return None

    markdown = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    named = re.match(r"^(chapter|section)\s+([A-Z0-9IVXLC]+(?:\.\d+)*)\b[.:\-]?\s*(.*)$", line, re.I)
    if named:
        title = " ".join(part for part in (named.group(1), named.group(2), named.group(3)) if part)
        return 1, title.strip()

    numbered = re.match(r"^(\d+(?:\.\d+){0,3})[.)]?\s+(.{2,160})$", line)
    if numbered and not numbered.group(2).rstrip().endswith((".", "。", ";", "；")):
        return numbered.group(1).count(".") + 1, line
    return None


def apply_heading(section_path: list[str], level: int, title: str) -> list[str]:
    """按照标题级别更新当前章节路径。"""

    parent_count = max(level - 1, 0)
    return [*section_path[:parent_count], title]


def extract_headings(text: str) -> list[dict[str, Any]]:
    """提取页面中的章节标题，供跨页继承章节上下文。"""

    headings = []
    for line in text.splitlines():
        detected = detect_heading(line)
        if detected:
            level, title = detected
            headings.append({"level": level, "title": title})
    return headings


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON Lines：一行是一个 JSON 对象。"""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """保存 JSON Lines。ensure_ascii=False 让中文直接可读。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _split_oversized_paragraph(
    text: str, absolute_start: int, chunk_size: int, overlap: int
) -> list[dict[str, Any]]:
    """只对超长段落使用有重叠的字符窗口，尽量在句末结束。"""

    chunks: list[dict[str, Any]] = []
    start = 0
    natural_boundaries = "。！？；\n"
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            search_from = start + int(chunk_size * 0.6)
            candidates = [text.rfind(mark, search_from, hard_end) for mark in natural_boundaries]
            best = max(candidates)
            if best >= search_from:
                end = best + 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start": absolute_start + start,
                    "end": absolute_start + end,
                }
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    *,
    initial_section_path: list[str] | None = None,
) -> list[dict[str, Any]]:
    """优先按空行划分自然段，只对超过 chunk_size 的段落再做字符窗口切分。

    这是第一版最容易解释的段落切块策略。`start/end` 是原文字符偏移；
    `paragraph_index` 可用于相邻段落扩展和结果追溯。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    section_path = list(initial_section_path or [])
    for paragraph_index, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S)):
        paragraph = match.group(0).strip()
        paragraph_start = match.start()
        lines = paragraph.splitlines()
        heading = detect_heading(lines[0]) if lines else None
        if heading:
            section_path = apply_heading(section_path, *heading)
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue
            body_offset = paragraph.find(body)
            paragraph = body
            paragraph_start += body_offset

        if len(paragraph) <= chunk_size:
            pieces = [
                {
                    "text": paragraph,
                    "start": paragraph_start,
                    "end": paragraph_start + len(paragraph),
                }
            ]
        else:
            pieces = _split_oversized_paragraph(paragraph, paragraph_start, chunk_size, overlap)
        for piece in pieces:
            piece["paragraph_index"] = paragraph_index
            piece["section_path"] = list(section_path)
            chunks.append(piece)

    return chunks


class BGEEmbedder:
    """用 Transformers 直接运行 BGE Embedding 模型。

    为什么不用你的 Qwen3-1.7B 直接生成向量？
    生成模型与检索向量模型的训练目标不同。标准 RAG 通常使用独立的小型
    Embedding 模型做检索，再让生成模型读取检索结果并回答。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("缺少 torch/transformers，请先安装 requirements.txt") from exc

        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool,
        batch_size: int = 32,
    ) -> "Any":
        """把文本编码为 L2 归一化后的 float32 向量。

        向量归一化后，FAISS 的内积（Inner Product）就等价于余弦相似度。
        BGE 使用第一个 token（CLS）的 hidden state 作为整句向量。
        """

        import numpy as np
        from tqdm.auto import tqdm

        if not texts:
            raise ValueError("不能编码空的文本列表")

        prepared = [self.query_prefix + text if is_query else text for text in texts]
        all_embeddings = []

        batch_starts = range(0, len(prepared), batch_size)
        progress = tqdm(
            batch_starts,
            total=len(batch_starts),
            desc="编码查询" if is_query else "编码文档 Chunk",
            unit="batch",
            dynamic_ncols=True,
            disable=len(prepared) <= batch_size,
        )
        for start in progress:
            batch = prepared[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}

            with self.torch.inference_mode():
                output = self.model(**tokens)
                embeddings = output.last_hidden_state[:, 0]
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.ascontiguousarray(np.concatenate(all_embeddings, axis=0), dtype="float32")


class FaissRetriever:
    """加载离线生成的 FAISS 索引，并执行在线向量检索。"""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        device: str = "cpu",
        embedding_model: str | None = None,
        embedder: BGEEmbedder | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("缺少 faiss，请先安装 requirements.txt") from exc

        index_dir = Path(index_dir)
        meta_path = index_dir / "index_meta.json"
        with meta_path.open("r", encoding="utf-8") as file:
            self.meta = json.load(file)

        self.chunks = read_jsonl(index_dir / "chunks.jsonl")
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"索引有 {self.index.ntotal} 条向量，但 chunks.jsonl 有 {len(self.chunks)} 行"
            )

        if embedder is not None:
            expected_model = embedding_model or self.meta["embedding_model"]
            if embedder.model_name != expected_model:
                raise ValueError(
                    f"共享 Embedder={embedder.model_name}，索引要求={expected_model}"
                )
            self.embedder = embedder
        else:
            model_name = embedding_model or self.meta["embedding_model"]
            self.embedder = BGEEmbedder(
                model_name=model_name,
                device=device,
                query_prefix=self.meta.get("query_prefix", DEFAULT_QUERY_PREFIX),
                max_length=int(self.meta.get("embedding_max_length", 512)),
            )

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """检索一个问题，返回带 rank 和 score 的 chunk。"""

        return self.search_many([question], top_k=top_k)[0]

    def search_many(
        self,
        questions: list[str],
        top_k: int = 3,
        *,
        query_vectors: Any | None = None,
    ) -> list[list[dict[str, Any]]]:
        """批量检索。评测时批量编码问题会快很多。"""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not questions:
            return []

        top_k = min(top_k, len(self.chunks))
        vectors = (
            self.embedder.encode(questions, is_query=True)
            if query_vectors is None
            else query_vectors
        )
        if len(vectors) != len(questions):
            raise ValueError("query_vectors 数量与 questions 不一致")
        scores, indices = self.index.search(vectors, top_k)

        all_results: list[list[dict[str, Any]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results = []
            for rank, (score, index) in enumerate(zip(row_scores, row_indices), start=1):
                if index < 0:
                    continue
                item = dict(self.chunks[int(index)])
                item["rank"] = rank
                item["score"] = float(score)
                results.append(item)
            all_results.append(results)
        return all_results

    def search_many_filtered(
        self,
        questions: list[str],
        allowed_source_ids: list[set[str]],
        *,
        top_k: int = 50,
        query_vectors: Any | None = None,
    ) -> list[list[dict[str, Any]]]:
        """在每个查询允许的 Parent 页面内执行精确 FAISS 检索。"""

        if len(questions) != len(allowed_source_ids):
            raise ValueError("questions 与 allowed_source_ids 数量不一致")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not questions:
            return []
        import faiss
        import numpy as np

        if not hasattr(self, "_source_to_indices"):
            mapping: dict[str, list[int]] = {}
            for index, chunk in enumerate(self.chunks):
                mapping.setdefault(str(chunk["source_id"]), []).append(index)
            self._source_to_indices = mapping
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
                    for index in self._source_to_indices.get(str(source_id), [])
                ],
                dtype=np.int64,
            )
            if len(ids) == 0:
                all_results.append([])
                continue
            selector = faiss.IDSelectorBatch(ids)
            params = faiss.SearchParameters()
            params.sel = selector
            scores, indices = self.index.search(
                np.ascontiguousarray(vector[None, :], dtype="float32"),
                min(top_k, len(ids)),
                params=params,
            )
            results = []
            for score, index in zip(scores[0], indices[0]):
                if int(index) < 0:
                    continue
                item = dict(self.chunks[int(index)])
                rank = len(results) + 1
                item["rank"] = rank
                item["score"] = float(score)
                results.append(item)
            all_results.append(results)
        return all_results
