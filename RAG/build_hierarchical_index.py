"""构建 OHR 页级 Parent Summary 与精细 Child 的两层离线索引。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bm25_retriever import build_bm25_index, chunk_search_text
from chunk_questions import PROMPT_VERSION as CHILD_PROMPT_VERSION
from chunk_questions import generate_questions_for_chunks
from generation_backends import DEFAULT_AWQ_MODEL, DEFAULT_LLMQRT_ROOT, create_backend
from hierarchical_chunks import build_token_window_children
from parent_generation import generate_parent_records
from rag_core import BGEEmbedder, read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parent


def _copy_or_link(source: Path, target: Path, *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _save_flat_index(path: Path, vectors: np.ndarray) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(path))


def _build_question_vectors(
    records: list[dict[str, Any]],
    embedder: BGEEmbedder,
    output_dir: Path,
    *,
    batch_size: int,
    layer: str,
) -> dict[str, Any]:
    texts: list[str] = []
    record_indices: list[int] = []
    for record_index, record in enumerate(records):
        for question in record.get("generated_questions") or []:
            if isinstance(question, str) and question.strip():
                texts.append(question.strip())
                record_indices.append(record_index)
    if not texts:
        raise ValueError(f"{layer} 没有 generated_questions")
    vectors = embedder.encode(texts, is_query=False, batch_size=batch_size)
    _save_flat_index(output_dir / "index.faiss", vectors)
    np.save(output_dir / "question_chunk_indices.npy", np.asarray(record_indices, dtype=np.int64))
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        "chunk_count": len(records),
        "question_count": len(texts),
        "mapping_order": "record_line_then_generated_question_order",
        "embedding_model": embedder.model_name,
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_max_length": embedder.max_length,
        "query_prefix": embedder.query_prefix,
        "normalized": True,
        "index_type": "IndexFlatIP",
    }
    write_json(output_dir / "index_meta.json", meta)
    return meta


def generate_stage(args: argparse.Namespace) -> None:
    documents = read_jsonl(args.documents)
    print(
        f"按 {args.child_tokens} token、overlap={args.child_overlap_tokens} 构造精细 Child"
    )
    children = build_token_window_children(
        documents,
        tokenizer_name=args.embedding_model or "BAAI/bge-m3",
        chunk_tokens=args.child_tokens,
        overlap_tokens=args.child_overlap_tokens,
        max_page_chars=args.child_max_page_chars,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    backend = create_backend(
        args.backend,
        api_base=args.api_base,
        model=args.model,
        awq_model=args.awq_model,
        llmqrt_root=args.llmqrt_root,
        local_batch_size=args.child_batch_size,
    )
    try:
        parents, parent_meta = generate_parent_records(
            documents,
            backend,
            checkpoint_path=output / "parents.partial.jsonl",
            question_count=args.parent_question_count,
            batch_size=args.parent_batch_size,
            max_concurrency=args.parent_workers,
            max_tokens=args.parent_max_tokens,
            max_input_chars=args.parent_max_input_chars,
            retries=args.retries,
            resume=not args.no_resume,
        )
        child_meta = generate_questions_for_chunks(
            children,
            backend,
            checkpoint_path=output / "children.partial.jsonl",
            question_count=args.child_question_count,
            batch_size=args.child_batch_size,
            max_concurrency=args.child_workers,
            max_tokens=args.child_max_tokens,
            retries=args.retries,
            resume=not args.no_resume,
        )
    finally:
        backend.unload()

    parent_dir = output / "parent"
    child_dir = output / "child"
    write_jsonl(parent_dir / "chunks.jsonl", parents)
    write_jsonl(child_dir / "chunks.jsonl", children)
    write_json(
        child_dir / "generation_meta.json",
        {
            "chunk_strategy": "per_page_bge_token_window",
            "chunk_tokens": args.child_tokens,
            "overlap_tokens": args.child_overlap_tokens,
            "max_page_chars": args.child_max_page_chars,
            "tokenizer": args.embedding_model or "BAAI/bge-m3",
            "generated_questions": {
                **child_meta,
                "enabled": True,
                "prompt_version": CHILD_PROMPT_VERSION,
                "question_count_per_child": args.child_question_count,
            },
        },
    )
    write_json(
        output / "hierarchy_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": str(args.documents.resolve()),
            "child_source": "rebuilt_from_documents_with_token_windows",
            "parent_generation": parent_meta,
            "child_generation": child_meta,
            "parent_count": len(parents),
            "child_count": len(children),
            "mapping": "child.source_id == parent.id",
            "backend": args.backend,
            "generation_model": backend.model,
        },
    )
    for partial in (output / "parents.partial.jsonl", output / "children.partial.jsonl"):
        if partial.exists():
            partial.unlink()
    print(f"生成阶段完成：{output}")


def index_stage(args: argparse.Namespace) -> None:
    import faiss

    parent_dir = args.output_dir / "parent"
    child_dir = args.output_dir / "child"
    parents = read_jsonl(parent_dir / "chunks.jsonl")
    children = read_jsonl(child_dir / "chunks.jsonl")
    with (child_dir / "generation_meta.json").open("r", encoding="utf-8") as file:
        generation_meta = json.load(file)
    model_name = args.embedding_model or generation_meta["tokenizer"]
    max_length = args.embedding_max_length or 1024
    embedder = BGEEmbedder(
        model_name=model_name,
        device=args.embedding_device,
        query_prefix="",
        max_length=max_length,
    )

    parent_vectors = embedder.encode(
        [chunk_search_text(row) for row in parents],
        is_query=False,
        batch_size=args.embedding_batch_size,
    )
    _save_flat_index(parent_dir / "index.faiss", parent_vectors)
    write_json(
        parent_dir / "index_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(parents),
            "chunk_strategy": "one_llm_summary_parent_per_pdf_page",
            "embedding_text": "title + domain + section_path + summary",
            "embedding_model": model_name,
            "embedding_dimension": int(parent_vectors.shape[1]),
            "embedding_max_length": max_length,
            "query_prefix": "",
            "normalized": True,
            "index_type": "IndexFlatIP",
        },
    )
    _build_question_vectors(
        parents,
        embedder,
        parent_dir / "questions",
        batch_size=args.embedding_batch_size,
        layer="parent",
    )
    child_vectors = embedder.encode(
        [chunk_search_text(row) for row in children],
        is_query=False,
        batch_size=args.embedding_batch_size,
    )
    _save_flat_index(child_dir / "index.faiss", child_vectors)
    write_json(
        child_dir / "index_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(children),
            "chunk_strategy": generation_meta["chunk_strategy"],
            "embedding_text": "title + domain + section_path + child_text",
            "chunk_tokens": generation_meta["chunk_tokens"],
            "overlap_tokens": generation_meta["overlap_tokens"],
            "max_page_chars": generation_meta["max_page_chars"],
            "embedding_model": model_name,
            "embedding_dimension": int(child_vectors.shape[1]),
            "embedding_max_length": max_length,
            "query_prefix": "",
            "normalized": True,
            "index_type": "IndexFlatIP",
            "generated_questions": generation_meta["generated_questions"],
        },
    )
    _build_question_vectors(
        children,
        embedder,
        child_dir / "questions",
        batch_size=args.embedding_batch_size,
        layer="child",
    )
    del embedder, parent_vectors, child_vectors
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    build_bm25_index(
        parent_dir / "chunks.jsonl",
        parent_dir / "bm25_questions",
        overwrite=args.overwrite,
        include_generated_questions=True,
    )
    build_bm25_index(
        child_dir / "chunks.jsonl",
        child_dir / "bm25_questions",
        overwrite=args.overwrite,
        include_generated_questions=True,
    )
    # 核查 Child 文本向量数，防止 Chunk 顺序变化后静默错位。
    child_index = faiss.read_index(str(child_dir / "index.faiss"))
    if child_index.ntotal != len(children):
        raise ValueError("Child 文本 FAISS 数量与生成问题后的 chunks.jsonl 不一致")
    print(f"层次化索引完成：{args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "index", "all"])
    parser.add_argument("--documents", type=Path, default=ROOT / "data/ohr_bench/documents.jsonl")
    parser.add_argument("--child-source", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "storage/ohr_bench_hierarchical_questions"
    )
    parser.add_argument("--backend", choices=["vllm", "local-awq"], default="vllm")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--awq-model", type=Path, default=DEFAULT_AWQ_MODEL)
    parser.add_argument("--llmqrt-root", type=Path, default=DEFAULT_LLMQRT_ROOT)
    parser.add_argument("--parent-question-count", type=int, default=3)
    parser.add_argument("--child-question-count", type=int, default=2)
    parser.add_argument("--child-tokens", type=int, default=240)
    parser.add_argument("--child-overlap-tokens", type=int, default=40)
    parser.add_argument("--child-max-page-chars", type=int, default=30000)
    parser.add_argument("--parent-batch-size", type=int, default=16)
    parser.add_argument("--parent-workers", type=int, default=16)
    parser.add_argument("--parent-max-tokens", type=int, default=320)
    parser.add_argument("--parent-max-input-chars", type=int, default=12000)
    parser.add_argument("--child-batch-size", type=int, default=64)
    parser.add_argument("--child-workers", type=int, default=32)
    parser.add_argument("--child-max-tokens", type=int, default=128)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"generate", "all"}:
        generate_stage(args)
    if args.stage in {"index", "all"}:
        index_stage(args)


if __name__ == "__main__":
    main()
