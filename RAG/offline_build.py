"""离线阶段：切分文档、生成向量并保存 JSONL + FAISS。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from chunk_questions import generate_questions_for_chunks
from generation_backends import DEFAULT_AWQ_MODEL, DEFAULT_LLMQRT_ROOT, create_backend
from rag_core import (
    BGEEmbedder,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_PREFIX,
    read_jsonl,
    split_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=ROOT / "data/ohr_bench/documents.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--query-prefix",
        default=DEFAULT_QUERY_PREFIX,
        help="查询侧指令前缀；BGE-M3 默认使用空字符串",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--chunk-size", type=int, default=500, help="自然段超过此长度才继续切分")
    parser.add_argument("--overlap", type=int, default=80, help="仅超长自然段切分时重复的字符数")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument(
        "--generate-questions",
        action="store_true",
        help="切块后离线为每个 Chunk 生成可回答问题，再建立原文向量索引",
    )
    parser.add_argument("--question-count", type=int, default=3)
    parser.add_argument(
        "--question-backend",
        choices=["vllm", "local-awq"],
        default="local-awq",
        help="vllm=并发 HTTP 连续批处理；local-awq=本地 LLMQRT 批量生成",
    )
    parser.add_argument("--question-api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--question-model", help="vLLM served model name；默认自动查询")
    parser.add_argument("--question-awq-model", type=Path, default=DEFAULT_AWQ_MODEL)
    parser.add_argument("--question-llmqrt-root", type=Path, default=DEFAULT_LLMQRT_ROOT)
    parser.add_argument(
        "--question-batch-size",
        type=int,
        default=8,
        help="本地 AWQ 的 GPU batch；vLLM 模式下是每轮提交数量",
    )
    parser.add_argument(
        "--question-workers",
        type=int,
        default=16,
        help="vLLM HTTP 并发数；本地 AWQ 模式忽略此参数",
    )
    parser.add_argument("--question-max-tokens", type=int, default=192)
    parser.add_argument("--question-temperature", type=float, default=0.0)
    parser.add_argument("--question-retries", type=int, default=2)
    parser.add_argument(
        "--question-limit",
        type=int,
        default=0,
        help="仅生成前 N 个 Chunk，用于冒烟测试；0 表示全部",
    )
    parser.add_argument(
        "--question-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="从 OUTPUT_DIR/chunks.questions.partial.jsonl 断点续跑",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_jsonl(args.documents)
    if not documents:
        raise ValueError(f"知识库为空：{args.documents}")

    # chunks.jsonl 的行号和 FAISS 中的向量编号严格一一对应。
    chunks = []
    embedding_texts = []
    for document in documents:
        for chunk_index, piece in enumerate(
            split_text(
                document["text"],
                args.chunk_size,
                args.overlap,
                initial_section_path=document.get("section_path"),
            )
        ):
            chunk = {
                "id": f"{document['id']}-chunk-{chunk_index:03d}",
                "source_id": document["id"],
                "title": document.get("title", ""),
                "text": piece["text"],
                "start": piece["start"],
                "end": piece["end"],
                "paragraph_index": piece["paragraph_index"],
                "section_path": piece["section_path"],
            }
            for metadata_key in (
                "source_path",
                "source",
                "format",
                "domain",
                "document_id",
                "doc_name",
                "page_index",
                "page",
            ):
                if metadata_key in document:
                    chunk[metadata_key] = document[metadata_key]
            chunks.append(chunk)
            # 标题、领域和章节也是有价值的检索信息，所以和正文一起编码。
            embedding_parts = [chunk["title"]]
            if chunk.get("domain"):
                embedding_parts.append(str(chunk["domain"]))
            if chunk["section_path"]:
                embedding_parts.append(" > ".join(chunk["section_path"]))
            embedding_parts.append(chunk["text"])
            embedding_texts.append("\n".join(part for part in embedding_parts if part).strip())

    print(f"读取 {len(documents)} 篇文档，切成 {len(chunks)} 个 chunk")
    question_generation = {"enabled": False}
    question_checkpoint = args.output_dir / "chunks.questions.partial.jsonl"
    if args.generate_questions:
        for name in (
            "question_count",
            "question_batch_size",
            "question_workers",
            "question_max_tokens",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"{name.replace('_', '-')} 必须大于 0")
        if args.question_retries < 0 or args.question_limit < 0:
            raise ValueError("question-retries 和 question-limit 不能小于 0")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "正在加载问题生成模型："
            f"backend={args.question_backend}, batch={args.question_batch_size}"
        )
        question_backend = create_backend(
            args.question_backend,
            api_base=args.question_api_base,
            model=args.question_model,
            awq_model=args.question_awq_model,
            llmqrt_root=args.question_llmqrt_root,
            local_batch_size=args.question_batch_size,
        )
        try:
            question_generation = generate_questions_for_chunks(
                chunks,
                question_backend,
                checkpoint_path=question_checkpoint,
                question_count=args.question_count,
                batch_size=args.question_batch_size,
                max_concurrency=args.question_workers,
                max_tokens=args.question_max_tokens,
                temperature=args.question_temperature,
                retries=args.question_retries,
                resume=args.question_resume,
                limit=args.question_limit,
            )
            question_generation.update(
                {
                    "enabled": True,
                    "backend": args.question_backend,
                    "resume": args.question_resume,
                }
            )
        finally:
            # 本地 AWQ 与 BGE 顺序使用同一块 GPU，避免两个模型同时驻留。
            question_backend.unload()

    print(f"正在加载 Embedding 模型：{args.embedding_model} ({args.device})")
    embedder = BGEEmbedder(
        model_name=args.embedding_model,
        device=args.device,
        query_prefix=args.query_prefix,
        max_length=args.embedding_max_length,
    )
    vectors = embedder.encode(embedding_texts, is_query=False, batch_size=args.batch_size)

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("缺少 faiss，请先安装 requirements.txt") from exc

    # IndexFlatIP 是精确搜索，不做近似压缩。小数据集最适合用它验证正确性。
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "chunks.jsonl", chunks)
    faiss.write_index(index, str(args.output_dir / "index.faiss"))
    write_json(
        args.output_dir / "index_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents_file": str(args.documents.resolve()),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "chunk_strategy": "heading_aware_paragraph_then_sentence_window_for_oversized_paragraphs",
            "embedding_model": args.embedding_model,
            "embedding_dimension": int(vectors.shape[1]),
            "embedding_max_length": args.embedding_max_length,
            "query_prefix": args.query_prefix,
            "normalized": True,
            "index_type": "IndexFlatIP",
            "question_generation": question_generation,
        },
    )
    if question_checkpoint.exists():
        question_checkpoint.unlink()

    print("离线索引完成：")
    print(f"  chunk 内容：{args.output_dir / 'chunks.jsonl'}")
    print(f"  FAISS 索引：{args.output_dir / 'index.faiss'}")
    print(f"  索引说明：{args.output_dir / 'index_meta.json'}")


if __name__ == "__main__":
    main()
