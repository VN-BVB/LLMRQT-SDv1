"""Build a BGE/FAISS index from generated_questions stored in chunks.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path

from question_retriever import build_question_faiss_index


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument("--output", type=Path, help="默认 INDEX_DIR/questions/")
    parser.add_argument("--embedding-model")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = build_question_faiss_index(
        args.index_dir,
        output_dir=args.output,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        overwrite=args.overwrite,
    )
    print("问题 FAISS 索引完成：")
    print(f"  问题数：{metadata['question_count']}")
    print(f"  输出目录：{args.output or args.index_dir / 'questions'}")


if __name__ == "__main__":
    main()
