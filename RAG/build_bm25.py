"""从现有 chunks.jsonl 建立 SQLite FTS5/BM25 离线索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from bm25_retriever import build_bm25_index


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument("--output", type=Path, help="默认写到 INDEX_DIR/bm25/")
    parser.add_argument(
        "--include-generated-questions",
        action="store_true",
        help="将 generated_questions 追加到对应 Chunk 的 BM25 可搜索文本",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.index_dir / "bm25"
    metadata = build_bm25_index(
        args.index_dir / "chunks.jsonl",
        output,
        overwrite=args.overwrite,
        include_generated_questions=args.include_generated_questions,
    )
    print("BM25 索引完成：")
    print(f"  Chunk 数：{metadata['chunk_count']}")
    print(f"  索引文件：{output}")
    print(f"  分词方式：{metadata['tokenizer']}")


if __name__ == "__main__":
    main()
