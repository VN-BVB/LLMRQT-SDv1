"""把 PDF、Word(.docx)、Markdown 和 TXT 统一转换成离线 documents.jsonl。"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from rag_core import write_jsonl


ROOT = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="文件或目录；目录会递归扫描")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/custom/documents.jsonl",
        help="统一后的原始文档 JSONL",
    )
    return parser.parse_args()


def discover_files(inputs: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        path = value.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                files.add(candidate)
    return sorted(files)


def clean_text(text: str) -> str:
    """统一换行与空白，同时保留空行作为段落边界。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def stable_id(path: Path, part: str = "") -> str:
    digest = hashlib.sha1(f"{path.resolve()}::{part}".encode("utf-8")).hexdigest()[:12]
    return f"doc-{digest}"


def read_plain(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("text", b"", 0, 1, f"无法识别文本编码：{path}")


def extract_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    common = {"title": path.stem, "source_path": str(path), "format": suffix.lstrip(".")}

    if suffix in {".txt", ".md", ".markdown"}:
        text = clean_text(read_plain(path))
        return [{"id": stable_id(path), **common, "text": text}] if text else []

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("读取 Word 需要 python-docx：pip install python-docx") from exc
        document = Document(path)
        # Word 原生 paragraph 就是最可靠的第一版结构；用空行保留边界。
        text = clean_text("\n\n".join(p.text for p in document.paragraphs if p.text.strip()))
        return [{"id": stable_id(path), **common, "text": text}] if text else []

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("读取 PDF 需要 pypdf：pip install pypdf") from exc
        rows = []
        for page_number, page in enumerate(PdfReader(path).pages, start=1):
            # 每页单独成 document，检索结果可以准确回溯页码。
            text = clean_text(page.extract_text(extraction_mode="layout") or "")
            if text:
                rows.append(
                    {
                        "id": stable_id(path, f"page-{page_number}"),
                        **common,
                        "page": page_number,
                        "text": text,
                    }
                )
        return rows

    raise ValueError(f"不支持的文件类型：{path}")


def main() -> None:
    args = parse_args()
    files = discover_files(args.inputs)
    if not files:
        raise ValueError(f"未发现支持的文件：{', '.join(sorted(SUPPORTED_SUFFIXES))}")

    rows = []
    for path in files:
        extracted = extract_file(path)
        rows.extend(extracted)
        print(f"{path}: {len(extracted)} 条")

    write_jsonl(args.output, rows)
    print(f"完成：{len(files)} 个文件 -> {len(rows)} 条原始记录")
    print(f"输出：{args.output}")


if __name__ == "__main__":
    main()
