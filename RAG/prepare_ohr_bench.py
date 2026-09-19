"""把 OHR-Bench Parquet + 官方 QA 转成当前 RAG 使用的 JSONL。"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_core import apply_heading, extract_headings, write_json, write_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET = ROOT / "data/ohr_bench/raw/OHR-Bench_v2.parquet"
DEFAULT_QA = ROOT / "data/ohr_bench/raw/qas_v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/ohr_bench"
DEFAULT_QA_URL = (
    "https://raw.githubusercontent.com/opendatalab/OHR-Bench/main/data/qas_v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--qa-input", type=Path, default=DEFAULT_QA)
    parser.add_argument("--qa-url", default=DEFAULT_QA_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--text-column",
        default="gt_text",
        help="默认使用人工核验文本；也可选择某个 OCR/noise 列做鲁棒性实验",
    )
    parser.add_argument("--domains", nargs="+", help="只保留指定领域，例如 academic manual")
    parser.add_argument("--max-documents", type=int, default=0, help="0 表示不限制逻辑 PDF 数")
    parser.add_argument(
        "--max-documents-per-domain",
        type=int,
        default=0,
        help="每个领域最多保留多少份 PDF；用于构造均衡冒烟集",
    )
    parser.add_argument("--max-pages", type=int, default=0, help="0 表示不限制页面记录数")
    parser.add_argument("--max-questions", type=int, default=0, help="0 表示保留全部匹配问题")
    return parser.parse_args()


def download_qa(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    print(f"QA 文件不存在，正在下载：{url}")
    request = urllib.request.Request(url, headers={"User-Agent": "rag-ohr-bench/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as file:
        file.write(response.read())
    temporary.replace(output)
    print(f"QA 已保存：{output}")


def logical_document_id(doc_name: str) -> str:
    digest = hashlib.sha1(doc_name.encode("utf-8")).hexdigest()[:16]
    return f"ohr-doc-{digest}"


def page_document_id(doc_name: str, page_index: int) -> str:
    return f"{logical_document_id(doc_name)}-page-{page_index:04d}"


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def load_page_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("读取 OHR-Bench Parquet 需要 pyarrow") from exc

    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)

    parquet = pq.ParquetFile(args.parquet)
    available = set(parquet.schema_arrow.names)
    required = {"domain", "doc_name", "page_idx", args.text_column}
    missing = required - available
    if missing:
        raise ValueError(f"Parquet 缺少字段：{sorted(missing)}；可用字段：{sorted(available)}")

    table = parquet.read(columns=["domain", "doc_name", "page_idx", args.text_column])
    allowed_domains = set(args.domains or [])
    rows = []
    for raw in table.to_pylist():
        domain = str(raw["domain"])
        if allowed_domains and domain not in allowed_domains:
            continue
        text = normalize_text(raw[args.text_column])
        if not text:
            continue
        rows.append(
            {
                "domain": domain,
                "doc_name": str(raw["doc_name"]),
                "page_index": int(raw["page_idx"]),
                "text": text,
            }
        )

    rows.sort(key=lambda item: (item["domain"], item["doc_name"], item["page_index"]))

    selected_documents: set[str] = set()
    per_domain_count: Counter[str] = Counter()
    for row in rows:
        doc_name = row["doc_name"]
        if doc_name in selected_documents:
            continue
        domain = row["domain"]
        if args.max_documents and len(selected_documents) >= args.max_documents:
            break
        if args.max_documents_per_domain and per_domain_count[domain] >= args.max_documents_per_domain:
            continue
        selected_documents.add(doc_name)
        per_domain_count[domain] += 1

    if args.max_documents or args.max_documents_per_domain:
        rows = [row for row in rows if row["doc_name"] in selected_documents]
    if args.max_pages:
        rows = rows[: args.max_pages]
    return rows


def convert_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_sections: defaultdict[str, list[str]] = defaultdict(list)
    documents = []

    for row in rows:
        doc_name = row["doc_name"]
        page_index = row["page_index"]
        initial_section_path = list(active_sections[doc_name])
        headings = extract_headings(row["text"])
        documents.append(
            {
                "id": page_document_id(doc_name, page_index),
                "document_id": logical_document_id(doc_name),
                "doc_name": doc_name,
                "title": Path(doc_name).name,
                "domain": row["domain"],
                "format": "structured_pdf_page",
                "source": "OHR-Bench",
                "page_index": page_index,
                "page": page_index + 1,
                "section_path": initial_section_path,
                "headings": headings,
                "text": row["text"],
            }
        )
        for heading in headings:
            active_sections[doc_name] = apply_heading(
                active_sections[doc_name], heading["level"], heading["title"]
            )
    return documents


def load_questions(
    qa_path: Path,
    documents: list[dict[str, Any]],
    max_questions: int,
) -> tuple[list[dict[str, Any]], int]:
    with qa_path.open("r", encoding="utf-8") as file:
        raw_questions = json.load(file)
    if not isinstance(raw_questions, list):
        raise ValueError("OHR-Bench QA 根对象应为 list")

    available_page_ids = {document["id"] for document in documents}
    questions = []
    skipped = 0
    for raw in raw_questions:
        page_indices = raw["evidence_page_no"]
        if isinstance(page_indices, int):
            page_indices = [page_indices]
        page_indices = list(dict.fromkeys(int(value) for value in page_indices))
        gold_doc_ids = [page_document_id(raw["doc_name"], page) for page in page_indices]
        # 子集评测必须包含一道题的全部证据页，否则 Recall 的分母和语义都会失真。
        if not gold_doc_ids or not all(page_id in available_page_ids for page_id in gold_doc_ids):
            skipped += 1
            continue

        questions.append(
            {
                "id": str(raw["ID"]),
                "question": str(raw["questions"]).strip(),
                "answers": [str(raw["answers"]).strip()],
                "gold_doc_ids": gold_doc_ids,
                "gold_document_ids": [logical_document_id(raw["doc_name"])],
                "doc_name": raw["doc_name"],
                "domain": raw["doc_type"],
                "answer_form": raw["answer_form"],
                "evidence_source": raw["evidence_source"],
                "evidence_context": raw["evidence_context"],
                "evidence_page_indices": page_indices,
                "evidence_pages": [page + 1 for page in page_indices],
            }
        )
        if max_questions and len(questions) >= max_questions:
            break
    return questions, skipped


def main() -> None:
    args = parse_args()
    if not args.qa_input.exists():
        download_qa(args.qa_url, args.qa_input)

    page_rows = load_page_rows(args)
    if not page_rows:
        raise ValueError("筛选后没有可用的 OHR-Bench 页面")
    documents = convert_documents(page_rows)
    questions, skipped_questions = load_questions(
        args.qa_input, documents, args.max_questions
    )
    if not questions:
        raise ValueError("没有问题能匹配当前文档子集，请放宽领域/文档/页面限制")

    documents_path = args.output_dir / "documents.jsonl"
    questions_path = args.output_dir / "questions.jsonl"
    write_jsonl(documents_path, documents)
    write_jsonl(questions_path, questions)

    domain_pages = Counter(document["domain"] for document in documents)
    domain_questions = Counter(question["domain"] for question in questions)
    logical_document_count = len({document["document_id"] for document in documents})
    write_json(
        args.output_dir / "dataset_meta.json",
        {
            "dataset": "OHR-Bench",
            "parquet": str(args.parquet.resolve()),
            "qa_file": str(args.qa_input.resolve()),
            "text_column": args.text_column,
            "logical_document_count": logical_document_count,
            "page_document_count": len(documents),
            "question_count": len(questions),
            "skipped_question_count": skipped_questions,
            "page_count_by_domain": dict(sorted(domain_pages.items())),
            "question_count_by_domain": dict(sorted(domain_questions.items())),
            "page_number_note": "page_index/evidence_page_indices are zero-based; page/evidence_pages are one-based",
        },
    )

    print(f"逻辑 PDF：{logical_document_count}")
    print(f"页面记录：{len(documents)}")
    print(f"可评测问题：{len(questions)}（因缺少证据页跳过 {skipped_questions}）")
    print(f"文档 JSONL：{documents_path}")
    print(f"问题 JSONL：{questions_path}")


if __name__ == "__main__":
    main()
