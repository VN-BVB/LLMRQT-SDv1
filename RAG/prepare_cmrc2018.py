"""下载并整理小型中文公开数据集 CMRC 2018 trial。

原始 CMRC 文件是 SQuAD 风格的嵌套 JSON；本脚本把它展开成两个容易阅读的
JSONL 文件：documents.jsonl（知识库）和 questions.jsonl（评测问题）。
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from rag_core import write_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_URL = (
    "https://raw.githubusercontent.com/ymcui/cmrc2018/"
    "master/squad-style-data/cmrc2018_trial.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="已有的 CMRC JSON；不填就自动下载")
    parser.add_argument("--url", default=DEFAULT_URL, help="公开数据集下载地址")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "data/raw/cmrc2018_trial.json")
    parser.add_argument("--documents-output", type=Path, default=ROOT / "data/cmrc2018/documents.jsonl")
    parser.add_argument("--questions-output", type=Path, default=ROOT / "data/cmrc2018/questions.jsonl")
    parser.add_argument("--max-documents", type=int, default=0, help="0 表示保留全部段落")
    parser.add_argument("--max-questions", type=int, default=0, help="0 表示保留全部问题")
    return parser.parse_args()


def download(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载：{url}")
    request = urllib.request.Request(url, headers={"User-Agent": "rag-beginner-demo/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, output.open("wb") as file:
        file.write(response.read())
    print(f"已保存原始数据：{output}")
    return output


def convert(
    source: Path,
    documents_output: Path,
    questions_output: Path,
    max_documents: int = 0,
    max_questions: int = 0,
) -> tuple[int, int]:
    with source.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if "data" not in raw:
        raise ValueError("输入文件不是 SQuAD/CMRC 格式：根对象缺少 data 字段")

    documents = []
    questions = []
    stop = False

    for article_index, article in enumerate(raw["data"]):
        title = article.get("title", f"文章 {article_index}")
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            if max_documents and len(documents) >= max_documents:
                stop = True
                break

            doc_id = f"cmrc-{article_index:04d}-{paragraph_index:03d}"
            context = paragraph["context"].strip()
            documents.append(
                {
                    "id": doc_id,
                    "title": title,
                    "text": context,
                    "source": "CMRC 2018 trial",
                }
            )

            for qa_index, qa in enumerate(paragraph.get("qas", [])):
                if max_questions and len(questions) >= max_questions:
                    continue
                # 开发/试用数据常有多个等价人工答案，去重后全部保留。
                answers = []
                for answer in qa.get("answers", []):
                    text = answer.get("text", "").strip()
                    if text and text not in answers:
                        answers.append(text)
                if not answers:
                    continue
                questions.append(
                    {
                        "id": str(qa.get("id", f"q-{article_index}-{paragraph_index}-{qa_index}")),
                        "question": qa["question"].strip(),
                        "answers": answers,
                        "gold_doc_ids": [doc_id],
                    }
                )
        if stop:
            break

    write_jsonl(documents_output, documents)
    write_jsonl(questions_output, questions)
    return len(documents), len(questions)


def main() -> None:
    args = parse_args()
    source = args.input or args.raw_output
    if args.input is None and not source.exists():
        download(args.url, source)
    elif not source.exists():
        raise FileNotFoundError(source)

    doc_count, question_count = convert(
        source,
        args.documents_output,
        args.questions_output,
        args.max_documents,
        args.max_questions,
    )
    print(f"完成：{doc_count} 篇文档，{question_count} 个问题")
    print(f"知识库：{args.documents_output}")
    print(f"评测集：{args.questions_output}")


if __name__ == "__main__":
    main()

