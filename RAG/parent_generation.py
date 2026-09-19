"""离线生成页级 Parent Summary 与可检索问题，支持断点续跑。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from generation_backends import ChatBackend


PROMPT_VERSION = "page_summary_and_questions_v1"
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.I | re.S)


def build_parent_messages(
    document: dict[str, Any],
    *,
    question_count: int,
    max_input_chars: int,
) -> list[dict[str, str]]:
    text = str(document.get("text", ""))
    if len(text) > max_input_chars:
        # 页首通常包含标题/章节，页尾可能包含跨页结论；同时保留两端。
        head = int(max_input_chars * 0.75)
        tail = max_input_chars - head
        text = text[:head] + "\n[中间内容因上下文长度省略]\n" + text[-tail:]
    section = " > ".join(document.get("section_path") or [])
    return [
        {
            "role": "system",
            "content": (
                "你负责为层次化检索增强生成系统构造页级索引。"
                "阅读一页资料后输出一个合法 JSON 对象，字段必须是 summary 和 questions。"
                "summary 应覆盖页面主题、关键实体、事实、数字与结论，使用原文主要语言，"
                "不得加入页面之外的知识，最多 120 个英文单词或 220 个中文字符。"
                f"questions 是最多 {question_count} 个字符串组成的数组；问题必须能依据本页直接回答，"
                "彼此关注不同要点，保留关键专名、年份、数字和单位，并能脱离‘本文/本页’独立理解。"
                "每个问题必须以英文问号?或中文问号？结尾，不能输出标题或答案。"
                "只输出 JSON，不要 Markdown、编号、解释或思考过程。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"资料名：{document.get('doc_name', document.get('title', ''))}\n"
                f"章节：{section}\n页码：{document.get('page', '')}\n正文：\n{text}"
            ),
        },
    ]


def parse_parent_output(text: str, *, question_count: int) -> tuple[str, list[str]]:
    cleaned = _THINK_PATTERN.sub("", text).strip()
    decoder = json.JSONDecoder()
    value = None
    for match in re.finditer(r"\{", cleaned):
        candidate_text = cleaned[match.start() :]
        try:
            candidate, _ = decoder.raw_decode(candidate_text)
        except json.JSONDecodeError:
            # 论文中的 LaTeX（如 \xi）常被模型原样放进 JSON 字符串，
            # 但它不是合法 JSON escape；仅转义这些非法反斜杠后再解析。
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate_text)
            try:
                candidate, _ = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, dict):
            value = candidate
            break
    if value is None:
        # 小模型偶尔会在 JSON 字符串内直接写未转义的引号（如 "recall"）。
        # 字段边界仍清晰时保守提取，避免一条脏输出终止数小时离线任务。
        match = re.search(
            r'"summary"\s*:\s*"(.*?)"\s*,\s*"questions"\s*:\s*\[(.*)\]\s*\}',
            cleaned,
            re.S,
        )
        if not match:
            raise ValueError(f"无法解析 Parent JSON：{text[:200]!r}")
        summary = " ".join(match.group(1).split()).strip()
        raw_questions = []
        for line in match.group(2).splitlines():
            question = line.strip().rstrip(",").strip().strip('"')
            if question:
                raw_questions.append(question)
    else:
        summary = " ".join(str(value.get("summary", "")).split()).strip()
        raw_questions = value.get("questions", [])
    if not summary or not isinstance(raw_questions, list):
        raise ValueError(f"Parent JSON 缺少 summary/questions：{text[:200]!r}")
    questions: list[str] = []
    seen = set()
    for raw in raw_questions:
        if not isinstance(raw, str):
            continue
        question = " ".join(raw.split()).strip()
        normalized = re.sub(r"\W+", "", question).casefold()
        if (
            not question
            or len(question) > 300
            or not question.endswith(("?", "？"))
            or not normalized
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        questions.append(question)
        if len(questions) >= question_count:
            break
    return summary, questions


def _load_checkpoint(path: Path, documents: list[dict[str, Any]]) -> int:
    if not path.exists():
        return 0
    completed = 0
    valid_lines: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                break
            raise ValueError(f"Parent 检查点第 {line_number} 行损坏：{path}")
        if completed >= len(documents) or row.get("id") != documents[completed].get("id"):
            raise ValueError(f"Parent 检查点与文档顺序不一致：{path} 第 {line_number} 行")
        documents[completed]["summary"] = row["summary"]
        documents[completed]["generated_questions"] = row.get("generated_questions", [])
        documents[completed]["parent_generation_fallback"] = bool(
            row.get("parent_generation_fallback", False)
        )
        completed += 1
        valid_lines.append(line if line.endswith("\n") else line + "\n")
    if len(valid_lines) != len([line for line in lines if line.strip()]):
        with path.open("w", encoding="utf-8") as file:
            file.writelines(valid_lines)
    return completed


def generate_parent_records(
    documents: list[dict[str, Any]],
    backend: ChatBackend,
    *,
    checkpoint_path: str | Path,
    question_count: int = 3,
    batch_size: int = 16,
    max_concurrency: int = 16,
    max_tokens: int = 320,
    max_input_chars: int = 12000,
    retries: int = 2,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """生成 Parent；每完成一个 batch 就追加检查点。"""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and not resume:
        checkpoint_path.unlink()
    completed = _load_checkpoint(checkpoint_path, documents) if resume else 0
    fallback_count = sum(
        bool(row.get("parent_generation_fallback")) for row in documents[:completed]
    )

    from tqdm.auto import tqdm

    progress = tqdm(
        total=len(documents), initial=completed, desc="生成 Parent Summary+问题", unit="page"
    )
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for start in range(completed, len(documents), batch_size):
            current = documents[start : start + batch_size]
            messages = [
                build_parent_messages(
                    row,
                    question_count=question_count,
                    max_input_chars=max_input_chars,
                )
                for row in current
            ]
            outputs = backend.chat_many(
                messages,
                max_tokens=max_tokens,
                temperature=0.0,
                max_concurrency=max_concurrency,
            )
            for offset, (row, output) in enumerate(zip(current, outputs)):
                parsed = None
                last_error: Exception | None = None
                for attempt in range(retries + 1):
                    try:
                        parsed = parse_parent_output(output, question_count=question_count)
                        break
                    except ValueError as exc:
                        last_error = exc
                        if attempt < retries:
                            output = backend.chat(messages[offset], max_tokens=max_tokens, temperature=0.0)
                if parsed is None:
                    # 极少数 OCR 页会诱导小模型输出别的 schema。保留可检索的
                    # 原文首段作为降级摘要，并显式打标，不能伪装成正常 LLM Summary。
                    summary = " ".join(str(row.get("text", "")).split())[:1200]
                    questions = []
                    row["parent_generation_fallback"] = True
                    fallback_count += 1
                else:
                    summary, questions = parsed
                    row["parent_generation_fallback"] = False
                row["summary"] = summary
                row["generated_questions"] = questions
                checkpoint.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "summary": summary,
                            "generated_questions": questions,
                            "parent_generation_fallback": row[
                                "parent_generation_fallback"
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            checkpoint.flush()
            progress.update(len(current))
    progress.close()

    parents = []
    for document in documents:
        parent = {key: value for key, value in document.items() if key != "text"}
        parent["source_id"] = document["id"]
        parent["parent_id"] = document["id"]
        parent["text"] = document["summary"]
        parent["generated_questions"] = document.get("generated_questions", [])
        parents.append(parent)
    return parents, {
        "prompt_version": PROMPT_VERSION,
        "parent_count": len(parents),
        "question_count_per_parent": question_count,
        "max_input_chars": max_input_chars,
        "max_tokens": max_tokens,
        "fallback_count": fallback_count,
    }
