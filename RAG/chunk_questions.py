"""离线为每个 Chunk 生成可由该 Chunk 独立回答的问题。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from generation_backends import ChatBackend


PROMPT_VERSION = "chunk_answerable_questions_v2_token_window"
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.I | re.S)
_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)、])\s*")


def build_question_messages(
    chunk: dict[str, Any],
    *,
    question_count: int,
) -> list[dict[str, str]]:
    """构造严格 JSON 输出 Prompt；问题语言跟随原文。"""

    section = " > ".join(chunk.get("section_path") or [])
    metadata = [
        f"标题：{chunk.get('title', '')}",
        f"领域：{chunk.get('domain', '')}",
        f"章节：{section}",
        f"页码：{chunk.get('page', '')}",
    ]
    return [
        {
            "role": "system",
            "content": (
                "你负责为检索增强生成系统离线构造问题。"
                f"请针对资料片段生成最多{question_count}个用户可能提出的问题。"
                "每个问题必须能够仅依靠片段直接回答，不得补充外部事实或进行无依据推断。"
                "问题应当彼此关注不同事实，并保留重要的人名、名称、年份、数字和单位。"
                "问题必须能够脱离‘本文/该片段/上述内容’等指代独立理解，并使用资料正文的主要语言。"
                "每个字符串必须是真正的问句，并以英文问号?或中文问号？结尾，不能输出标题或答案。"
                "240-token 窗口的开头或结尾可能是跨窗口的半句话；这不构成输出空数组的理由。"
                "只要窗口内存在名称、定义、方法、因果关系、比较、年份、数字或其他可验证事实，"
                "就必须生成问题。只有整段确实全是页眉页脚、作者名单、乱码或没有任何可回答事实时，"
                "才输出空数组。"
                "只输出由字符串组成的合法 JSON 数组，不要 Markdown、编号、解释或思考过程。"
            ),
        },
        {
            "role": "user",
            "content": "\n".join(metadata) + f"\n资料片段：\n{chunk['text']}",
        },
    ]


def _json_array_from_text(text: str) -> list[Any] | None:
    cleaned = _THINK_PATTERN.sub("", text).strip()
    cleaned = _FENCE_PATTERN.sub("", cleaned).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", cleaned):
        candidate_text = cleaned[match.start() :]
        try:
            value, _ = decoder.raw_decode(candidate_text)
        except json.JSONDecodeError:
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate_text)
            try:
                value, _ = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                continue
        if isinstance(value, list):
            return value
    return None


def parse_generated_questions(text: str, *, question_count: int) -> list[str]:
    """解析模型输出，兼容 JSON code fence，并清理重复问题。"""

    if question_count <= 0:
        raise ValueError("question_count 必须大于 0")
    values = _json_array_from_text(text)
    if values is None:
        # 少数模型即使被要求输出 JSON，仍可能输出编号列表；允许一次保守回退。
        lines = []
        for line in _THINK_PATTERN.sub("", text).splitlines():
            has_list_prefix = _LIST_PREFIX_PATTERN.match(line) is not None
            candidate = _LIST_PREFIX_PATTERN.sub("", line).strip().strip('"“”')
            if candidate and (has_list_prefix or candidate.endswith(("?", "？"))):
                lines.append(candidate)
        if not lines:
            raise ValueError(f"无法从模型输出解析问题数组：{text[:200]!r}")
        values = lines

    questions: list[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        question = " ".join(value.split()).strip()
        if not question or len(question) > 300 or not question.endswith(("?", "？")):
            continue
        normalized = re.sub(r"\W+", "", question).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        questions.append(question)
        if len(questions) >= question_count:
            break
    return questions


def _load_checkpoint(
    path: Path,
    chunks: list[dict[str, Any]],
) -> int:
    """恢复按 Chunk 顺序追加的临时检查点，容忍最后一行写入中断。"""

    if not path.exists():
        return 0
    completed = 0
    valid_lines: list[str] = []
    truncated_tail = False
    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # 进程被终止时只可能留下不完整的最后一行。
                if line_number == len(lines):
                    truncated_tail = True
                    break
                raise ValueError(f"问题生成检查点第 {line_number} 行损坏：{path}")
            if completed >= len(chunks) or row.get("id") != chunks[completed]["id"]:
                raise ValueError(
                    f"问题生成检查点与当前 Chunk 顺序不一致：{path} 第 {line_number} 行"
                )
            questions = row.get("generated_questions")
            if not isinstance(questions, list) or not all(
                isinstance(question, str) for question in questions
            ):
                raise ValueError(f"问题生成检查点格式错误：{path} 第 {line_number} 行")
            chunks[completed]["generated_questions"] = questions
            chunks[completed]["question_generation_fallback"] = bool(
                row.get("question_generation_fallback", False)
            )
            completed += 1
            valid_lines.append(line if line.endswith("\n") else line + "\n")
    if truncated_tail:
        # 否则后续 append 会把新记录接在损坏的半行之后。
        with path.open("w", encoding="utf-8") as file:
            file.writelines(valid_lines)
    return completed


def generate_questions_for_chunks(
    chunks: list[dict[str, Any]],
    backend: ChatBackend,
    *,
    checkpoint_path: str | Path,
    question_count: int = 3,
    batch_size: int = 8,
    max_concurrency: int = 16,
    max_tokens: int = 192,
    temperature: float = 0.0,
    retries: int = 2,
    resume: bool = True,
    limit: int = 0,
) -> dict[str, Any]:
    """批量生成问题并按完成顺序追加临时检查点。"""

    if question_count <= 0:
        raise ValueError("question_count 必须大于 0")
    if batch_size <= 0 or max_concurrency <= 0 or max_tokens <= 0:
        raise ValueError("batch_size、max_concurrency 和 max_tokens 必须大于 0")
    if retries < 0:
        raise ValueError("retries 不能小于 0")
    if limit < 0:
        raise ValueError("limit 不能小于 0")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and not resume:
        checkpoint_path.unlink()
    completed = _load_checkpoint(checkpoint_path, chunks) if resume else 0
    target_count = min(len(chunks), limit) if limit else len(chunks)
    if completed > target_count:
        raise ValueError(
            f"检查点已有 {completed} 条，但本次只要求生成 {target_count} 条；"
            "请使用 --no-question-resume"
        )

    from tqdm.auto import tqdm

    progress = tqdm(
        total=target_count,
        initial=completed,
        desc="LLM 生成 Chunk 问题",
        unit="chunk",
        dynamic_ncols=True,
    )
    generated_question_count = sum(
        len(chunk.get("generated_questions", [])) for chunk in chunks[:completed]
    )
    empty_chunk_count = sum(
        not chunk.get("generated_questions") for chunk in chunks[:completed]
    )
    fallback_count = sum(
        bool(chunk.get("question_generation_fallback")) for chunk in chunks[:completed]
    )

    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for start in range(completed, target_count, batch_size):
            current_chunks = chunks[start : min(start + batch_size, target_count)]
            messages_batch = [
                build_question_messages(chunk, question_count=question_count)
                for chunk in current_chunks
            ]
            raw_outputs: list[str] | None = None
            last_error: Exception | None = None
            for _ in range(retries + 1):
                try:
                    raw_outputs = backend.chat_many(
                        messages_batch,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_concurrency=max_concurrency,
                    )
                    if len(raw_outputs) != len(current_chunks):
                        raise RuntimeError(
                            "批量生成返回数量不一致："
                            f"期望 {len(current_chunks)}，实际 {len(raw_outputs)}"
                        )
                    break
                except Exception as exc:  # 网络或推理错误按参数重试整批
                    last_error = exc
            if raw_outputs is None:
                raise RuntimeError(f"Chunk 问题生成失败，起始位置 {start}") from last_error

            parsed: list[list[str] | None] = []
            for output in raw_outputs:
                try:
                    parsed.append(
                        parse_generated_questions(output, question_count=question_count)
                    )
                except ValueError:
                    parsed.append(None)

            # 只对格式错误的少数输出逐条重试，避免整批重新生成。
            for offset, questions in enumerate(parsed):
                if questions is not None:
                    continue
                output: str | None = None
                last_error = None
                for _ in range(retries + 1):
                    try:
                        output = backend.chat(
                            messages_batch[offset],
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        parsed[offset] = parse_generated_questions(
                            output,
                            question_count=question_count,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                if parsed[offset] is None:
                    parsed[offset] = []
                    current_chunks[offset]["question_generation_fallback"] = True
                    fallback_count += 1

            for chunk, questions in zip(current_chunks, parsed):
                assert questions is not None
                chunk["generated_questions"] = questions
                chunk.setdefault("question_generation_fallback", False)
                generated_question_count += len(questions)
                empty_chunk_count += int(not questions)
                checkpoint.write(
                    json.dumps(
                        {
                            "id": chunk["id"],
                            "generated_questions": questions,
                            "question_generation_fallback": chunk[
                                "question_generation_fallback"
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            checkpoint.flush()
            progress.update(len(current_chunks))
    progress.close()

    # limit 仅用于冒烟测试；未处理的 Chunk 明确保存为空数组。
    for chunk in chunks[target_count:]:
        chunk.setdefault("generated_questions", [])

    return {
        "prompt_version": PROMPT_VERSION,
        "model": backend.model,
        "requested_questions_per_chunk": question_count,
        "processed_chunk_count": target_count,
        "generated_question_count": generated_question_count,
        "empty_chunk_count": empty_chunk_count,
        "batch_size": batch_size,
        "max_concurrency": max_concurrency,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "fallback_count": fallback_count,
    }
