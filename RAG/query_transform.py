"""LLM 查询改写、Step-back 回退与复杂问题拆解。"""

from __future__ import annotations

import json
import re
from typing import Any


PROMPT_VERSION = "rewrite_stepback_decompose_v1"


def build_query_transform_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你负责为 RAG 检索生成查询变体。只输出合法 JSON 对象，不要 Markdown、解释或答案。"
                "rewrite 是保留原问题实体、数字、年份和约束的简洁检索表达；"
                "step_back 是更抽象、更宽泛但仍与原问题直接相关的背景检索问题；"
                "subqueries 是解决原问题所需的 1 到 3 个可独立检索的子问题。"
                "不要回答问题，不要虚构原问题没有的实体、数字或条件。"
                "简单事实问题无需强行拆成多个子问题，可只输出一个。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"原问题：{question}\n"
                "输出格式："
                '{"rewrite":"...","step_back":"...","subqueries":["..."]}'
            ),
        },
    ]


def _clean_query(value: Any, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    value = " ".join(value.split()).strip()
    return value[:500] if value else fallback


def parse_query_transform(text: str, question: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("查询变体输出中没有 JSON 对象")
    candidate = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned[start : end + 1])
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("查询变体顶层必须是 JSON 对象")
    rewrite = _clean_query(value.get("rewrite"), question)
    step_back = _clean_query(value.get("step_back"), question)
    raw_subqueries = value.get("subqueries")
    if not isinstance(raw_subqueries, list):
        raw_subqueries = []
    subqueries = []
    seen = set()
    for item in raw_subqueries:
        query = _clean_query(item, "")
        normalized = re.sub(r"\W+", "", query).casefold()
        if query and normalized and normalized not in seen:
            seen.add(normalized)
            subqueries.append(query)
        if len(subqueries) >= 3:
            break
    if not subqueries:
        subqueries = [rewrite]
    return {
        "rewrite": rewrite,
        "step_back": step_back,
        "subqueries": subqueries,
    }


def selected_queries(row: dict[str, Any], mode: str) -> list[str]:
    original = str(row["question"])
    candidates = [original]
    if mode in {"rewrite", "all"}:
        candidates.append(str(row.get("rewrite") or original))
    if mode in {"step_back", "all"}:
        candidates.append(str(row.get("step_back") or original))
    if mode in {"decompose", "all"}:
        candidates.extend(str(item) for item in row.get("subqueries") or [])
    output = []
    seen = set()
    for query in candidates:
        query = " ".join(query.split()).strip()
        normalized = re.sub(r"\W+", "", query).casefold()
        if query and normalized and normalized not in seen:
            seen.add(normalized)
            output.append(query)
    return output
