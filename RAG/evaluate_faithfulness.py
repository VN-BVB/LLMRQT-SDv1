"""对 evaluate_generation.py 的结果做逐事实忠实度与相关性评测。

Faithfulness 采用 RAGAS 常见的逐事实口径：先把回答拆成原子事实，再判断每个事实
是否能由检索上下文支持。这里使用本地 Qwen 作为 LLM judge，不依赖外部 API。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from generation_backends import DEFAULT_AWQ_MODEL, DEFAULT_LLMQRT_ROOT, create_backend
from rag_core import read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--chunks",
        type=Path,
        help="旧 predictions 未保存 text 时，用对应 chunks.jsonl 按 chunk id 补回证据",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend", choices=["vllm", "local-awq"], default="vllm")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--awq-model", type=Path, default=DEFAULT_AWQ_MODEL)
    parser.add_argument("--llmqrt-root", type=Path, default=DEFAULT_LLMQRT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context-chars", type=int, default=18000)
    return parser.parse_args()


def build_messages(row: dict[str, Any], max_context_chars: int) -> list[dict[str, str]]:
    contexts = []
    used = 0
    for number, chunk in enumerate(row.get("retrieved_chunks") or [], start=1):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        contexts.append(f"[Evidence {number}]\n{text}")
        used += len(text)
    evidence = "\n\n".join(contexts) or "(no evidence text was saved)"
    references = row.get("answers") or []
    user = (
        f"Question:\n{row['question']}\n\n"
        f"Reference answers:\n{json.dumps(references, ensure_ascii=False)}\n\n"
        f"Retrieved evidence:\n{evidence}\n\n"
        f"Answer to evaluate:\n{row.get('rag_answer') or ''}\n\n"
        "Return exactly one JSON object with this schema:\n"
        '{"claims":[{"claim":"...","supported":true}],'
        '"answer_relevance":0.0,"answer_correctness":0.0}\n'
        "Split the answer into atomic externally verifiable factual claims. "
        "A claim is supported only when the retrieved evidence entails it; do not use prior "
        "knowledge. Scores must be numbers from 0 to 1. "
        "answer_relevance only measures whether the response directly attempts to answer the "
        "question, independent of factual correctness: use 1 for a direct answer, 0.5 for a "
        "partial answer, and 0 only for an off-topic response or refusal. "
        "answer_correctness compares the answer with the reference answers: use 1 when they "
        "agree, 0.5 when partially correct, and 0 when they disagree. A concise numeric answer "
        "can still have answer_relevance 1 even when its number is incorrect."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a strict RAG evaluator. Judge only from the supplied evidence and "
                "reference answers. Output valid JSON only, without markdown or explanation."
            ),
        },
        {"role": "user", "content": user},
    ]


def parse_judgement(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    # Small local judges sometimes emit LaTeX-style invalid JSON escapes such as ``\$``.
    # Preserve the literal backslash instead of discarding an otherwise valid judgement.
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        # Also tolerate a sequence of top-level objects, e.g. one object for claims and two
        # following objects for the scalar scores. Merge them in their emitted order.
        decoder = json.JSONDecoder()
        cursor = 0
        result = {}
        while cursor < len(cleaned):
            while cursor < len(cleaned) and cleaned[cursor] in " \t\r\n,":
                cursor += 1
            if cursor >= len(cleaned):
                break
            current, cursor = decoder.raw_decode(cleaned, cursor)
            if not isinstance(current, dict):
                raise ValueError("judge 顶层 JSON 必须是对象")
            result.update(current)
    claims = result.get("claims")
    if not isinstance(claims, list):
        raise ValueError("judge 输出缺少 claims 数组")
    normalized_claims = []
    for claim in claims:
        if not isinstance(claim, dict) or not str(claim.get("claim") or "").strip():
            continue
        supported = claim.get("supported")
        if isinstance(supported, str):
            supported = supported.strip().lower() in {"true", "yes", "1", "supported"}
        normalized_claims.append(
            {"claim": str(claim["claim"]).strip(), "supported": bool(supported)}
        )

    def score(name: str) -> float:
        value = float(result.get(name, 0.0))
        return min(1.0, max(0.0, value))

    return {
        "claims": normalized_claims,
        "answer_relevance": score("answer_relevance"),
        "answer_correctness": score("answer_correctness"),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("predictions 为空")
    if args.chunks is not None:
        chunks_by_id = {str(chunk["id"]): chunk for chunk in read_jsonl(args.chunks)}
        for row in rows:
            enriched = []
            for saved in row.get("retrieved_chunks") or []:
                original = chunks_by_id.get(str(saved.get("id")), {})
                enriched.append({**original, **saved})
            row["retrieved_chunks"] = enriched
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("batch-size 和 workers 必须大于 0")
    output_dir = args.output_dir or args.predictions.parent
    client = create_backend(
        args.backend,
        api_base=args.api_base,
        model=args.model,
        awq_model=args.awq_model,
        llmqrt_root=args.llmqrt_root,
        local_batch_size=args.batch_size,
    )
    judged_rows = []
    parse_failures = 0
    try:
        for start in range(0, len(rows), args.batch_size):
            current = rows[start : start + args.batch_size]
            raw_outputs = client.chat_many(
                [build_messages(row, args.max_context_chars) for row in current],
                max_tokens=args.max_tokens,
                temperature=0.0,
                max_concurrency=args.workers,
            )
            for row, raw in zip(current, raw_outputs):
                try:
                    judgement = parse_judgement(raw)
                    claims = judgement["claims"]
                    faithfulness = (
                        sum(claim["supported"] for claim in claims) / len(claims)
                        if claims
                        else None
                    )
                    error = None
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    parse_failures += 1
                    judgement = {"claims": [], "answer_relevance": None, "answer_correctness": None}
                    faithfulness = None
                    error = str(exc)
                judged_rows.append(
                    {
                        "id": row["id"],
                        "faithfulness": faithfulness,
                        "hallucination_rate": (
                            None if faithfulness is None else 1.0 - faithfulness
                        ),
                        "answer_relevance": judgement["answer_relevance"],
                        "answer_correctness_judge": judgement["answer_correctness"],
                        "answer_f1": (row.get("scores") or {}).get("rag_f1"),
                        "claims": judgement["claims"],
                        "judge_raw": raw,
                        "parse_error": error,
                    }
                )
            print(f"忠实度评测：{min(start + args.batch_size, len(rows))}/{len(rows)}")
    finally:
        client.unload()

    def available_average(name: str) -> float | None:
        values = [float(row[name]) for row in judged_rows if row.get(name) is not None]
        return mean(values) if values else None

    faithfulness = available_average("faithfulness")
    answer_f1 = available_average("answer_f1")
    if faithfulness is not None and answer_f1 is not None and faithfulness + answer_f1 > 0:
        composite = 2 * faithfulness * answer_f1 / (faithfulness + answer_f1)
    else:
        composite = None
    summary = {
        "question_count": len(judged_rows),
        "judge_model": client.model,
        "judge_backend": args.backend,
        "faithfulness": faithfulness,
        "hallucination_rate": (
            None if faithfulness is None else 1.0 - faithfulness
        ),
        "answer_relevance": available_average("answer_relevance"),
        "answer_correctness_judge": available_average("answer_correctness_judge"),
        "answer_f1": answer_f1,
        "f1_faithfulness_hmean": composite,
        "parse_failure_count": parse_failures,
        "scored_faithfulness_count": sum(
            row["faithfulness"] is not None for row in judged_rows
        ),
        "method": "claim-level LLM-as-a-judge, RAGAS-style faithfulness",
        "caveat": (
            "本地 Qwen judge 与被评模型相同时存在自评偏差；正式报告应固定 judge prompt，"
            "并人工复核随机样本。"
        ),
    }
    write_jsonl(output_dir / "faithfulness_rows.jsonl", judged_rows)
    write_json(output_dir / "quality_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
