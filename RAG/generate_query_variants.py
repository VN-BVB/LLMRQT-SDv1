"""为固定评测问题生成 Rewrite、Step-back 与 Decomposition 查询变体。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from generation_backends import DEFAULT_AWQ_MODEL, DEFAULT_LLMQRT_ROOT, create_backend
from query_transform import PROMPT_VERSION, build_query_transform_messages, parse_query_transform
from rag_core import read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/ohr_bench/questions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ohr_bench/query_variants.jsonl")
    parser.add_argument("--backend", choices=["vllm", "local-awq"], default="vllm")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--awq-model", type=Path, default=DEFAULT_AWQ_MODEL)
    parser.add_argument("--llmqrt-root", type=Path, default=DEFAULT_LLMQRT_ROOT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.workers, args.max_tokens) <= 0 or args.limit < 0:
        raise ValueError("batch-size、workers、max-tokens 必须大于0，limit不能小于0")

    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    client = create_backend(
        args.backend,
        api_base=args.api_base,
        model=args.model,
        awq_model=args.awq_model,
        llmqrt_root=args.llmqrt_root,
        local_batch_size=args.batch_size,
    )
    rows = []
    failures = 0
    total_seconds = 0.0
    try:
        for start in range(0, len(questions), args.batch_size):
            current = questions[start : start + args.batch_size]
            messages = [build_query_transform_messages(row["question"]) for row in current]
            batch_started = time.perf_counter()
            outputs = client.chat_many(
                messages,
                max_tokens=args.max_tokens,
                temperature=0.0,
                max_concurrency=args.workers,
            )
            batch_seconds = time.perf_counter() - batch_started
            total_seconds += batch_seconds
            per_query_seconds = batch_seconds / len(current)
            for question, message, output in zip(current, messages, outputs):
                parsed = None
                last_output = output
                for attempt in range(args.retries + 1):
                    try:
                        parsed = parse_query_transform(last_output, question["question"])
                        break
                    except (ValueError, TypeError):
                        if attempt < args.retries:
                            last_output = client.chat(
                                message, max_tokens=args.max_tokens, temperature=0.0
                            )
                fallback = parsed is None
                if parsed is None:
                    failures += 1
                    parsed = {
                        "rewrite": question["question"],
                        "step_back": question["question"],
                        "subqueries": [question["question"]],
                    }
                rows.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        **parsed,
                        "fallback": fallback,
                        "transform_latency_seconds": per_query_seconds,
                    }
                )
            print(f"查询变体：{min(start + args.batch_size, len(questions))}/{len(questions)}")
    finally:
        client.unload()

    write_jsonl(args.output, rows)
    write_json(
        args.output.with_suffix(".meta.json"),
        {
            "question_count": len(rows),
            "prompt_version": PROMPT_VERSION,
            "backend": args.backend,
            "model": client.model,
            "fallback_count": failures,
            "total_seconds": total_seconds,
            "ms_per_query": total_seconds * 1000 / len(rows),
        },
    )
    print(f"查询变体完成：{args.output}，fallback={failures}/{len(rows)}")


if __name__ == "__main__":
    main()
