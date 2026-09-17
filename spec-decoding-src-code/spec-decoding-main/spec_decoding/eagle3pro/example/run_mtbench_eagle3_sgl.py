#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MT-Bench 吞吐评测（对齐公开 EAGLE MT-Bench 基准口径）。

- 80 道双轮 open-ended 题（FastChat question.jsonl）
- 每题 turn1 → turn2，system prompt 与公开 EAGLE bench 一致
- 指标：completion tok/s、accept length = total_new_tokens / total_spec_rounds
- 对比：EAGLE3 speculative vs HF generate(use_cache=True) greedy baseline

跑法（80 题 × max_new_tokens=2048，Docker 内示例；将 <容器名>、<工程根> 换成你的环境）::

    docker exec <容器名> bash -lc '
    cd <工程根> && PYTHONPATH=. python3 -m spec_decoding.eagle3pro.example.run_mtbench_eagle3_sgl \\
        --num-questions 80 --max-new-tokens 2048
    '

本地（与容器内同一挂载路径时，在 <工程根> 下）::

    PYTHONPATH=. python3 -m spec_decoding.eagle3pro.example.run_mtbench_eagle3_sgl \\
        --num-questions 80 --max-new-tokens 2048

快速 smoke（10 题 × 256）::

    PYTHONPATH=. python3 -m spec_decoding.eagle3pro.example.run_mtbench_eagle3_sgl \\
        --num-questions 10 --max-new-tokens 256 --skip-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_DEFAULT_QUESTIONS = _HERE / "mtbench" / "question.jsonl"
_FASTCHAT_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)

MTBENCH_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as "
    "possible, while being safe.  Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content. Please ensure that your "
    "responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)


@dataclass
class RunStats:
    completion_tokens: int = 0
    spec_rounds: int = 0
    wall_s: float = 0.0


@dataclass
class BenchStats:
    spec: RunStats = field(default_factory=RunStats)
    baseline: RunStats = field(default_factory=RunStats)
    num_questions: int = 0


def _load_target(target_id: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fallback = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    last_err: Optional[Exception] = None
    for mid in [target_id, fallback]:
        try:
            print(f"[mtbench] loading target {mid} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            print(f"[mtbench] target ready: {mid}", flush=True)
            return tok, model, mid
        except Exception as e:
            print(f"[mtbench] failed to load {mid}: {e}", flush=True)
            last_err = e
    raise RuntimeError(f"could not load target; last error: {last_err}")


def _ensure_questions(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mtbench] downloading question.jsonl -> {path}", flush=True)
    import urllib.request

    urllib.request.urlretrieve(_FASTCHAT_QUESTIONS_URL, path)


def load_questions(path: Path, num_questions: int) -> List[Dict[str, Any]]:
    _ensure_questions(path)
    questions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions[:num_questions]


def _chat_input_ids(tok, messages: List[Dict[str, str]], device: torch.device) -> torch.LongTensor:
    if getattr(tok, "chat_template", None):
        out = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if isinstance(out, dict):
            ids = out["input_ids"]
        elif hasattr(out, "input_ids"):
            ids = out.input_ids
        else:
            ids = out
        return ids.to(device)
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return tok(text, return_tensors="pt")["input_ids"].to(device)


def _generate_spec_turn(
    gen,
    tok,
    messages: List[Dict[str, str]],
    *,
    max_new_tokens: int,
    eos: Optional[int],
    device: torch.device,
) -> Tuple[str, int, int, int, float]:
    input_ids = _chat_input_ids(tok, messages, device)
    plen = int(input_ids.shape[1])
    rounds_before = gen.n_rounds
    accepted_before = gen.n_accepted_tokens
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = gen.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_ids = out_ids[0, plen:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    n_new = len(new_ids)
    n_rounds = gen.n_rounds - rounds_before
    n_acc = gen.n_accepted_tokens - accepted_before
    return text, n_new, n_rounds, n_acc, dt


def _generate_baseline_turn(
    target,
    tok,
    messages: List[Dict[str, str]],
    *,
    max_new_tokens: int,
    eos: Optional[int],
    device: torch.device,
) -> Tuple[str, int, float]:
    input_ids = _chat_input_ids(tok, messages, device)
    plen = int(input_ids.shape[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = target.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=eos,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_ids = out[0, plen:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text, len(new_ids), dt


def _run_mtbench_conversations(
    tok,
    questions: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
    eos: Optional[int],
    device: torch.device,
    spec_gen=None,
    baseline_target=None,
) -> BenchStats:
    stats = BenchStats(num_questions=len(questions))
    base_messages: List[Dict[str, str]] = [{"role": "system", "content": MTBENCH_SYSTEM}]

    for qi, q in enumerate(questions):
        qid = q.get("question_id", qi)
        turns = q["turns"]
        print(f"[mtbench] question {qi + 1}/{len(questions)} id={qid}", flush=True)

        messages = list(base_messages)
        for ti, user_text in enumerate(turns[:2]):
            messages.append({"role": "user", "content": user_text})

            if spec_gen is not None:
                text, n_new, n_rounds, _n_acc, dt = _generate_spec_turn(
                    spec_gen,
                    tok,
                    messages,
                    max_new_tokens=max_new_tokens,
                    eos=eos,
                    device=device,
                )
                stats.spec.completion_tokens += n_new
                stats.spec.spec_rounds += n_rounds
                stats.spec.wall_s += dt
                print(
                    f"  turn {ti + 1} [spec]: new_tokens={n_new} wall={dt:.2f}s",
                    flush=True,
                )
            elif baseline_target is not None:
                text, bn, bdt = _generate_baseline_turn(
                    baseline_target,
                    tok,
                    messages,
                    max_new_tokens=max_new_tokens,
                    eos=eos,
                    device=device,
                )
                stats.baseline.completion_tokens += bn
                stats.baseline.wall_s += bdt
                print(
                    f"  turn {ti + 1} [baseline]: new_tokens={bn} wall={bdt:.2f}s",
                    flush=True,
                )
            else:
                text = ""

            messages.append({"role": "assistant", "content": text})

    return stats


def _format_report(
    args: argparse.Namespace,
    target_id: str,
    eagle_layers: List[int],
    stats: BenchStats,
) -> str:
    spec_tps = stats.spec.completion_tokens / max(stats.spec.wall_s, 1e-9)
    accept_len = stats.spec.completion_tokens / max(stats.spec.spec_rounds, 1)
    base_tps = stats.baseline.completion_tokens / max(stats.baseline.wall_s, 1e-9)
    speedup = spec_tps / base_tps if stats.baseline.wall_s > 0 else 0.0

    lines = [
        f"[eagle3_sgl mtbench] target={target_id} draft={args.eagle3_draft} "
        f"verify={args.verify_mode}(backend={args.verify_attn_backend}) "
        f"ar_draft={args.autoregressive_draft} inc_acts=True inc_draft_ext=True "
        f"eagle_layers={eagle_layers} topk={args.topk} num_steps={args.num_steps} "
        f"max_nodes={args.max_tree_nodes} dtype={args.dtype}",
        f"  num_questions={stats.num_questions}  max_new_tokens={args.max_new_tokens}",
        f"  completion_tokens={stats.spec.completion_tokens}  spec_rounds={stats.spec.spec_rounds}  "
        f"accept_length={accept_len:.2f}",
        f"  wall={stats.spec.wall_s:.2f}s  spec_tok/s={spec_tps:.2f}",
    ]
    if stats.baseline.wall_s > 0:
        lines.append(
            f"  baseline(HF greedy) completion_tokens={stats.baseline.completion_tokens}  "
            f"wall={stats.baseline.wall_s:.2f}s  base_tok/s={base_tps:.2f}  speedup={speedup:.2f}x"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="MT-Bench throughput for EAGLE3")
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eagle3-draft", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    p.add_argument("--question-file", type=Path, default=_DEFAULT_QUESTIONS)
    p.add_argument("--num-questions", type=int, default=80)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="每轮 assistant 生成上限（公开 EAGLE MT-Bench 默认 2048）")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--max-tree-nodes", type=int, default=64)
    p.add_argument("--verify-mode", default="full_model_tree",
                   choices=["reference_paths", "full_model_tree"])
    p.add_argument("--verify-attn-backend", default="auto",
                   choices=["auto", "eager", "flash_attn", "triton_tree"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no-autoregressive-draft", dest="autoregressive_draft", action="store_false")
    p.set_defaults(autoregressive_draft=True)
    p.add_argument("--skip-baseline", action="store_true",
                   help="只跑 speculative，跳过 HF greedy baseline")
    p.add_argument("--spec-only", action="store_true", help=argparse.SUPPRESS)  # alias
    p.add_argument("--perf-log", type=Path,
                   default=_HERE / "perf-mtbench.log")
    args = p.parse_args()
    if args.spec_only:
        args.skip_baseline = True

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle3pro import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
        make_eagle3_sgl_draft_topk_fn,
    )

    questions = load_questions(args.question_file, args.num_questions)
    print(f"[mtbench] loaded {len(questions)} questions from {args.question_file}", flush=True)

    tok, target, target_id = _load_target(args.target, device, dtype)
    n_layers = int(target.config.num_hidden_layers)
    eos = tok.eos_token_id

    draft_cfg_path = hf_hub_download(args.eagle3_draft, "config.json")
    draft_vocab_size = int(json.load(open(draft_cfg_path)).get("draft_vocab_size", 32000))
    draft_bin = hf_hub_download(args.eagle3_draft, "pytorch_model.bin")

    cfg = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        verify_mode=args.verify_mode,
        verify_attn_backend=args.verify_attn_backend,
    )
    eagle_layers = cfg.resolve_eagle_layers(n_layers)

    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        num_layers=1,
        draft_vocab_size=draft_vocab_size,
        draft_weights_path=draft_bin,
        device=device,
        dtype=dtype,
    )
    draft_topk_fn = make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk)
    gen = Eagle3SglGenerator(
        target,
        cfg,
        draft_topk_fn,
        draft_model=draft,
        autoregressive_draft=args.autoregressive_draft,
    )

    print("[mtbench] running EAGLE3 ...", flush=True)
    spec_stats = _run_mtbench_conversations(
        tok,
        questions,
        max_new_tokens=args.max_new_tokens,
        eos=eos,
        device=device,
        spec_gen=gen,
    ).spec

    base_stats = RunStats()
    if not args.skip_baseline:
        print("[mtbench] running HF greedy baseline ...", flush=True)
        base_stats = _run_mtbench_conversations(
            tok,
            questions,
            max_new_tokens=args.max_new_tokens,
            eos=eos,
            device=device,
            baseline_target=target,
        ).baseline

    stats = BenchStats(spec=spec_stats, baseline=base_stats, num_questions=len(questions))
    accept_len = stats.spec.completion_tokens / max(stats.spec.spec_rounds, 1)
    spec_tps = stats.spec.completion_tokens / max(stats.spec.wall_s, 1e-9)

    rec = _format_report(args, target_id, eagle_layers, stats)
    print("\n==================== MT-Bench EAGLE3 ====================")
    print(rec, end="")
    print(
        f"#questions: {stats.num_questions}, "
        f"Throughput: {spec_tps:.2f} token/s, "
        f"Acceptance length: {accept_len:.2f}"
    )
    print("=============================================================")

    args.perf_log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.perf_log, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        f.write(rec)
        f.write("\n")
    print(f"[mtbench] perf appended to {args.perf_log.resolve()}")


if __name__ == "__main__":
    main()
