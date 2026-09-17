#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
torch.profiler 剖析 EAGLE3-SGL：定位单步投机里最耗时 / 多余的操作。

输出三部分：
  1) 分阶段（record_function "stage:*"）device 时间占比：target_prefill / draft_prefill_init /
     draft_build_tree / target_verify / target_replay / draft_extend。
  2) Top-N 算子（按 self CUDA 时间）—— 看哪个 kernel 最贵。
  3) Top CPU 算子 / 同步点（aten::item / cuda*Synchronize / copy_ / repeat_interleave 等）——
     看哪些是调度 / 同步 / 多余 H2D-D2H开销。

设置 EAGLE3_SGL_PROFILE=1

跑法（确保在容器内）::

    cd /home/spec-decoding && PYTHONPATH=. python3 -m spec_decoding.eagle3_sgl.example.profile_eagle3_sgl \\
        --max-new-tokens 128 --topk 8 --num-steps 8 --max-tree-nodes 64 \\
        --trace-out spec_decoding/eagle3_sgl/example/eagle3_trace.json
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("EAGLE3_SGL_PROFILE", "1") 

import time

import torch
from torch.profiler import ProfilerActivity, profile


def _load_target(target_id: str, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fallback = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    last_err = None
    for mid in [target_id, fallback]:
        try:
            print(f"[profile] loading target {mid} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            print(f"[profile] target ready: {mid}", flush=True)
            return tok, model, mid
        except Exception as e:
            print(f"[profile] failed to load {mid}: {e}", flush=True)
            last_err = e
    raise RuntimeError(f"could not load target; last error: {last_err}")


def _stage_table(prof) -> str:
    """汇总 record_function "stage:*" 标签的 device 时间（含子 kernel）。"""
    rows = []
    for evt in prof.key_averages():
        key = evt.key
        if not key.startswith("stage:"):
            continue
        cuda_us = float(getattr(evt, "self_cuda_time_total", 0) or 0)
        dev_us = float(getattr(evt, "cuda_time_total", 0) or 0)
        cpu_us = float(getattr(evt, "cpu_time_total", 0) or 0)
        rows.append((key, evt.count, dev_us, cpu_us))
    rows.sort(key=lambda r: r[2], reverse=True)
    total_dev = sum(r[2] for r in rows) or 1.0
    out = ["\n================= 分阶段 device 时间（stage:* 标签，含子 kernel）=================",
           f"{'stage':<28}{'calls':>8}{'dev_ms':>12}{'%dev':>8}{'cpu_ms':>12}"]
    for key, cnt, dev_us, cpu_us in rows:
        out.append(
            f"{key:<28}{cnt:>8}{dev_us / 1e3:>12.2f}{100 * dev_us / total_dev:>7.1f}%{cpu_us / 1e3:>12.2f}"
        )
    out.append(f"{'(stage total)':<28}{'':>8}{total_dev / 1e3:>12.2f}{100.0:>7.1f}%")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="torch.profiler for EAGLE3-SGL")
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eagle3-draft", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup-tokens", type=int, default=32)
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
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--trace-out", default="")
    p.add_argument("--report-out", default="")
    args = p.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle3_sgl import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
        make_eagle3_sgl_draft_topk_fn,
    )
    import spec_decoding.eagle3_sgl.runner as runner_mod

    assert runner_mod._PROFILE, "EAGLE3_SGL_PROFILE 未启用（需在 import 前设置）"

    tok, target, target_id = _load_target(args.target, device, dtype)
    n_layers = int(target.config.num_hidden_layers)

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
    print(f"[profile] eagle_layers={eagle_layers} draft_vocab_size={draft_vocab_size}", flush=True)

    draft = build_eagle3_sgl_draft(
        target, eagle_layers, num_layers=1,
        draft_vocab_size=draft_vocab_size, draft_weights_path=draft_bin,
        device=device, dtype=dtype,
    )
    draft_topk_fn = make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk)
    gen = Eagle3SglGenerator(
        target, cfg, draft_topk_fn,
        draft_model=draft, autoregressive_draft=args.autoregressive_draft,
    )

    if getattr(tok, "chat_template", None):
        out = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(out, "input_ids"):
            ids = out.input_ids
        elif isinstance(out, dict):
            ids = out["input_ids"]
        else:
            ids = out
        input_ids = ids.to(device)
    else:
        input_ids = tok(args.prompt, return_tensors="pt")["input_ids"].to(device)
    eos = tok.eos_token_id

    # 预热（含 CUDA graph / kernel autotune / 编译缓存），不计入 profile
    print("[profile] warmup ...", flush=True)
    with torch.inference_mode():
        gen.generate(input_ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)
    gen.n_rounds = 0
    gen.n_accepted_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[profile] profiling generate(max_new_tokens={args.max_new_tokens}) ...", flush=True)
    t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        with torch.inference_mode():
            out_ids = gen.generate(input_ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n_new = int(out_ids.shape[1] - input_ids.shape[1])
    accept = gen.n_accepted_tokens / max(1, gen.n_rounds)

    header = (
        f"\n==================== EAGLE3-SGL torch.profiler ====================\n"
        f"target={target_id} verify={args.verify_mode}({args.verify_attn_backend}) "
        f"topk={args.topk} steps={args.num_steps} nodes={args.max_tree_nodes} dtype={args.dtype}\n"
        f"new_tokens={n_new} rounds={gen.n_rounds} accept/round={accept:.2f} "
        f"wall={wall:.2f}s spec_tok/s={n_new / wall:.2f}"
    )
    print(header)

    stage = _stage_table(prof)
    print(stage)

    by_self_cuda = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=args.top_n
    )
    by_cpu = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=args.top_n
    )
    print("\n================= Top 算子（self CUDA 时间）=================")
    print(by_self_cuda)
    print("\n================= Top 算子（self CPU 时间，看调度/同步）=================")
    print(by_cpu)

    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"[profile] chrome trace -> {os.path.abspath(args.trace_out)}")

    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(stage + "\n")
            f.write("\n== Top self CUDA ==\n")
            f.write(by_self_cuda + "\n")
            f.write("\n== Top self CPU ==\n")
            f.write(by_cpu + "\n")
        print(f"[profile] report -> {os.path.abspath(args.report_out)}")


if __name__ == "__main__":
    main()
