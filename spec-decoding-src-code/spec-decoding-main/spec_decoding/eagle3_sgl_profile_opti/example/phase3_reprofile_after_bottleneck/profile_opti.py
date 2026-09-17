#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""torch.profiler 剖析 优化版 eagle3_sgl_profile_opti（Step1-4 后）。

与 profile_eagle3_sgl.py 同结构，但 import的是优化版，并额外抽取关键计数
（cudaStreamSynchronize / Memcpy HtoD / aten::copy_ / aten::item）以刷新对照表。
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("EAGLE3_SGL_PROFILE", "1")

import time

import torch
from torch.profiler import ProfilerActivity, profile


def _load_target(target_id, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for mid in [target_id, "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            print(f"[profile] loading target {mid} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            return tok, model, mid
        except Exception as e:
            print(f"[profile] failed {mid}: {e}", flush=True)
    raise RuntimeError("could not load target")


def _stage_table(prof):
    rows = []
    for evt in prof.key_averages():
        if not evt.key.startswith("stage:"):
            continue
        dev_us = float(getattr(evt, "self_cuda_time_total", 0) or 0)
        cpu_us = float(getattr(evt, "cpu_time_total", 0) or 0)
        rows.append((evt.key, evt.count, dev_us, cpu_us))
    rows.sort(key=lambda r: r[2], reverse=True)
    total = sum(r[2] for r in rows) or 1.0
    out = ["\n========= 分阶段 device 时间（stage:* 标签，含子 kernel）=========",
           f"{'stage':<28}{'calls':>8}{'dev_ms':>12}{'%dev':>8}{'cpu_ms':>12}"]
    for k, c, d, cp in rows:
        out.append(f"{k:<28}{c:>8}{d / 1e3:>12.2f}{100 * d / total:>7.1f}%{cp / 1e3:>12.2f}")
    out.append(f"{'(stage total)':<28}{'':>8}{total / 1e3:>12.2f}{100.0:>7.1f}%")
    return "\n".join(out)


def _counts_table(prof):
    """抽取关键调度/同步/拷贝事件的计数与耗时，刷新对照表。"""
    want = ["cudaStreamSynchronize", "cudaMemcpyAsync", "Memcpy HtoD", "Memcpy DtoH",
            "aten::copy_", "aten::item", "aten::_local_scalar_dense", "aten::cat",
            "aten::index_select", "cudaLaunchKernel"]
    agg = {}
    for evt in prof.key_averages():
        for w in want:
            if w.lower() in evt.key.lower():
                cpu = float(getattr(evt, "cpu_time_total", 0) or 0)
                cu = float(getattr(evt, "self_cuda_time_total", 0) or 0)
                c, pc, pcu = agg.get(w, (0, 0.0, 0.0))
                agg[w] = (c + evt.count, pc + cpu, pcu + cu)
                break
    out = ["\n========= 关键事件计数（调度/同步/拷贝）=========",
           f"{'event':<28}{'count':>12}{'cpu_ms':>12}{'cuda_ms':>12}"]
    for w in want:
        if w in agg:
            c, cpu, cu = agg[w]
            out.append(f"{w:<28}{c:>12}{cpu / 1e3:>12.1f}{cu / 1e3:>12.1f}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eagle3-draft", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup-tokens", type=int, default=32)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--max-tree-nodes", type=int, default=64)
    p.add_argument("--verify-mode", default="full_model_tree")
    p.add_argument("--verify-attn-backend", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--trace-out", default="")
    p.add_argument("--cuda-graph", action="store_true", help="Step5: static KV + verify CUDA graph")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle3_sgl_profile_opti import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
        make_eagle3_sgl_draft_topk_fn,
    )
    import spec_decoding.eagle3_sgl_profile_opti.runner as runner_mod

    assert runner_mod._PROFILE, "EAGLE3_SGL_PROFILE 未启用"

    tok, target, tid = _load_target(args.target, device, dtype)
    n_layers = int(target.config.num_hidden_layers)
    dvocab = int(json.load(open(hf_hub_download(args.eagle3_draft, "config.json"))).get("draft_vocab_size", 32000))
    draft_bin = hf_hub_download(args.eagle3_draft, "pytorch_model.bin")
    cfg = Eagle3SglConfig(topk=args.topk, num_steps=args.num_steps, max_tree_nodes=args.max_tree_nodes,
                          verify_mode=args.verify_mode, verify_attn_backend=args.verify_attn_backend,
                          decode_cuda_graph=args.cuda_graph)
    el = cfg.resolve_eagle_layers(n_layers)
    draft = build_eagle3_sgl_draft(target, el, num_layers=1, draft_vocab_size=dvocab,
                                   draft_weights_path=draft_bin, device=device, dtype=dtype)
    gen = Eagle3SglGenerator(target, cfg, make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk),
                             draft_model=draft, autoregressive_draft=True)

    out = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (out.input_ids if hasattr(out, "input_ids") else out).to(device)
    eos = tok.eos_token_id

    print("[profile] warmup ...", flush=True)
    with torch.inference_mode():
        gen.generate(ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)
    gen.n_rounds = 0; gen.n_accepted_tokens = 0
    torch.cuda.synchronize()

    print("[profile] profiling ...", flush=True)
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            oid = gen.generate(ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_new = int(oid.shape[1] - ids.shape[1])
    accept = gen.n_accepted_tokens / max(1, gen.n_rounds)

    print(f"\n==================== eagle3_sgl_profile_opti torch.profiler ====================\n"
          f"target={tid} verify={args.verify_mode}({args.verify_attn_backend}) "
          f"topk={args.topk} steps={args.num_steps} nodes={args.max_tree_nodes}\n"
          f"new_tokens={n_new} rounds={gen.n_rounds} accept/round={accept:.2f} "
          f"wall={wall:.2f}s spec_tok/s={n_new / wall:.2f} ms/round={1e3 * wall / max(1, gen.n_rounds):.1f}")
    print(_stage_table(prof))
    print(_counts_table(prof))
    print("\n========= Top 算子（self CUDA）=========")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.top_n))
    print("\n========= Top 算子（self CPU，看调度/同步）=========")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top_n))
    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"[profile] trace -> {os.path.abspath(args.trace_out)}")


if __name__ == "__main__":
    main()
