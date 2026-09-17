#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Step-1 验证：tree_verify_full._select_append_root_child 张量化贪心接受（去逐节点 .item()）。

做两件事：
  1) 正确性：同 prompt / 同配置下，eagle3_sgl_profile_opti（已改）与 eagle3_sgl（原始）输出
     token 序列必须逐位相同（greedy 等价，无损）。
  2) 收益：torch.profiler 分别测两边，报告 cudaStreamSynchronize 次数 + target_verify 阶段耗时 + tok/s。

跑法（容器内）::

    cd /home/spec-decoding && EAGLE3_SGL_PROFILE=1 PYTHONPATH=. \\
        python3 -m spec_decoding.eagle3_sgl_profile_opti.example.validate_step1 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("EAGLE3_SGL_PROFILE", "1")  # 必须在 import 包前

import torch
from torch.profiler import ProfilerActivity, profile


def _load_target(target_id, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for mid in [target_id, "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            return tok, model, mid
        except Exception as e:
            print(f"[v] failed {mid}: {e}", flush=True)
    raise RuntimeError("could not load target")


def _build_gen(pkg, target, n_layers, cfg_kwargs, draft_bin, draft_vocab, device, dtype, topk):
    cfg = pkg.Eagle3SglConfig(**cfg_kwargs)
    eagle_layers = cfg.resolve_eagle_layers(n_layers)
    draft = pkg.build_eagle3_sgl_draft(
        target, eagle_layers, num_layers=1, draft_vocab_size=draft_vocab,
        draft_weights_path=draft_bin, device=device, dtype=dtype,
    )
    fn = pkg.make_eagle3_sgl_draft_topk_fn(draft, topk=topk)
    return pkg.Eagle3SglGenerator(target, cfg, fn, draft_model=draft, autoregressive_draft=True)


def _profile_run(gen, input_ids, max_new_tokens, eos):
    gen.n_rounds = 0
    gen.n_accepted_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            out = gen.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sync = 0
    verify_ms = 0.0
    for evt in prof.key_averages():
        if "cudaStreamSynchronize" in evt.key:
            sync += evt.count
        if evt.key == "stage:target_verify":
            verify_ms = float(getattr(evt, "cpu_time_total", 0) or 0) / 1e3
    n_new = int(out.shape[1] - input_ids.shape[1])
    return out, dict(wall=wall, sync=sync, verify_ms=verify_ms, n_new=n_new,
                     rounds=gen.n_rounds, tok_s=n_new / wall)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eagle3-draft", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--max-tree-nodes", type=int, default=64)
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from huggingface_hub import hf_hub_download
    import spec_decoding.eagle3_sgl as orig
    import spec_decoding.eagle3_sgl_profile_opti as opti

    tok, target, target_id = _load_target(args.target, device, dtype)
    n_layers = int(target.config.num_hidden_layers)
    draft_vocab = int(json.load(open(hf_hub_download(args.eagle3_draft, "config.json"))).get("draft_vocab_size", 32000))
    draft_bin = hf_hub_download(args.eagle3_draft, "pytorch_model.bin")
    cfg_kwargs = dict(topk=args.topk, num_steps=args.num_steps, max_tree_nodes=args.max_tree_nodes,
                      verify_mode="full_model_tree", verify_attn_backend="auto")

    out = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt"
    )
    ids = out.input_ids if hasattr(out, "input_ids") else (out["input_ids"] if isinstance(out, dict) else out)
    input_ids = ids.to(device)
    eos = tok.eos_token_id

    print("[v] building generators ...", flush=True)
    g_orig = _build_gen(orig, target, n_layers, cfg_kwargs, draft_bin, draft_vocab, device, dtype, args.topk)
    g_opti = _build_gen(opti, target, n_layers, cfg_kwargs, draft_bin, draft_vocab, device, dtype, args.topk)

    with torch.inference_mode():
        g_orig.generate(input_ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)
        g_opti.generate(input_ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)

    print("[v] profiling ORIG ...", flush=True)
    out_o, s_o = _profile_run(g_orig, input_ids, args.max_new_tokens, eos)
    print("[v] profiling OPTI ...", flush=True)
    out_p, s_p = _profile_run(g_opti, input_ids, args.max_new_tokens, eos)

    # 正确性：逐位对齐（取较短长度对齐，二者应完全一致）
    lo = min(out_o.shape[1], out_p.shape[1])
    same = bool(torch.equal(out_o[:, :lo], out_p[:, :lo])) and out_o.shape[1] == out_p.shape[1]

    print("\n==================== Step-1 验证结果 ====================")
    print(f"target={target_id}  cfg=topk{args.topk}/steps{args.num_steps}/nodes{args.max_tree_nodes}")
    print(f"输出逐位一致(greedy 等价): {same}  (orig_len={out_o.shape[1]} opti_len={out_p.shape[1]})")
    print(f"\n{'metric':<28}{'ORIG':>14}{'OPTI':>14}{'变化':>12}")
    def row(name, ko, kp, unit="", better_low=True):
        vo, vp = s_o[ko], s_p[ko]
        delta = (vp - vo) / vo * 100 if vo else 0.0
        print(f"{name:<28}{vo:>14.2f}{vp:>14.2f}{delta:>11.1f}%")
    row("cudaStreamSynchronize 次数", "sync", "sync")
    row("target_verify 阶段(ms)", "verify_ms", "verify_ms")
    row("wall(s, 含profiler开销)", "wall", "wall")
    row("tok/s(含profiler开销)", "tok_s", "tok_s")
    print(f"\n[orig] rounds={s_o['rounds']} new={s_o['n_new']}   [opti] rounds={s_p['rounds']} new={s_p['n_new']}")
    if not same:
        print("\n[WARN] 输出不一致！打印前若干 diff 位置：")
        for i in range(lo):
            a, b = int(out_o[0, i]), int(out_p[0, i])
            if a != b:
                print(f"  pos {i}: orig={a} opti={b}")
                break


if __name__ == "__main__":
    main()
