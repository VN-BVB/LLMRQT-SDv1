#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""无 profiler 的干净 A/B：原始 eagle3_sgl vs 优化版 eagle3_sgl_profile_opti 的真实吞吐 tok/s 测试。"""

from __future__ import annotations

import argparse
import json
import time

import torch


def _load_target(target_id, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for mid in [target_id, "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            return tok, m, mid
        except Exception as e:
            print(f"[ab] failed {mid}: {e}", flush=True)
    raise RuntimeError("no target")


def _build(pkg, target, n_layers, cfgk, draft_bin, dvocab, device, dtype, topk):
    cfg = pkg.Eagle3SglConfig(**cfgk)
    el = cfg.resolve_eagle_layers(n_layers)
    draft = pkg.build_eagle3_sgl_draft(target, el, num_layers=1, draft_vocab_size=dvocab,
                                       draft_weights_path=draft_bin, device=device, dtype=dtype)
    fn = pkg.make_eagle3_sgl_draft_topk_fn(draft, topk=topk)
    return pkg.Eagle3SglGenerator(target, cfg, fn, draft_model=draft, autoregressive_draft=True)


def _timed(gen, ids, mnt, eos, reps):
    best = None
    for _ in range(reps):
        gen.n_rounds = 0; gen.n_accepted_tokens = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = gen.generate(ids, max_new_tokens=mnt, eos_token_id=eos)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = int(out.shape[1] - ids.shape[1])
        best = (n / dt, n, dt, gen.n_rounds, gen.n_accepted_tokens / max(1, gen.n_rounds), out) if best is None or n / dt > best[0] else best
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eagle3-draft", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--reps", type=int, default=3)
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

    tok, target, tid = _load_target(args.target, device, dtype)
    n_layers = int(target.config.num_hidden_layers)
    dvocab = int(json.load(open(hf_hub_download(args.eagle3_draft, "config.json"))).get("draft_vocab_size", 32000))
    draft_bin = hf_hub_download(args.eagle3_draft, "pytorch_model.bin")
    cfgk = dict(topk=args.topk, num_steps=args.num_steps, max_tree_nodes=args.max_tree_nodes,
                verify_mode="full_model_tree", verify_attn_backend="auto")
    out = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = out.input_ids if hasattr(out, "input_ids") else (out["input_ids"] if isinstance(out, dict) else out)
    ids = ids.to(device); eos = tok.eos_token_id

    go = _build(orig, target, n_layers, cfgk, draft_bin, dvocab, device, dtype, args.topk)
    gp = _build(opti, target, n_layers, cfgk, draft_bin, dvocab, device, dtype, args.topk)
    with torch.inference_mode():
        go.generate(ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)
        gp.generate(ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)

    ro = _timed(go, ids, args.max_new_tokens, eos, args.reps)
    rp = _timed(gp, ids, args.max_new_tokens, eos, args.reps)
    same = bool(torch.equal(ro[5], rp[5])) if ro[5].shape == rp[5].shape else False
    print("\n==================== 干净 A/B（无 profiler，取最优 rep）====================")
    print(f"cfg=topk{args.topk}/steps{args.num_steps}/nodes{args.max_tree_nodes}  逐位一致={same}")
    print(f"{'impl':<10}{'tok/s':>10}{'new':>6}{'wall_s':>9}{'rounds':>8}{'acc/rnd':>9}")
    print(f"{'ORIG':<10}{ro[0]:>10.2f}{ro[1]:>6}{ro[2]:>9.3f}{ro[3]:>8}{ro[4]:>9.2f}")
    print(f"{'OPTI':<10}{rp[0]:>10.2f}{rp[1]:>6}{rp[2]:>9.3f}{rp[3]:>8}{rp[4]:>9.2f}")
    print(f"加速: {rp[0] / ro[0]:.2f}x  ({100 * (rp[0] - ro[0]) / ro[0]:+.1f}%)")


if __name__ == "__main__":
    main()
