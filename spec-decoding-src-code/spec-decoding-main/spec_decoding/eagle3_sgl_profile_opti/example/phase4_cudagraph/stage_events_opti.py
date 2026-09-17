#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""分阶段 GPU 时间（CUDA events，仅末尾 sync 一次）——对 CUDA graph 路径公平。

sync-式计时（time_stages_opti.py）会在每个阶段前后 cudaSynchronize，破坏 graph 的异步
流水（把 CPU 建 mask 的时间也串行计入）→ 对 graph 路径严重高估。这里改用 event 计时：
每个阶段用一对 cuda event 记录其 kernel 的 GPU 时间，整段只在最后 synchronize 一次。

--cuda-graph 开关切换 eager-opti(Step4) / CUDA-graph(Step5)。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import torch


def _load(target_id, dev, dt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for mid in [target_id, "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            return AutoTokenizer.from_pretrained(mid), AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).to(dev).eval()
        except Exception as e:
            print("load fail", mid, e)
    raise RuntimeError("no model")


class EventTimer:
    """event 计时：每个阶段记录 (start,end) event 对；末尾 sync 一次后累加 GPU 时间。"""

    def __init__(self):
        self.pairs = defaultdict(list)

    def wrap(self, name, fn):
        def inner(*a, **k):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            try:
                return fn(*a, **k)
            finally:
                e.record()
                self.pairs[name].append((s, e))
        return inner

    def summarize(self):
        torch.cuda.synchronize()
        return {n: sum(s.elapsed_time(e) for s, e in ps) for n, ps in self.pairs.items()}, \
               {n: len(ps) for n, ps in self.pairs.items()}


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
    p.add_argument("--cuda-graph", action="store_true")
    args = p.parse_args()
    dev, dt = "cuda", torch.float16

    from huggingface_hub import hf_hub_download
    import spec_decoding.eagle3_sgl_profile_opti as pkg
    import spec_decoding.eagle3_sgl_profile_opti.runner as R
    import spec_decoding.eagle3_sgl_profile_opti.tree_verify_full as TVF

    tok, target = _load(args.target, dev, dt)
    nL = int(target.config.num_hidden_layers)
    dvocab = int(json.load(open(hf_hub_download(args.eagle3_draft, "config.json"))).get("draft_vocab_size", 32000))
    dbin = hf_hub_download(args.eagle3_draft, "pytorch_model.bin")
    cfg = pkg.Eagle3SglConfig(topk=args.topk, num_steps=args.num_steps, max_tree_nodes=args.max_tree_nodes,
                              verify_mode="full_model_tree", verify_attn_backend="auto",
                              decode_cuda_graph=args.cuda_graph)
    el = cfg.resolve_eagle_layers(nL)
    draft = pkg.build_eagle3_sgl_draft(target, el, num_layers=1, draft_vocab_size=dvocab, draft_weights_path=dbin, device=dev, dtype=dt)
    gen = pkg.Eagle3SglGenerator(target, cfg, pkg.make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk), draft_model=draft, autoregressive_draft=True)

    out = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt")
    ids = (out.input_ids if hasattr(out, "input_ids") else out).to(dev)
    eos = tok.eos_token_id
    with torch.inference_mode():
        gen.generate(ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)  # graph 在此捕获
    gen.n_rounds = 0; gen.n_accepted_tokens = 0

    T = EventTimer()
    gen._build_tree = T.wrap("draft_build_tree", gen._build_tree)
    gen.draft_model.extend_tokens = T.wrap("draft_extend", gen.draft_model.extend_tokens)
    gen.draft_model.prefill = T.wrap("draft_prefill_init", gen.draft_model.prefill)

    state = {"in_verify": 0, "first": True}

    def verify_mark(name, fn):
        base = T.wrap(name, fn)

        def inner(*a, **k):
            state["in_verify"] += 1
            try:
                return base(*a, **k)
            finally:
                state["in_verify"] -= 1
        return inner

    R.full_tree_verify_extend = verify_mark("target_verify", TVF.full_tree_verify_extend)
    R.full_tree_verify_triton = verify_mark("target_verify", TVF.full_tree_verify_triton)
    if args.cuda_graph:
        from spec_decoding.eagle3_sgl_profile_opti.target_graph import StaticGraphVerifier
        StaticGraphVerifier.verify = verify_mark("target_verify", StaticGraphVerifier.verify)

    _orig_tf = gen.target_m.forward

    def tf_wrap(*a, **k):
        if state["first"]:
            state["first"] = False
            nm = "target_prefill"
        elif state["in_verify"] > 0:
            return _orig_tf(*a, **k)  # eager verify 内部前向，已计入 verify
        else:
            nm = "target_replay"
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            return _orig_tf(*a, **k)
        finally:
            e.record(); T.pairs[nm].append((s, e))

    gen.target_m.forward = tf_wrap

    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        oid = gen.generate(ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    n_new = int(oid.shape[1] - ids.shape[1]); rounds = max(1, gen.n_rounds)

    totals, calls = T.summarize()
    order = ["target_prefill", "draft_prefill_init", "draft_build_tree", "target_verify", "target_replay", "draft_extend"]
    gpu_sum = sum(totals.get(k, 0.0) for k in order)
    mode = "CUDA-graph (Step5)" if args.cuda_graph else "eager-opti (Step4)"
    print(f"\n==================== 分阶段 GPU 时间（event 计时）— {mode} ====================")
    print(f"new_tokens={n_new} rounds={gen.n_rounds} accept/round={gen.n_accepted_tokens / rounds:.2f} "
          f"wall={wall:.2f}s spec_tok/s={n_new / wall:.2f} ms/round(wall)={1e3 * wall / rounds:.1f}")
    print(f"{'stage':<22}{'calls':>7}{'gpu_ms':>10}{'%gpu':>8}{'ms/round':>10}")
    for k in order:
        g = totals.get(k, 0.0)
        print(f"{k:<22}{calls.get(k, 0):>7}{g:>10.1f}{100 * g / max(gpu_sum, 1e-9):>7.1f}%{g / rounds:>10.2f}")
    print(f"{'(stage GPU sum)':<22}{'':>7}{gpu_sum:>10.1f}{100.0:>7.1f}%{gpu_sum / rounds:>10.2f}")
    print(f"{'(wall - GPU sum)':<22}{'':>7}{1e3 * wall - gpu_sum:>10.1f}{'':>8}{(1e3 * wall - gpu_sum) / rounds:>10.2f}  # GPU 空闲/CPU 启动间隙")


if __name__ == "__main__":
    main()
