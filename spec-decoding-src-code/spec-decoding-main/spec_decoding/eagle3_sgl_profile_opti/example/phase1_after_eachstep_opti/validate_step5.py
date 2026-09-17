#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step5 验证：eager-opti vs CUDA-graph 路径的输出一致性 / 接受率 / 吞吐。

两条路径共享同一 target / draft 权重；唯一区别是 decode_cuda_graph 开关。
预期：输出 token 序列一致（或仅 fp16 末尾微小分叉），acc/round 基本相同，tok/s 提升。
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import spec_decoding.eagle3_sgl_profile_opti as pkg


def _load(dev, dt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for mid in ["meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            return AutoTokenizer.from_pretrained(mid), AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).to(dev).eval()
        except Exception as e:
            print("load fail", mid, e)
    raise RuntimeError("no model")


def _build(target, el, dvocab, dbin, dev, dt, topk, steps, nodes, graph, capture=True):
    cfg = pkg.Eagle3SglConfig(topk=topk, num_steps=steps, max_tree_nodes=nodes,
                              verify_mode="full_model_tree", verify_attn_backend="auto",
                              decode_cuda_graph=graph, graph_capture=capture)
    draft = pkg.build_eagle3_sgl_draft(target, el, num_layers=1, draft_vocab_size=dvocab,
                                       draft_weights_path=dbin, device=dev, dtype=dt)
    return pkg.Eagle3SglGenerator(target, cfg, pkg.make_eagle3_sgl_draft_topk_fn(draft, topk=topk),
                                  draft_model=draft, autoregressive_draft=True)


def _timed(gen, ids, mnt, eos, reps):
    best = None
    for _ in range(reps):
        gen.n_rounds = 0; gen.n_accepted_tokens = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = gen.generate(ids, max_new_tokens=mnt, eos_token_id=eos)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        n = int(out.shape[1] - ids.shape[1])
        cur = (n / dt, n, dt, gen.n_rounds, gen.n_accepted_tokens / max(1, gen.n_rounds), out)
        if best is None or cur[0] > best[0]:
            best = cur
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--max-tree-nodes", type=int, default=64)
    args = p.parse_args()
    dev, dt = "cuda", torch.float16
    from huggingface_hub import hf_hub_download
    tok, target = _load(dev, dt)
    nL = int(target.config.num_hidden_layers)
    dvocab = int(json.load(open(hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "config.json"))).get("draft_vocab_size", 32000))
    dbin = hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "pytorch_model.bin")
    cfg0 = pkg.Eagle3SglConfig()
    el = cfg0.resolve_eagle_layers(nL)

    out = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt")
    ids = (out.input_ids if hasattr(out, "input_ids") else out).to(dev)
    eos = tok.eos_token_id

    g_eager = _build(target, el, dvocab, dbin, dev, dt, args.topk, args.num_steps, args.max_tree_nodes, False)
    g_stat = _build(target, el, dvocab, dbin, dev, dt, args.topk, args.num_steps, args.max_tree_nodes, True, capture=False)
    g_graph = _build(target, el, dvocab, dbin, dev, dt, args.topk, args.num_steps, args.max_tree_nodes, True, capture=True)
    with torch.inference_mode():
        for g in (g_eager, g_stat, g_graph):
            g.generate(ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)

    re = _timed(g_eager, ids, args.max_new_tokens, eos, args.reps)
    rs = _timed(g_stat, ids, args.max_new_tokens, eos, args.reps)
    rg = _timed(g_graph, ids, args.max_new_tokens, eos, args.reps)

    def _cmp(a, b):
        m = min(a.shape[1], b.shape[1])
        diff = (a[0, :m] != b[0, :m])
        fd = int(diff.float().argmax().item()) if bool(diff.any()) else m
        return m, fd

    m_sg, fd_sg = _cmp(rs[5], rg[5])   # static-eager vs graph：应完全一致（graph 无损）
    m_eg, fd_eg = _cmp(re[5], rg[5])   # eager-opti(triton) vs graph(sdpa)：可能 fp16 末尾分叉
    print("\n==================== Step5 验证：eager-opti / static-eager / CUDA-graph ====================")
    print(f"cfg=topk{args.topk}/steps{args.num_steps}/nodes{args.max_tree_nodes}")
    print(f"[graph 无损] static-eager vs CUDA-graph：公共长度 {m_sg}，首个分叉={fd_sg}{'（完全一致✓）' if fd_sg==m_sg else ''}")
    print(f"[triton↔sdpa] eager-opti vs CUDA-graph：公共长度 {m_eg}，首个分叉={fd_eg}{'（完全一致）' if fd_eg==m_eg else '（fp16 分叉，acc/rnd 见下）'}")
    print(f"{'impl':<16}{'tok/s':>10}{'new':>6}{'wall_s':>9}{'rounds':>8}{'acc/rnd':>9}")
    print(f"{'EAGER-opti(tri)':<16}{re[0]:>10.2f}{re[1]:>6}{re[2]:>9.3f}{re[3]:>8}{re[4]:>9.2f}")
    print(f"{'STATIC-eager':<16}{rs[0]:>10.2f}{rs[1]:>6}{rs[2]:>9.3f}{rs[3]:>8}{rs[4]:>9.2f}")
    print(f"{'CUDA-graph':<16}{rg[0]:>10.2f}{rg[1]:>6}{rg[2]:>9.3f}{rg[3]:>8}{rg[4]:>9.2f}")
    print(f"加速 vs eager-opti: {rg[0] / re[0]:.2f}x  ({100 * (rg[0] - re[0]) / re[0]:+.1f}%)")
    print(f"graph vs static-eager: {rg[0] / rs[0]:.2f}x  (纯 CUDA graph 贡献)")


if __name__ == "__main__":
    main()
