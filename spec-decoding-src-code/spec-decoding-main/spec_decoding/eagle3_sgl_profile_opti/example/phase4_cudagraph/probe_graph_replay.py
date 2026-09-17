#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""测verify 前向在 graph replay 下的纯 GPU 时间，对比 eager triton verify。得到cuda graph的收益

捕获后直接 time verifier._graph.replay() ×N（单次 sync），不掺入建树/select 的 CPU 开销。
"""
from __future__ import annotations

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


def main():
    dev, dt = "cuda", torch.float16
    from huggingface_hub import hf_hub_download
    tok, target = _load(dev, dt)
    nL = int(target.config.num_hidden_layers)
    dvocab = int(json.load(open(hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "config.json"))).get("draft_vocab_size", 32000))
    dbin = hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "pytorch_model.bin")
    el = pkg.Eagle3SglConfig().resolve_eagle_layers(nL)
    draft = pkg.build_eagle3_sgl_draft(target, el, num_layers=1, draft_vocab_size=dvocab, draft_weights_path=dbin, device=dev, dtype=dt)
    cfg = pkg.Eagle3SglConfig(topk=8, num_steps=8, max_tree_nodes=64, verify_mode="full_model_tree",
                              verify_attn_backend="auto", decode_cuda_graph=True)
    gen = pkg.Eagle3SglGenerator(target, cfg, pkg.make_eagle3_sgl_draft_topk_fn(draft, topk=8), draft_model=draft, autoregressive_draft=True)
    out = tok.apply_chat_template([{"role": "user", "content": "Explain what speculative decoding is, in two sentences."}], add_generation_prompt=True, return_tensors="pt")
    ids = (out.input_ids if hasattr(out, "input_ids") else out).to(dev)
    with torch.inference_mode():
        gen.generate(ids, max_new_tokens=24, eos_token_id=tok.eos_token_id)  # 捕获 + 填充 KV

    v = gen._sg_state["verifier"]
    N = 100
    # 纯 graph replay GPU 时间
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(N):
        v._graph.replay()
    e.record(); torch.cuda.synchronize()
    print(f"verify graph.replay() 纯 GPU: {s.elapsed_time(e) / N:.3f} ms/round  (L={v._L}, max_len={v.max_len})")

    # 同一前向 eager（不 replay，直接跑 _run_forward）GPU 时间，作对照
    torch.cuda.synchronize(); s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    s2.record()
    with torch.inference_mode():
        for _ in range(N):
            v._run_forward()
    e2.record(); torch.cuda.synchronize()
    print(f"同一 verify 前向 eager（_run_forward）: {s2.elapsed_time(e2) / N:.3f} ms/round")


if __name__ == "__main__":
    main()
