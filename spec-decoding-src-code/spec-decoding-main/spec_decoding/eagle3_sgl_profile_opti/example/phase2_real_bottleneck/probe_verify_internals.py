#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""定位 verify 阶段 ~76ms/轮 里的开销构成（裸 64-token 前向只 ~17ms）。

用同步式 CUDA event 计时包裹 verify 内部各子步得到它们的耗时：position_ids / triton metadata /
attention-context  / _extend_then_crop(真正前向) / accept 选择。无 profiler 干扰。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import torch


class T:
    def __init__(self):
        self.ms = defaultdict(float)
        self.n = defaultdict(int)

    def wrap(self, name, fn):
        def inner(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                torch.cuda.synchronize()
                self.ms[name] += (time.perf_counter() - t0) * 1e3
                self.n[name] += 1
        return inner


def _load_target(tid, dev, dt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for mid in [tid, "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            return AutoTokenizer.from_pretrained(mid), AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).to(dev).eval(), mid
        except Exception as e:
            print(f"[probe] failed {mid}: {e}")
    raise RuntimeError("no target")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--max-tree-nodes", type=int, default=64)
    args = p.parse_args()
    dev, dt = "cuda", torch.float16
    from huggingface_hub import hf_hub_download
    import spec_decoding.eagle3_sgl_profile_opti as pkg
    import spec_decoding.eagle3_sgl_profile_opti.tree_verify_full as TVF
    import spec_decoding.eagle3_sgl_profile_opti.runner as R

    tok, target, tid = _load_target("meta-llama/Llama-3.1-8B-Instruct", dev, dt)
    nL = int(target.config.num_hidden_layers)
    dvocab = int(json.load(open(hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "config.json"))).get("draft_vocab_size", 32000))
    dbin = hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "pytorch_model.bin")
    cfg = pkg.Eagle3SglConfig(topk=args.topk, num_steps=args.num_steps, max_tree_nodes=args.max_tree_nodes,
                              verify_mode="full_model_tree", verify_attn_backend="auto")
    el = cfg.resolve_eagle_layers(nL)
    draft = pkg.build_eagle3_sgl_draft(target, el, num_layers=1, draft_vocab_size=dvocab, draft_weights_path=dbin, device=dev, dtype=dt)
    gen = pkg.Eagle3SglGenerator(target, cfg, pkg.make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk), draft_model=draft, autoregressive_draft=True)

    out = tok.apply_chat_template([{"role": "user", "content": "Explain what speculative decoding is, in two sentences."}], add_generation_prompt=True, return_tensors="pt")
    ids = (out.input_ids if hasattr(out, "input_ids") else out).to(dev)
    eos = tok.eos_token_id
    with torch.inference_mode():
        gen.generate(ids, max_new_tokens=args.warmup_tokens, eos_token_id=eos)

    t = T()
    # 顶层 verify（runner 引用的名字）
    R.full_tree_verify_triton = t.wrap("verify_total", R.full_tree_verify_triton)
    R.full_tree_verify_extend = t.wrap("verify_total", R.full_tree_verify_extend)
    # verify 内部子步（这些名字是 TVF 模块全局，被 full_tree_verify_* 在调用时查找）
    TVF.layout_tree_position_ids = t.wrap("pos_ids", TVF.layout_tree_position_ids)
    TVF.build_triton_tree_verify_metadata = t.wrap("triton_meta", TVF.build_triton_tree_verify_metadata)
    TVF._extend_then_crop = t.wrap("extend+fwd+crop", TVF._extend_then_crop)
    TVF._select_append_root_child = t.wrap("accept_select", TVF._select_append_root_child)
    _ctx = TVF.tree_verify_triton_attention_context
    TVF.tree_verify_triton_attention_context = t.wrap("attn_ctx_install", _ctx)
    # 建树
    import spec_decoding.eagle3_sgl_profile_opti.tree_draft as TD
    R._build_tree_timed = None

    gen.n_rounds = 0; gen.n_accepted_tokens = 0
    torch.cuda.synchronize(); s = time.perf_counter()
    with torch.inference_mode():
        o = gen.generate(ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
    torch.cuda.synchronize(); wall = (time.perf_counter() - s) * 1e3
    rounds = gen.n_rounds
    print(f"\n=== verify 内部拆解（{rounds} 轮，wall={wall:.0f}ms，{int(o.shape[1]-ids.shape[1])} tok）===")
    order = ["verify_total", "pos_ids", "triton_meta", "attn_ctx_install", "extend+fwd+crop", "accept_select"]
    print(f"{'sub-step':<22}{'calls':>7}{'tot_ms':>10}{'ms/round':>10}")
    for k in order:
        print(f"{k:<22}{t.n[k]:>7}{t.ms[k]:>10.1f}{t.ms[k]/max(1,rounds):>10.2f}")
    inner = sum(t.ms[k] for k in order if k != "verify_total")
    print(f"{'(residual: tensor/python)':<22}{'':>7}{t.ms['verify_total']-inner:>10.1f}{(t.ms['verify_total']-inner)/max(1,rounds):>10.2f}")


if __name__ == "__main__":
    main()
