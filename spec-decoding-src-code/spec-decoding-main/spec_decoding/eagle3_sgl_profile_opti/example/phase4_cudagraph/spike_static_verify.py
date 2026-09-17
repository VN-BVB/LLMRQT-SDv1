#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step5 关键性 de-risk spike：

验证StaticCache + cache_position 原地写 + padded 4D 树 mask + 普通 SDPA一次树前向得到的
node_logits，与现有 DynamicCache extend verify（full_tree_verify_extend 内部）逐元素一致。
若一致，则 static-KV 路径成立，再上 CUDA graph。
"""
from __future__ import annotations

import json

import torch

from huggingface_hub import hf_hub_download

import spec_decoding.eagle3_sgl_profile_opti as pkg
from spec_decoding.eagle3_sgl_profile_opti.tree_verify_mask import (
    _ancestor_self_matrix,
    layout_tree_position_ids,
)
from spec_decoding.eagle3_sgl_profile_opti.tree_draft import tree_draft_to_verify_layout
from spec_decoding.eagle3_sgl_profile_opti.runner import _leaf_indices


def _load(dev, dt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for mid in ["meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            return AutoTokenizer.from_pretrained(mid), AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).to(dev).eval()
        except Exception as e:
            print("load fail", mid, e)
    raise RuntimeError("no model")


def build_padded_tree_mask(layout, cur_len, max_len, device, dtype):
    """[1,1,L,max_len] additive bias：前缀[0:cur_len]=0，树段[cur_len:cur_len+L]按祖先矩阵，其余=-inf。"""
    L = len(layout)
    neg = torch.finfo(dtype).min
    A = _ancestor_self_matrix(layout.parent, L, device)  # [L,L] bool
    allow = torch.zeros((L, max_len), dtype=torch.bool, device=device)
    allow[:, :cur_len] = True
    allow[:, cur_len:cur_len + L] = A
    bias = torch.full((1, 1, L, max_len), neg, dtype=dtype, device=device)
    bias[0, 0][allow] = 0.0
    return bias


def main():
    dev, dt = "cuda", torch.float16
    tok, target = _load(dev, dt)
    nL = int(target.config.num_hidden_layers)
    dvocab = int(json.load(open(hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "config.json"))).get("draft_vocab_size", 32000))
    dbin = hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "pytorch_model.bin")
    cfg = pkg.Eagle3SglConfig(topk=8, num_steps=8, max_tree_nodes=64, verify_mode="full_model_tree", verify_attn_backend="auto")
    el = cfg.resolve_eagle_layers(nL)
    draft = pkg.build_eagle3_sgl_draft(target, el, num_layers=1, draft_vocab_size=dvocab, draft_weights_path=dbin, device=dev, dtype=dt)
    gen = pkg.Eagle3SglGenerator(target, cfg, pkg.make_eagle3_sgl_draft_topk_fn(draft, topk=8), draft_model=draft, autoregressive_draft=True)

    out = tok.apply_chat_template([{"role": "user", "content": "Explain what speculative decoding is, in two sentences."}], add_generation_prompt=True, return_tensors="pt")
    ids = (out.input_ids if hasattr(out, "input_ids") else out).to(dev)

    with torch.inference_mode():
        # prefill -> DynamicCache past
        o0 = target(ids, use_cache=True)
        past_dyn = o0.past_key_values
        recovery = o0.logits[0, -1]
        acts_all, _, _ = pkg.hidden.forward_target_eagle_acts(target, ids, el, use_cache=False, last_token_only=False)
        d_logits, d_hidden, d_kv = draft.prefill(ids, acts_all)
        from spec_decoding.eagle3_sgl_profile_opti.tree_draft import expand_draft_tree_ar
        draft_res = expand_draft_tree_ar(cfg, draft, d_logits, d_hidden, d_kv, int(ids.shape[1]) - 1, device=dev)
        layout = tree_draft_to_verify_layout(draft_res)
        L = len(layout)
        cur_len = int(ids.shape[1])
        print(f"prompt_len={cur_len} L={L}")

        # ---- 参考路径：DynamicCache extend（与 full_tree_verify_extend 内部一致）----
        from spec_decoding.eagle3_sgl_profile_opti.tree_verify_full import _extend_then_crop
        from spec_decoding.eagle3_sgl_profile_opti.tree_verify_mask import build_tree_cross_attn_bias_with_prefix
        tree_tokens = torch.tensor([layout.token_ids], device=dev, dtype=torch.long)
        cross = build_tree_cross_attn_bias_with_prefix(layout, cur_len, device=dev, dtype=dt).view(1, 1, L, cur_len + L)
        pos = layout_tree_position_ids(layout, cur_len).unsqueeze(0).to(dev)
        ref_logits = _extend_then_crop(target, tree_tokens, past_dyn, position_ids=pos, attention_mask=cross)  # [L,V]

        # ---- 新路径：StaticCache + cache_position + padded mask + SDPA ----
        from transformers import StaticCache
        max_len = cur_len + 256
        try:
            sc = StaticCache(config=target.config, max_batch_size=1, max_cache_len=max_len, device=dev, dtype=dt)
        except TypeError:
            sc = StaticCache(target.config, 1, max_len, dev, dt)
        # prefill 进 static cache
        target(ids, past_key_values=sc, cache_position=torch.arange(cur_len, device=dev), use_cache=True)
        mask = build_padded_tree_mask(layout, cur_len, max_len, dev, dt)
        cache_pos = torch.arange(cur_len, cur_len + L, device=dev)
        out_s = target(tree_tokens, past_key_values=sc, position_ids=pos, cache_position=cache_pos,
                       attention_mask=mask, use_cache=True)
        new_logits = out_s.logits[0]  # [L,V]

    same_argmax = torch.equal(ref_logits.argmax(-1), new_logits.argmax(-1))
    max_abs = (ref_logits.float() - new_logits.float()).abs().max().item()
    print(f"argmax 完全一致 = {same_argmax}   max|Δlogits| = {max_abs:.4g}")
    # 接受决策一致性（真正影响输出）
    leaves = _leaf_indices(draft_res.parent)
    from spec_decoding.eagle3_sgl_profile_opti.tree_verify_full import _select_append_root_child
    a_ref = _select_append_root_child(ref_logits, recovery, draft_res.parent, draft_res.token_ids, leaves)[0]
    a_new = _select_append_root_child(new_logits, recovery, draft_res.parent, draft_res.token_ids, leaves)[0]
    print(f"append 列表一致 = {a_ref == a_new}\n  ref={a_ref}\n  new={a_new}")


if __name__ == "__main__":
    main()
