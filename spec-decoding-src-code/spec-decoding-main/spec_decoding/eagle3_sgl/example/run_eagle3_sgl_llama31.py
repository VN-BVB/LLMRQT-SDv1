#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
真·EAGLE3 端到端：真实 target + 真实 EAGLE3 draft 权重，并把各项性能数据写入 perf.log。

模型来源（与公开 EAGLE3 Llama-3.1 配置一致）：
- target：meta-llama/Llama-3.1-8B-Instruct（无 HF token 时自动回退等价镜像
  NousResearch/Meta-Llama-3.1-8B-Instruct）。
- eagle3 draft：yuhuili/EAGLE3-LLaMA3.1-Instruct-8B（公开权重）。其 checkpoint 含
  fc / midlayer(GQA) / norm / lm_head(draft_vocab=32000) / d2t，由 build_eagle3_sgl_draft
  直接加载（真实权重 → 真实接受率）。

EAGLE3 draft 条件 = target 低/中/高多层 hidden 拼接（eagle_layers，默认 [2, L//2, L-3]）。
verify 用复用前缀 KV 的 extend 树掩码 verify（full_model_tree：HF sdpa 或 triton_tree kernel，
parent→child 接受 + recovery_logits root 接受），无损（与干净贪心一致）。draft 不逐节点刷新
target（refresh_target_each_node=False），适合大模型。

跑法：
    PYTHONPATH=. python3 -m spec_decoding.eagle3_sgl.example.run_eagle3_sgl_llama31 \
        --max-new-tokens 96 --topk 4 --num-steps 5 --max-tree-nodes 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch


def _load_target(
    target_id: str, device, dtype, fallback_target: Optional[str] = None
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    last_err = None
    model_ids = [target_id]
    if fallback_target is not None and fallback_target != target_id:
        model_ids.append(fallback_target)
    for mid in model_ids:
        try:
            print(f"[eagle3-e2e] loading target {mid} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype).to(device).eval()
            # transformers 5.x 使用 dtype；避免已弃用 torch_dtype 警告，不改变权重精度。
            print(f"[eagle3-e2e] target ready: {mid}", flush=True)
            return tok, model, mid
        except Exception as e:  # gated / network
            print(f"[eagle3-e2e] failed to load {mid}: {e}", flush=True)
            last_err = e
    raise RuntimeError(f"could not load target; last error: {last_err}")


def main(
    *,
    default_target: str = "meta-llama/Llama-3.1-8B-Instruct",
    default_draft: str = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    fallback_target: Optional[str] = "NousResearch/Meta-Llama-3.1-8B-Instruct",
    description: str = "Real EAGLE3 on Llama-3.1-8B",
    default_max_new_tokens: int = 96,
    default_topk: int = 8,
    default_num_steps: int = 8,
    default_max_tree_nodes: int = 64,
    default_verify_mode: str = "full_model_tree",
) -> None:
    """Run real-weight EAGLE3 with configurable model-specific defaults."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--target", default=default_target)
    p.add_argument("--eagle3-draft", default=default_draft)
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    p.add_argument("--topk", type=int, default=default_topk)
    p.add_argument("--num-steps", type=int, default=default_num_steps)
    p.add_argument("--max-tree-nodes", type=int, default=default_max_tree_nodes)
    p.add_argument("--tree-expand-mode", default="cumulative", choices=["cumulative", "static"],
                   help="cumulative = EAGLE-2 累计 log 概率 beam 剪枝；static = EAGLE-1 静态 top-k BFS 树（AR draft 下忽略）")
    p.add_argument("--verify-mode", default=default_verify_mode,
                   choices=["reference_paths", "full_model_tree"])
    p.add_argument("--verify-attn-backend", default="auto",
                   choices=["auto", "eager", "flash_attn", "triton_tree"],
                   help="full_model_tree extend verify attention: triton_tree uses Triton kernel; else sdpa")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no-autoregressive-draft", dest="autoregressive_draft", action="store_false",
                   help="禁用 KV-cache 自回归 draft（回退单 token draft_topk_fn）")
    p.set_defaults(autoregressive_draft=True)
    p.add_argument("--perf-log", default="perf.log")
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

    tok, target, target_id = _load_target(
        args.target, device, dtype, fallback_target=fallback_target
    )
    n_layers = int(target.config.num_hidden_layers)

    # eagle3 draft：取 config 的 draft_vocab_size，下载权重 bin，构造并加载真实权重
    draft_cfg_path = hf_hub_download(args.eagle3_draft, "config.json")
    draft_vocab_size = int(json.load(open(draft_cfg_path)).get("draft_vocab_size", 32000))
    draft_bin = hf_hub_download(args.eagle3_draft, "pytorch_model.bin")

    cfg = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        tree_expand_mode=args.tree_expand_mode,
        verify_mode=args.verify_mode,
        verify_attn_backend=args.verify_attn_backend,
    )
    eagle_layers = cfg.resolve_eagle_layers(n_layers)  # 默认 [2, L//2, L-3]
    print(f"[eagle3-e2e] eagle_layers={eagle_layers} draft_vocab_size={draft_vocab_size}", flush=True)

    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        num_layers=1,
        draft_vocab_size=draft_vocab_size,
        draft_weights_path=draft_bin,  # 加载真实 EAGLE3 权重
        device=device,
        dtype=dtype,
    )
    draft_topk_fn = make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk)

    # 自回归 draft（KV + RoPE + feature 递归，对齐官方 EAGLE3 AR 推理）
    gen = Eagle3SglGenerator(
        target,
        cfg,
        draft_topk_fn,
        draft_model=draft,
        autoregressive_draft=args.autoregressive_draft,
    )

    # 输入（instruct 模型套 chat template，否则 OOD）
    if not args.no_chat_template and getattr(tok, "chat_template", None):
        out = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = (out if isinstance(out, torch.Tensor) else out["input_ids"]).to(device)
    else:
        input_ids = tok(args.prompt, return_tensors="pt")["input_ids"].to(device)
    plen = int(input_ids.shape[1])
    eos = tok.eos_token_id

    # 贪心基线：干净整段前向（use_cache=False）逐步 argmax，与 verify 数值同源（避免 fp16 KV 平局翻转误判）
    with torch.inference_mode():
        cur = input_ids.clone()
        greedy_ids = []
        for _ in range(args.max_new_tokens):
            nt = int(target(cur, use_cache=False).logits[0, -1].argmax())
            greedy_ids.append(nt)
            cur = torch.cat([cur, torch.tensor([[nt]], device=device)], dim=1)
            if eos is not None and nt == eos:
                break
    greedy_cont = greedy_ids

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out_ids = gen.generate(input_ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    spec_cont = out_ids[0, plen:].tolist()

    n_new = len(spec_cont)
    m = min(len(greedy_cont), n_new)
    lossless = greedy_cont == spec_cont
    # 必须连长度也完全相等；仅比较公共前缀会把“多生成一个 token”误报成无损。
    accept_length = gen.n_accepted_tokens / max(1, gen.n_rounds)
    draft_accept_rate = gen.n_matched_draft_tokens / max(
        1, gen.n_proposed_draft_tokens
    )

    # 用 baseline 贪心（逐 token）作为基准时间参照
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    with torch.inference_mode():
        bg = target.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             num_beams=1, use_cache=True, pad_token_id=eos)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    base_dt = time.time() - t1
    base_new = int(bg.shape[1] - plen)

    rec = (
        f"[eagle3_sgl] target={target_id} draft={args.eagle3_draft} "
        f"verify={args.verify_mode}(backend={args.verify_attn_backend}) "
        f"tree={args.tree_expand_mode} ar_draft={args.autoregressive_draft} inc_acts=True inc_draft_ext=True eagle_layers={eagle_layers} topk={args.topk} num_steps={args.num_steps} "
        f"max_nodes={args.max_tree_nodes} dtype={args.dtype}\n"
        f"  new_tokens={n_new}  rounds={gen.n_rounds}  accept_length={accept_length:.2f}  "
        f"draft_accept_rate={draft_accept_rate:.3f}\n"
        # accept_length 含每轮必出的 target token；draft_accept_rate 才是真正的草稿命中率。
        f"  wall={dt:.2f}s  spec_tok/s={n_new / dt:.2f}  lossless={lossless}\n"
        f"  baseline(HF greedy) new_tokens={base_new}  wall={base_dt:.2f}s  "
        f"base_tok/s={base_new / base_dt:.2f}  speedup={ (n_new/dt)/(base_new/base_dt):.2f}x\n"
    )
    print("\n==================== EAGLE3 ====================")
    print(rec, end="")
    print("[greedy]", tok.decode(torch.tensor(greedy_cont[:m]), skip_special_tokens=True))
    print("[eagle3_sgl]", tok.decode(torch.tensor(spec_cont[:m]), skip_special_tokens=True))
    print("====================================================")

    with open(args.perf_log, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        f.write(rec)
        f.write("\n")
    print(f"[eagle3-e2e] perf appended to {os.path.abspath(args.perf_log)}")
    assert lossless, "EAGLE3 output diverged from clean-greedy baseline"
    print("[eagle3-e2e] OK (lossless)")


if __name__ == "__main__":
    main()
