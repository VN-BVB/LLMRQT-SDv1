#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
真·EAGLE2 端到端：真实 target + 真实 EAGLE draft 权重，并把各项性能数据写入 perf.log。

EAGLE2 与 EAGLE1 共用同一份 draft checkpoint（draft 网络结构一致：fc(2H→H) + 1 层 GQA
decoder + 复用 target embedding/lm_head）；EAGLE2 只在 推理期 改进了树展开策略
（累计 log 概率 beam + 全局剪枝，tree_expand_mode=cumulative），通常比 EAGLE1 的静态
top-k 树接受更多 token。

模型选择（与 sglang / vllm 的 EAGLE 配置一致）：
- target：meta-llama/Meta-Llama-3-8B-Instruct（无 HF token 时自动回退等价镜像
  NousResearch/Meta-Llama-3-8B-Instruct）。
- draft ：yuhuili/EAGLE-LLaMA3-Instruct-8B（官方 EAGLE draft 权重，无 draft 最终 norm）。

verify 用复用前缀 KV 的 extend 树掩码 verify（full_model_tree：HF sdpa 或 triton_tree kernel，
parent→child 接受 + recovery_logits root 接受），无损（与干净贪心一致）。draft 不逐节点刷新
target（refresh_target_each_node=False），适合大模型。

跑法：
    PYTHONPATH=. python3 -m spec_decoding.eagle2.example.run_eagle2_llama3 \
        --max-new-tokens 96 --topk 4 --num-steps 5 --max-tree-nodes 32
"""

from __future__ import annotations

import argparse
import os
import time

import torch


def _load_target(target_id: str, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fallback = "NousResearch/Meta-Llama-3-8B-Instruct"
    last_err = None
    for mid in [target_id, fallback]:
        try:
            print(f"[eagle2-e2e] loading target {mid} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            print(f"[eagle2-e2e] target ready: {mid}", flush=True)
            return tok, model, mid
        except Exception as e:  # gated / network
            print(f"[eagle2-e2e] failed to load {mid}: {e}", flush=True)
            last_err = e
    raise RuntimeError(f"could not load target; last error: {last_err}")


def main() -> None:
    p = argparse.ArgumentParser(description="Real EAGLE2 on Llama-3-8B")
    p.add_argument("--target", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--eagle-draft", default="yuhuili/EAGLE-LLaMA3-Instruct-8B")
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--max-tree-nodes", type=int, default=32)
    p.add_argument("--tree-expand-mode", default="cumulative", choices=["cumulative", "bfs"],
                   help="cumulative = EAGLE-2 累计 log 概率 beam 剪枝；bfs = 调试对照")
    p.add_argument("--verify-mode", default="full_model_tree",
                   choices=["reference_paths", "full_model_tree"])
    p.add_argument("--verify-attn-backend", default="auto",
                   choices=["auto", "eager", "flash_attn", "triton_tree"],
                   help="full_model_tree extend verify attention: triton_tree uses SGLang kernel; else sdpa")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--perf-log", default="perf.log")
    args = p.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle2 import (
        Eagle2TreeConfig,
        Eagle2Generator,
        build_eagle2_draft,
        make_eagle2_draft_topk_fn,
    )

    tok, target, target_id = _load_target(args.target, device, dtype)

    # 官方 EAGLE draft 权重（单 bin），build 时加载（use_final_norm 自动按真实权重=False）
    draft_bin = hf_hub_download(args.eagle_draft, "pytorch_model.bin")
    draft = build_eagle2_draft(target, num_layers=1, draft_weights_path=draft_bin,
                               device=device, dtype=dtype)
    draft_topk_fn = make_eagle2_draft_topk_fn(draft, topk=args.topk)

    cfg = Eagle2TreeConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        tree_expand_mode=args.tree_expand_mode,
        verify_mode=args.verify_mode,
        verify_attn_backend=args.verify_attn_backend,
    )
    # 大模型上不逐节点刷新 target（只在根取一次 hidden），避免 O(树节点) 次 target 前向
    gen = Eagle2Generator(target, cfg, draft_topk_fn, refresh_target_each_node=False)

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
    lossless = greedy_cont[:m] == spec_cont[:m]
    accept_per_round = gen.n_accepted_tokens / max(1, gen.n_rounds)

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
        f"[eagle2] target={target_id} draft={args.eagle_draft} "
        f"verify={args.verify_mode}(backend={args.verify_attn_backend}) "
        f"tree={args.tree_expand_mode} topk={args.topk} num_steps={args.num_steps} "
        f"max_nodes={args.max_tree_nodes} dtype={args.dtype}\n"
        f"  new_tokens={n_new}  rounds={gen.n_rounds}  accept/round={accept_per_round:.2f}\n"
        f"  wall={dt:.2f}s  spec_tok/s={n_new / dt:.2f}  lossless={lossless}\n"
        f"  baseline(HF greedy) new_tokens={base_new}  wall={base_dt:.2f}s  "
        f"base_tok/s={base_new / base_dt:.2f}  speedup={ (n_new/dt)/(base_new/base_dt):.2f}x\n"
    )
    print("\n==================== EAGLE2 ====================")
    print(rec, end="")
    print("[greedy]", tok.decode(torch.tensor(greedy_cont[:m]), skip_special_tokens=True))
    print("[eagle2]", tok.decode(torch.tensor(spec_cont[:m]), skip_special_tokens=True))
    print("================================================")

    with open(args.perf_log, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        f.write(rec)
        f.write("\n")
    print(f"[eagle2-e2e] perf appended to {os.path.abspath(args.perf_log)}")
    assert lossless, "EAGLE2 output diverged from clean-greedy baseline"
    print("[eagle2-e2e] OK (lossless)")


if __name__ == "__main__":
    main()
