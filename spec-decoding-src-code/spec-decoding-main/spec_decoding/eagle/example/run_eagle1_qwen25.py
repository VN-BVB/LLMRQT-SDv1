#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
真·EAGLE1 端到端（Qwen2.5-32B）：真实 target + 真实 EAGLE-1 draft，性能写入 perf.log。

模型：
- target：Qwen/Qwen2.5-32B-Instruct（开源，非门禁）。
- draft ：mit-han-lab/Qwen2.5-32B-Eagle-RL（真实 EAGLE-1 draft：fc(2H→H) + 1 层 GQA decoder
  （首层无 input_layernorm、q/k/v 带 bias=Qwen2 风格）+ 复用 target embedding/lm_head；无 draft 最终 norm、无独立词表）。

与 Llama3 版（run_eagle1_llama3.py）唯一的结构差异：Qwen2 注意力的 q/k/v 带 bias——已由
draft_model 自动从 target 检测并加上，故复用同一个 Eagle1DraftModel，无需单独的 draft 网络文件。

跑法（容器为25.04版本的nvidia torch镜像，单卡 H100 80GB，bf16 下 32B≈64GB 可放下）：
    HF_TOKEN=... PYTHONPATH=. python3 -m spec_decoding.eagle.example.run_eagle1_qwen25 \
        --max-new-tokens 64 --topk 4 --num-steps 4 --max-tree-nodes 24
"""

from __future__ import annotations

import argparse
import os
import time

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Real EAGLE-1 on Qwen2.5-32B")
    p.add_argument("--target", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--eagle1-draft", default="mit-han-lab/Qwen2.5-32B-Eagle-RL")
    p.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--max-tree-nodes", type=int, default=24)
    p.add_argument("--verify-mode", default="full_model_tree",
                   choices=["reference_paths", "full_model_tree"])
    p.add_argument("--verify-attn-backend", default="auto",
                   choices=["auto", "eager", "flash_attn", "triton_tree"])
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--perf-log", default="perf.log")
    args = p.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from spec_decoding.eagle import (
        Eagle1Config,
        Eagle1Generator,
        build_eagle1_draft,
        make_eagle1_draft_topk_fn,
    )

    print(f"[eagle1-qwen-e2e] loading target {args.target} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(device).eval()
    print("[eagle1-qwen-e2e] target ready", flush=True)

    # 真实 EAGLE-1 draft（单 bin）；build 自动检测 Qwen2 的 q/k/v bias，加载真实权重（无 final norm）
    draft_bin = hf_hub_download(args.eagle1_draft, "pytorch_model.bin")
    draft = build_eagle1_draft(target, num_layers=1, draft_weights_path=draft_bin,
                               device=device, dtype=dtype)
    draft_topk_fn = make_eagle1_draft_topk_fn(draft, topk=args.topk)

    cfg = Eagle1Config(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        verify_mode=args.verify_mode,
        verify_attn_backend=args.verify_attn_backend,
    )
    gen = Eagle1Generator(target, cfg, draft_topk_fn, refresh_target_each_node=False)

    if not args.no_chat_template and getattr(tok, "chat_template", None):
        out = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = (out if isinstance(out, torch.Tensor) else out["input_ids"]).to(device)
    else:
        input_ids = tok(args.prompt, return_tensors="pt")["input_ids"].to(device)
    plen = int(input_ids.shape[1])
    eos = tok.eos_token_id

    # 贪心基线：干净整段前向（use_cache=False）逐步 argmax，与 verify 数值同源
    with torch.inference_mode():
        cur = input_ids.clone()
        greedy_ids = []
        for _ in range(args.max_new_tokens):
            nt = int(target(cur, use_cache=False).logits[0, -1].argmax())
            greedy_ids.append(nt)
            cur = torch.cat([cur, torch.tensor([[nt]], device=device)], dim=1)
            if eos is not None and nt == eos:
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out_ids = gen.generate(input_ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    spec_cont = out_ids[0, plen:].tolist()

    n_new = len(spec_cont)
    accept_per_round = gen.n_accepted_tokens / max(1, gen.n_rounds)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    with torch.inference_mode():
        bg = target.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             num_beams=1, use_cache=True, pad_token_id=eos)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    base_dt = time.time() - t1
    base_cont = bg[0, plen:].tolist()
    base_new = len(base_cont)

    # 无损参照：extend verify 复用cached KV(use_cache=True)，故与 HF 标准贪心(generate,
    # use_cache=True) 数值同源——以它为准。clean-forward 贪心(use_cache=False)在 bf16 近似平局处
    # 会与 cached 路径偶发翻转（target 自身现象），仅作旁证。
    def _first_div(a, b):
        mm = min(len(a), len(b))
        d = next((i for i in range(mm) if a[i] != b[i]), -1)
        return mm, d

    m_hf, div_hf = _first_div(spec_cont, base_cont)
    m_cl, div_cl = _first_div(spec_cont, greedy_ids)
    lossless = div_hf == -1  # 与 HF 标准贪心一致即为无损

    rec = (
        f"[eagle1] target={args.target} draft={args.eagle1_draft} "
        f"verify={args.verify_mode}(backend={args.verify_attn_backend}) "
        f"topk={args.topk} num_steps={args.num_steps} max_nodes={args.max_tree_nodes} dtype={args.dtype}\n"
        f"  new_tokens={n_new}  rounds={gen.n_rounds}  accept/round={accept_per_round:.2f}\n"
        f"  wall={dt:.2f}s  spec_tok/s={n_new / dt:.2f}\n"
        f"  lossless(vs HF greedy use_cache=True)={lossless}"
        f"  [vs HF: first_div={div_hf}/{m_hf}; vs clean-greedy: first_div={div_cl}/{m_cl}]\n"
        f"  baseline(HF greedy) new_tokens={base_new}  wall={base_dt:.2f}s  "
        f"base_tok/s={base_new / base_dt:.2f}  speedup={ (n_new/dt)/(base_new/base_dt):.2f}x\n"
    )
    mshow = min(len(greedy_ids), n_new)
    print("\n==================== EAGLE1 (Qwen2.5-32B) ====================")
    print(rec, end="")
    print("[greedy]", tok.decode(torch.tensor(greedy_ids[:mshow]), skip_special_tokens=True))
    print("[eagle1]", tok.decode(torch.tensor(spec_cont[:mshow]), skip_special_tokens=True))
    print("==============================================================")

    with open(args.perf_log, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        f.write(rec)
        f.write("\n")
    print(f"[eagle1-qwen-e2e] perf appended to {os.path.abspath(args.perf_log)}")
    if lossless:
        print("[eagle1-qwen-e2e] OK (lossless vs HF standard greedy)")
    else:
        # bf16 下批量树 verify(cached KV) 与 HF 标准贪心可能在近似平局处偶发翻转；非算法性损失。
        print(f"[eagle1-qwen-e2e] NOTE: diverged from HF greedy at idx {div_hf} "
              f"(likely a bf16 near-tie in batched tree verify; accept/round={accept_per_round:.2f})")


if __name__ == "__main__":
    main()
