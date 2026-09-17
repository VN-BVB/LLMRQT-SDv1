#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CPU smoke for EAGLE1: lightweight draft (reuse target embedding/lm_head + 1 decoder
layer, skip layer-0 input_layernorm) + static top-k tree + tree verify.

Draft weights are random here, so accept rate is low — but greedy tree verify keeps
the output lossless vs. plain greedy. Production needs a trained EAGLE1 draft.

    PYTHONPATH=. python3 -m spec_decoding.eagle.example.run_smoke_eagle1 --device cpu
"""

from __future__ import annotations

import argparse

import torch
from transformers import GPT2LMHeadModel

from spec_decoding.eagle import (
    Eagle1Config,
    Eagle1Generator,
    build_eagle1_draft,
    make_eagle1_draft_topk_fn,
)


def main() -> None:
    """加载 tiny 模型，构造轻量 EAGLE1 draft，跑几步生成做 CPU 冒烟。"""
    p = argparse.ArgumentParser(description="EAGLE1 tiny smoke")
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[eagle1-smoke] load {args.model}")
    target = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()

    # lightweight EAGLE1 draft reusing target embedding / lm_head
    draft = build_eagle1_draft(target, num_layers=args.num_layers, device=device)
    draft_topk_fn = make_eagle1_draft_topk_fn(draft, topk=args.topk)

    cfg = Eagle1Config(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=8,
        verify_mode="reference_paths",  # CPU-friendly; full_model_tree needs the mask path
    )
    gen = Eagle1Generator(target, cfg, draft_topk_fn)

    input_ids = torch.tensor([[1, 2, 3]], device=device)
    out = gen.generate(input_ids, max_new_tokens=args.max_new_tokens)
    print(f"[eagle1-smoke] in={input_ids.shape[1]} out={out.shape[1]} ids={out[0].tolist()}")
    print("[eagle1-smoke] OK")


if __name__ == "__main__":
    main()
