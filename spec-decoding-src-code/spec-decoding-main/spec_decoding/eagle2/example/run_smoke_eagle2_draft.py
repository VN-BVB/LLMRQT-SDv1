#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CPU smoke for EAGLE2 tree speculation using the lightweight EAGLE draft model
(reuse target embedding/lm_head + fc + 1 decoder layer, skip layer-0 input_layernorm).

draft 权重随机，接受率低，但流程可跑通（target verify 决定最终 token）。

    PYTHONPATH=. python3 -m spec_decoding.eagle2.example.run_smoke_eagle2_draft --device cpu
"""

from __future__ import annotations

import argparse

import torch
from transformers import GPT2LMHeadModel

from spec_decoding.eagle2 import (
    Eagle2TreeConfig,
    Eagle2Generator,
    build_eagle2_draft,
    make_eagle2_draft_topk_fn,
)


def main() -> None:
    p = argparse.ArgumentParser(description="EAGLE2 lightweight-draft tiny smoke")
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[eagle2-draft-smoke] load {args.model}")
    target = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()

    # lightweight EAGLE draft reusing target embedding / lm_head
    draft = build_eagle2_draft(target, num_layers=args.num_layers, device=device)
    draft_topk_fn = make_eagle2_draft_topk_fn(draft, topk=args.topk)

    cfg = Eagle2TreeConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=8,
        tree_expand_mode="cumulative",   # EAGLE2 累计概率树
        verify_mode="reference_paths",
    )
    gen = Eagle2Generator(target, cfg, draft_topk_fn)

    input_ids = torch.tensor([[1, 2, 3]], device=device)
    out = gen.generate(input_ids, max_new_tokens=args.max_new_tokens)
    print(f"[eagle2-draft-smoke] in={input_ids.shape[1]} out={out.shape[1]} ids={out[0].tolist()}")
    print("[eagle2-draft-smoke] OK")


if __name__ == "__main__":
    main()
