#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CPU smoke for EAGLE3-SGL: EAGLE3 多层 act 特征 + EAGLE1 静态 top-k 树 + 树掩码 verify。

draft 权重随机（接受率低），但流程可跑通；target verify 决定最终 token。

    PYTHONPATH=. python3 -m spec_decoding.eagle3_sgl.example.run_smoke_eagle3_sgl --device cpu
"""

from __future__ import annotations

import argparse

import torch
from transformers import GPT2LMHeadModel

from spec_decoding.eagle3_sgl import (
    Eagle3SglConfig,
    Eagle3SglGenerator,
    build_eagle3_sgl_draft,
    make_eagle3_sgl_draft_topk_fn,
)


def main() -> None:
    p = argparse.ArgumentParser(description="EAGLE3-SGL tiny smoke")
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[eagle3_sgl-smoke] load {args.model}")
    target = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()

    n_layers = int(getattr(target.config, "num_hidden_layers", getattr(target.config, "n_layer", 2)))
    eagle_layers = [0, 1] if n_layers <= 2 else None  # tiny 模型层数少，显式指定

    cfg = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=8,
        eagle_layers=eagle_layers,
        verify_mode="reference_paths",  # CPU 友好（full_model_tree 需 decoder mask patch）
    )
    layers = cfg.resolve_eagle_layers(n_layers)

    # 轻量 EAGLE3 draft（多层 act 条件），复用 target embedding/lm_head
    draft = build_eagle3_sgl_draft(target, layers, num_layers=args.num_layers, device=device)
    draft_topk_fn = make_eagle3_sgl_draft_topk_fn(draft, topk=args.topk)

    gen = Eagle3SglGenerator(
        target, cfg, draft_topk_fn, draft_model=draft, autoregressive_draft=True,
    )
    input_ids = torch.tensor([[1, 2, 3]], device=device)
    out = gen.generate(input_ids, max_new_tokens=args.max_new_tokens)
    print(f"[eagle3_sgl-smoke] eagle_layers={layers} in={input_ids.shape[1]} out={out.shape[1]} ids={out[0].tolist()}")
    print("[eagle3_sgl-smoke] OK")


if __name__ == "__main__":
    main()
