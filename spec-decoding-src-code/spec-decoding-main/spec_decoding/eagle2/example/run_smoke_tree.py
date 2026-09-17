#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CPU smoke demo for spec_decoding.eagle2 (no HF download).

Expands a mock top-k draft tree, builds verify layout + SDPA bias, and prints
leaf count / bias shape. For full Llama+EAGLE2 HF integration use
Eagle2Generator with real models.
"""

from __future__ import annotations

import argparse

import torch

from spec_decoding.eagle2 import (
    Eagle2TreeConfig,
    TreeVerifyLayout,
    build_tree_sdpa_attn_bias,
    expand_draft_tree_topk,
    tree_draft_to_verify_layout,
)


def _mock_draft(scores_by_token: dict[int, dict[int, float]]):
    def fn(hidden: torch.Tensor, parent_token_id: torch.LongTensor) -> tuple:
        del hidden
        t = int(parent_token_id[0, 0].item())
        table = scores_by_token.get(t, {100: -0.1, 200: -5.0, 300: -5.0})
        items = sorted(table.items(), key=lambda x: x[1], reverse=True)
        ids = torch.tensor([[x[0] for x in items]], dtype=torch.long)
        lps = torch.tensor([[x[1] for x in items]], dtype=torch.float32)
        return ids, lps

    return fn


def main() -> None:
    p = argparse.ArgumentParser(description="EAGLE2 tree draft + verify mask smoke")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--steps", type=int, default=2)
    args = p.parse_args()

    scores = {
        0: {100: -0.1, 200: -4.0},
        100: {101: -0.1},
        200: {201: -0.1},
    }
    cfg = Eagle2TreeConfig(
        topk=args.topk,
        num_steps=args.steps,
        max_tree_nodes=8,
        tree_expand_mode="cumulative",
    )
    h = torch.zeros(1, 8)
    root = torch.tensor([[0]], dtype=torch.long)
    res = expand_draft_tree_topk(cfg, h, root, _mock_draft(scores))
    layout: TreeVerifyLayout = tree_draft_to_verify_layout(res)
    bias = build_tree_sdpa_attn_bias(layout, device=torch.device("cpu"), dtype=torch.float32)

    print(f"[eagle2-smoke] tree nodes={len(layout.token_ids)} tokens={layout.token_ids}")
    print(f"[eagle2-smoke] verify bias shape={tuple(bias.shape)}")
    print("[eagle2-smoke] OK")


if __name__ == "__main__":
    main()
