#!/usr/bin/env python3
"""校验向量化后的树掩码与原始双重循环逐元素等价（随机树多次）， 验证build tree mask那里的修改的正确性。"""
import random

import torch

from spec_decoding.eagle3_sgl_profile_opti.tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
    build_tree_sdpa_attn_bias,
    build_tree_allow_mask_4d,
    _on_tree_path,
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ref_cross(layout, past, n):
    neg = torch.finfo(torch.float32).min
    b = torch.full((n, past + n), neg)
    for i in range(n):
        for j in range(past + n):
            if j < past or _on_tree_path(layout.parent, j - past, i):
                b[i, j] = 0.0
    return b


def rand_tree(n):
    parent = [-1]
    for i in range(1, n):
        parent.append(random.randint(0, i - 1))
    return TreeVerifyLayout(token_ids=list(range(n)), parent=parent, dfs_order=list(range(n)))


ok = True
for trial in range(200):
    n = random.randint(1, 40)
    past = random.randint(0, 50)
    lay = rand_tree(n)
    got = build_tree_cross_attn_bias_with_prefix(lay, past, device=dev, dtype=torch.float32).cpu()
    exp = ref_cross(lay, past, n)
    if not torch.equal((got == 0), (exp == 0)):
        ok = False
        print(f"MISMATCH cross trial={trial} n={n} past={past}")
        break
    sd = build_tree_sdpa_attn_bias(lay, device=dev, dtype=torch.float32).cpu()
    a4 = build_tree_allow_mask_4d(lay, device=dev, dtype=torch.float32).cpu()
    for i in range(n):
        for j in range(n):
            on = _on_tree_path(lay.parent, j, i)
            if (sd[i, j].item() == 0.0) != on or (a4[0, 0, i, j].item() == 1.0) != on:
                ok = False
                print(f"MISMATCH sdpa/4d trial={trial} i={i} j={j}")
                break
        if not ok:
            break
    if not ok:
        break

print("ALL EQUAL ✓" if ok else "FAILED ✗")
