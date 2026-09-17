# SPDX-License-Identifier: Apache-2.0
"""Metadata / backend resolution for tree verify attention kernels."""

import torch

from spec_decoding.eagle2.tree_verify_attn import (
    build_triton_tree_verify_metadata,
    resolve_verify_attn_backend,
    tree_custom_mask_bool_flat,
)
from spec_decoding.eagle2.tree_verify_mask import TreeVerifyLayout


def test_triton_custom_mask_shape_batch1():
    layout = TreeVerifyLayout(
        token_ids=[10, 20, 30],
        parent=[-1, 0, 0],
        dfs_order=[0, 1, 2],
    )
    past_len = 5
    l_n = 3
    flat = tree_custom_mask_bool_flat(
        layout, past_len, device=torch.device("cpu")
    )
    assert flat.dtype == torch.bool
    assert flat.numel() == l_n * (past_len + l_n)
    meta = build_triton_tree_verify_metadata(
        layout, past_len, device=torch.device("cpu")
    )
    assert meta.max_len_extend == l_n
    assert meta.kv_indptr.tolist() == [0, past_len + l_n]
    assert meta.mask_indptr.tolist() == [0, l_n * (past_len + l_n)]


def test_resolve_auto_topk1_without_flash_falls_back_eager():
    b = resolve_verify_attn_backend("auto", topk=1, layout_len=3)
    assert b in ("flash_attn", "eager")


def test_resolve_auto_topk_gt1_without_triton_falls_back_eager():
    b = resolve_verify_attn_backend("auto", topk=4, layout_len=3)
    assert b in ("triton_tree", "eager")
