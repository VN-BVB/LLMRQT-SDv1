# SPDX-License-Identifier: Apache-2.0
"""Sanity checks for tree_verify_full (no full model download)."""

import torch

from spec_decoding.eagle2.tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_allow_mask_4d,
    layout_tree_position_ids,
    node_depth_from_root,
)
def test_tree_allow_mask_blocks_sibling():
    layout = TreeVerifyLayout(
        token_ids=[10, 20, 30],
        parent=[-1, 0, 0],
        dfs_order=[0, 1, 2],
    )
    m = build_tree_allow_mask_4d(layout, device=torch.device("cpu"))
    assert m.shape == (1, 1, 3, 3)
    assert m[0, 0, 1, 2].item() == 0.0
    assert m[0, 0, 2, 1].item() == 0.0
    assert m[0, 0, 1, 1].item() == 1.0


def test_position_ids_offset_past_len():
    layout = TreeVerifyLayout(
        token_ids=[1, 2, 3],
        parent=[-1, 0, 1],
        dfs_order=[0, 1, 2],
    )
    pos = layout_tree_position_ids(layout, past_len=7)
    assert pos.tolist() == [7, 8, 9]
    assert node_depth_from_root(layout.parent, 2) == 2
