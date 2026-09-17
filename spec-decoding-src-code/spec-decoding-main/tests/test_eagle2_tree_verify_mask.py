# SPDX-License-Identifier: Apache-2.0
"""
EAGLE2 树 verify 掩码 的最小断言。

完整 EAGLE2 里：同一 verify forward 中多个树节点若用普通 causal mask，兄弟节点
会互 attend，logits 被污染。build_tree_sdpa_attn_bias 应保证「只沿父链」
可见；本测试用一棵根 + 两子节点手工检查兄弟互不可见。
"""

import unittest

import torch

from spec_decoding.eagle2 import TreeVerifyLayout, build_tree_sdpa_attn_bias


def test_tree_bias_allows_ancestors_blocks_siblings():
    # 摊平树：节点 0 为根，1 与 2 为两子节点（与 expand_draft_tree_topk 的 parent 约定一致）
    layout = TreeVerifyLayout(
        token_ids=[10, 20, 30],
        parent=[-1, 0, 0],
        dfs_order=[0, 1, 2],
    )
    bias = build_tree_sdpa_attn_bias(layout, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (3, 3)
    neg = torch.finfo(bias.dtype).min
    # child 1 may attend root and self
    assert torch.isclose(bias[1, 0], torch.tensor(0.0)).item() is True
    assert torch.isclose(bias[1, 1], torch.tensor(0.0)).item() is True
    # child 1 must not attend sibling 2
    assert bias[1, 2].item() == neg
    assert bias[2, 1].item() == neg
    assert torch.isclose(bias[2, 0], torch.tensor(0.0)).item() is True


def test_eagle2_package_exports_make_hidden_fn():
    try:
        from spec_decoding.eagle2 import make_hidden_residual_draft_topk_fn
    except ModuleNotFoundError as e:
        if "transformers" in str(e):
            raise unittest.SkipTest("transformers not installed")
        raise
    assert callable(make_hidden_residual_draft_topk_fn)
