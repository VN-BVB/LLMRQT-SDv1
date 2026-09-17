# SPDX-License-Identifier: Apache-2.0
"""累计 log 概率 beam 剪枝（mock draft，无需 transformers）。"""

import torch

from spec_decoding.eagle2.config import Eagle2TreeConfig
from spec_decoding.eagle2.tree_draft import expand_draft_tree_topk


def _mock_draft(scores_by_token):
    """token_id -> log p；每次 topk 按分数返回。"""

    def fn(hidden: torch.Tensor, parent_token_id: torch.LongTensor) -> tuple:
        del hidden
        t = int(parent_token_id[0, 0].item())
        table = scores_by_token.get(t, {100: -0.1, 200: -5.0, 300: -5.0})
        items = sorted(table.items(), key=lambda x: x[1], reverse=True)
        ids = torch.tensor([[x[0] for x in items]], dtype=torch.long)
        lps = torch.tensor([[x[1] for x in items]], dtype=torch.float32)
        return ids, lps

    return fn


def test_beam_keeps_higher_cumulative_path():
    # 根 0：好分支 100，差分支 200
    # 100 -> 101 累计仍优；200 -> 201 累计差
    scores = {
        0: {100: -0.1, 200: -4.0},
        100: {101: -0.1},
        200: {201: -0.1},
    }
    cfg = Eagle2TreeConfig(topk=2, num_steps=2, max_tree_nodes=2, tree_expand_mode="cumulative")
    h = torch.zeros(1, 8)
    root = torch.tensor([[0]], dtype=torch.long)
    res = expand_draft_tree_topk(cfg, h, root, _mock_draft(scores))
    assert 100 in res.token_ids
    assert 101 in res.token_ids
    assert 200 not in res.token_ids
    assert 201 not in res.token_ids
