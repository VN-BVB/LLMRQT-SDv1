# SPDX-License-Identifier: Apache-2.0
"""EAGLE1 static top-k tree expansion tests (CPU, mock draft fn, no model)."""

from __future__ import annotations

import unittest

import torch

from spec_decoding.eagle.config import Eagle1Config
from spec_decoding.eagle.tree_draft import TreeDraftResult, expand_draft_tree


def _mock_topk(scores_by_token):
    def fn(hidden, parent_token_id):
        t = int(parent_token_id[0, 0].item())
        table = scores_by_token.get(t, {100: -0.1, 200: -1.0, 300: -2.0})
        items = sorted(table.items(), key=lambda x: x[1], reverse=True)
        ids = torch.tensor([[i for i, _ in items]], dtype=torch.long)
        sc = torch.tensor([[s for _, s in items]], dtype=torch.float32)
        return ids, sc

    return fn


class TestEagle1TreeDraft(unittest.TestCase):
    def test_static_topk_tree_shape(self) -> None:
        cfg = Eagle1Config(topk=2, num_steps=2, max_tree_nodes=16)
        h = torch.zeros(1, 8)
        root = torch.tensor([[0]], dtype=torch.long)
        scores = {
            0: {100: -0.1, 200: -1.0},
            100: {101: -0.1, 102: -1.0},
            200: {201: -0.1, 202: -1.0},
        }
        res = expand_draft_tree(cfg, h, root, _mock_topk(scores))
        self.assertIsInstance(res, TreeDraftResult)
        # root layer: 2 nodes; each expands top-2 -> +4; total 6 (depth<=2)
        self.assertEqual(len(res.token_ids), 6)
        self.assertEqual(res.parent[:2], [-1, -1])
        self.assertTrue(all(d in (1, 2) for d in res.node_depth))

    def test_no_cumulative_scores(self) -> None:
        # EAGLE1 result carries no cumulative log-prob field.
        cfg = Eagle1Config(topk=2, num_steps=1, max_tree_nodes=8)
        res = expand_draft_tree(cfg, torch.zeros(1, 8), torch.tensor([[0]]), _mock_topk({}))
        self.assertFalse(hasattr(res, "cum_log_probs"))

    def test_max_tree_nodes_cap(self) -> None:
        cfg = Eagle1Config(topk=3, num_steps=4, max_tree_nodes=4)
        res = expand_draft_tree(cfg, torch.zeros(1, 8), torch.tensor([[0]]), _mock_topk({}))
        self.assertLessEqual(len(res.token_ids), 4)


if __name__ == "__main__":
    unittest.main()
