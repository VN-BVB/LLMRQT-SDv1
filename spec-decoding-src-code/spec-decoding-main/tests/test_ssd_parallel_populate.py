# SPDX-License-Identifier: Apache-2.0
"""Tests for build_populate_branches (Phase 4 target-side fork)."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from spec_decoding.ssd_parallel import Eagle3Config, build_populate_branches
from spec_decoding.ssd_parallel.types import CacheKey


class TestBuildPopulateBranches(unittest.TestCase):
    def test_branch_count_matches_mq_len(self):
        cfg = Eagle3Config(speculate_k=2, async_fan_out=2, enable_spec_cache=True)
        prefix = torch.tensor([[1, 2, 3]])
        spec = torch.tensor([[3, 10, 11]])  # recovery + K=2 drafts
        logits = torch.randn(1, 3, 32)
        cache_hits = torch.tensor([0])
        fake_acts = torch.randn(1, 48)

        with mock.patch(
            "spec_decoding.ssd_parallel.speculate_ssd.forward_target_eagle_acts",
            return_value=(fake_acts, None, None),
        ):
            branches = build_populate_branches(
                cfg,
                target=object(),
                seq_id=0,
                prefix_at_round=prefix,
                spec=spec,
                logits_p_kp1=logits,
                cache_hits=cache_hits,
                eagle_layers=[1, 2, 3],
            )

        self.assertEqual(len(branches), cfg.mq_len)
        for key, acts in branches:
            self.assertIsInstance(key, CacheKey)
            self.assertTrue(torch.equal(acts, fake_acts))


if __name__ == "__main__":
    unittest.main()
