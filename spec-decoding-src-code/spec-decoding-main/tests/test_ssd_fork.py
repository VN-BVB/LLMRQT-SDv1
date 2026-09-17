# SPDX-License-Identifier: Apache-2.0
"""CPU tests for spec_decoding.ssd.fork."""

from __future__ import annotations

import unittest

import torch

from spec_decoding.ssd.config import Eagle3Config
from spec_decoding.ssd.fork import (
    cache_keys_for_populate,
    get_forked_recovery_tokens,
    hypo_prefix_for_fork,
)
from spec_decoding.ssd.types import CacheKey


class TestFork(unittest.TestCase):
    def setUp(self):
        self.cfg = Eagle3Config(speculate_k=2, async_fan_out=2)

    def test_fork_greedy_excludes_returned_tokens(self):
        K = 2
        V = 16
        logits = torch.zeros(1, K + 1, V)
        spec = torch.tensor([[10, 1, 2]])
        logits[0, 0, 5] = 100.0
        logits[0, 0, 1] = 50.0
        logits[0, 1, 6] = 100.0
        logits[0, 1, 2] = 50.0
        logits[0, 2, 7] = 100.0

        rec, keep = get_forked_recovery_tokens(
            self.cfg, logits, spec, torch.tensor([1])
        )
        self.assertEqual(rec.shape, (1, self.cfg.mq_len))
        self.assertNotIn(1, rec[0].tolist())

    def test_hypo_prefix_keep_idx_zero(self):
        prefix = torch.tensor([[100, 10]])
        spec = torch.tensor([[10, 1, 2]])
        hypo = hypo_prefix_for_fork(prefix, spec, 0, 99)
        self.assertEqual(hypo.tolist(), [[100, 99]])

    def test_cache_keys_count(self):
        rec = torch.tensor([5, 6, 7, 8, 9, 10])
        keep = torch.tensor([0, 0, 1, 1, 2, 2])
        keys = cache_keys_for_populate(0, rec, keep)
        self.assertEqual(len(keys), 6)


if __name__ == "__main__":
    unittest.main()
