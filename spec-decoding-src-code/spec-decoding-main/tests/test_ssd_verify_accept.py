# SPDX-License-Identifier: Apache-2.0
"""CPU tests for eagle3.verify_accept."""

from __future__ import annotations

import unittest

import torch

from spec_decoding.ssd.verify_accept import (
    append_tokens_for_sequence,
    verify_greedy_accept,
)


class TestVerifyGreedyAccept(unittest.TestCase):
    def test_full_accept_bonus(self):
        K = 3
        V = 128
        logits_p = torch.zeros(1, K, V)
        spec = torch.tensor([[10, 1, 2, 3]])
        logits_p[0, 0, 1] = 100.0
        logits_p[0, 1, 2] = 100.0
        logits_p[0, 2, 3] = 100.0

        accepted, rec = verify_greedy_accept(logits_p, spec)
        self.assertEqual(accepted[0], [10, 1, 2, 3])
        # Bonus = argmax at last verify position (same as eagle_speculative).
        self.assertEqual(rec[0], 3)

        append = append_tokens_for_sequence(accepted[0], rec[0])
        self.assertEqual(append, [1, 2, 3, 3])

    def test_partial_accept(self):
        K = 3
        V = 8
        logits_p = torch.zeros(1, K, V)
        spec = torch.tensor([[5, 1, 2, 9]])
        logits_p[0, 0, 1] = 10.0
        logits_p[0, 1, 2] = 10.0
        logits_p[0, 2, 7] = 10.0

        accepted, rec = verify_greedy_accept(logits_p, spec)
        self.assertEqual(accepted[0], [5, 1, 2])
        self.assertEqual(rec[0], 7)

        append = append_tokens_for_sequence(accepted[0], rec[0])
        self.assertEqual(append, [1, 2, 7])


if __name__ == "__main__":
    unittest.main()
