# SPDX-License-Identifier: Apache-2.0
"""CPU tests for eagle3.vocab."""

from __future__ import annotations

import unittest

import torch

from spec_decoding.ssd.vocab import (
    DraftVocabMap,
    expand_draft_logits_to_target_vocab,
)


class TestDraftVocabMap(unittest.TestCase):
    def test_expand_scatter(self):
        d2t = torch.tensor([0, 10, 20], dtype=torch.long)
        vm = DraftVocabMap.from_offset_tensor(d2t, target_vocab_size=100)
        draft_logits = torch.tensor([[1.0, 2.0, 3.0]])
        full = expand_draft_logits_to_target_vocab(draft_logits, vm)
        self.assertEqual(full.shape, (1, 100))
        self.assertEqual(full[0, 0].item(), 1.0)
        self.assertEqual(full[0, 11].item(), 2.0)
        self.assertEqual(full[0, 22].item(), 3.0)
        self.assertTrue(torch.isinf(full[0, 1]).item())


if __name__ == "__main__":
    unittest.main()
