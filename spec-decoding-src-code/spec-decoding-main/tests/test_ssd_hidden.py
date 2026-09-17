# SPDX-License-Identifier: Apache-2.0
"""CPU tests for eagle3.hidden (no HF model)."""

from __future__ import annotations

import unittest

import torch

from spec_decoding.ssd.hidden import select_eagle_acts_from_hidden_states
from spec_decoding.ssd.config import default_eagle_layers


class TestSelectEagleActs(unittest.TestCase):
    def test_concat_last_token_three_layers(self):
        B, S, H = 2, 5, 4
        hs = tuple(torch.randn(B, S, H) for _ in range(5))
        layers = [0, 2, 3]
        out = select_eagle_acts_from_hidden_states(hs, layers)
        self.assertEqual(out.shape, (B, len(layers) * H))
        expected = torch.cat([hs[i + 1][:, -1, :] for i in layers], dim=-1)
        self.assertTrue(torch.allclose(out, expected))

    def test_default_layers_count(self):
        layers = default_eagle_layers(32)
        self.assertEqual(layers, [2, 16, 29])


if __name__ == "__main__":
    unittest.main()
