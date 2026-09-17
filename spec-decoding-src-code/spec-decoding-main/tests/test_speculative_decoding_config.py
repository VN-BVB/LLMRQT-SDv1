"""Lightweight tests for speculative decode config (no GPU model)."""

import unittest

from spec_decoding.speculative_decoding import SpeculativeConfig


class TestSpeculativeConfig(unittest.TestCase):
    def test_topk_must_be_one(self) -> None:
        with self.assertRaises(ValueError):
            SpeculativeConfig(topk=4)

    def test_default_topk(self) -> None:
        c = SpeculativeConfig()
        self.assertEqual(c.topk, 1)
        self.assertEqual(c.num_draft_tokens, 4)


if __name__ == "__main__":
    unittest.main()
