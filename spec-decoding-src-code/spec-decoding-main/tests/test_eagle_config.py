# SPDX-License-Identifier: Apache-2.0
"""EAGLE1 config tests (CPU, no model)."""

from __future__ import annotations

import unittest

from spec_decoding.eagle.config import Eagle1Config


class TestEagle1Config(unittest.TestCase):
    def test_defaults(self) -> None:
        c = Eagle1Config()
        self.assertEqual(c.topk, 4)
        self.assertEqual(c.num_steps, 3)
        self.assertEqual(c.max_tree_nodes, 32)
        self.assertEqual(c.verify_mode, "full_model_tree")

    def test_no_cumulative_mode(self) -> None:
        # EAGLE1 has no cumulative-probability tree-expand mode.
        self.assertFalse(hasattr(Eagle1Config(), "tree_expand_mode"))

    def test_invalid_topk(self) -> None:
        with self.assertRaises(ValueError):
            Eagle1Config(topk=0)

    def test_invalid_verify_mode(self) -> None:
        with self.assertRaises(ValueError):
            Eagle1Config(verify_mode="nope")


if __name__ == "__main__":
    unittest.main()
