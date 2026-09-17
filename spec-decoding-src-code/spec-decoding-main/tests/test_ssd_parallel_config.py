# SPDX-License-Identifier: Apache-2.0
"""Tests for spec_decoding.ssd_parallel.config."""

from __future__ import annotations

import unittest

from spec_decoding.ssd_parallel import Eagle3Config


class TestSSDParallelConfig(unittest.TestCase):
    def test_draft_async_requires_spec_cache(self):
        with self.assertRaises(ValueError):
            Eagle3Config(draft_async=True, enable_spec_cache=False)

    def test_draft_async_requires_jit_speculate(self):
        with self.assertRaises(ValueError):
            Eagle3Config(draft_async=True, jit_speculate=False)

    def test_draft_async_ok_with_phase3_flags(self):
        cfg = Eagle3Config(
            draft_async=True,
            enable_spec_cache=True,
            jit_speculate=True,
            draft_model_path="/tmp/draft",
        )
        self.assertTrue(cfg.draft_async)
        self.assertTrue(cfg.enable_spec_cache)


if __name__ == "__main__":
    unittest.main()
