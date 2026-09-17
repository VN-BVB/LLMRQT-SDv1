# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for core.eagle3 Phase 0 (config + types, no GPU/HF)."""

from __future__ import annotations

import unittest

from spec_decoding.eagle3 import (
    CacheKey,
    Eagle3Config,
    SpeculateResult,
    VerifyResult,
    default_eagle_layers,
)


class TestEagle3Config(unittest.TestCase):
    def test_defaults_and_mq_len(self):
        cfg = Eagle3Config(speculate_k=6, async_fan_out=3)
        self.assertEqual(cfg.fan_out_list, [3] * 7)
        self.assertEqual(cfg.mq_len, 21)

    def test_draft_async_requires_jit(self):
        with self.assertRaises(ValueError):
            Eagle3Config(draft_async=True, jit_speculate=False)

    def test_resolve_eagle_layers_default(self):
        cfg = Eagle3Config()
        layers = cfg.resolve_eagle_layers(32)
        self.assertEqual(layers, default_eagle_layers(32))

    def test_resolve_eagle_layers_invalid(self):
        cfg = Eagle3Config(eagle_layers=[0, 99])
        with self.assertRaises(ValueError):
            cfg.resolve_eagle_layers(32)


class TestEagle3Types(unittest.TestCase):
    def test_cache_key_tuple(self):
        k = CacheKey(seq_id=1, keep_idx=2, recovery_token=42)
        self.assertEqual(k.as_tuple(), (1, 2, 42))


class TestEagle3Imports(unittest.TestCase):
    def test_lazy_generator_import(self):
        from spec_decoding.eagle3 import Eagle3GreedyGenerator

        self.assertTrue(Eagle3GreedyGenerator.__name__ == "Eagle3GreedyGenerator")

    def test_generator_requires_fc(self):
        from spec_decoding.eagle3 import Eagle3GreedyGenerator

        with self.assertRaises(TypeError):
            Eagle3GreedyGenerator(target=object(), draft=object(), cfg=Eagle3Config())


if __name__ == "__main__":
    unittest.main()
