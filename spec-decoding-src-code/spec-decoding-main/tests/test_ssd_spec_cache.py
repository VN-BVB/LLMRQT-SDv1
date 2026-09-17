# SPDX-License-Identifier: Apache-2.0
"""CPU tests for spec_decoding.ssd speculation cache."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from spec_decoding.ssd.config import Eagle3Config
from spec_decoding.ssd.spec_cache import CacheEntry, SpeculationCache
from spec_decoding.ssd.speculate_ssd import speculate_with_cache
from spec_decoding.ssd.types import CacheKey


class _TinyDraft(nn.Module):
    def __init__(self, hidden: int = 8, vocab: int = 32):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden, "num_hidden_layers": 1})()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_key_values=None, use_cache=False, output_hidden_states=False, return_dict=True, **kwargs):
        h = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        from types import SimpleNamespace
        hs = (h,) if output_hidden_states else None
        return SimpleNamespace(
            logits=self.lm(h),
            hidden_states=hs,
            past_key_values=past_key_values,
        )


class TestSpecCache(unittest.TestCase):
    def test_replace_and_lookup(self):
        cache = SpeculationCache()
        spec = torch.tensor([10, 1, 2, 3])
        cache.replace_all(
            [CacheKey(0, -1, 10)],
            [CacheEntry(spec)],
        )
        self.assertEqual(len(cache), 1)
        self.assertIsNotNone(cache.lookup(CacheKey(0, -1, 10)))

    def test_speculate_with_cache_hit(self):
        cfg = Eagle3Config(speculate_k=2, enable_spec_cache=True)
        cache = SpeculationCache()
        cached_spec = torch.tensor([7, 11, 12, 13])
        cache.replace_all([CacheKey(0, -1, 7)], [CacheEntry(cached_spec)])
        draft = _TinyDraft()
        fc = nn.Linear(24, 8, bias=False)
        result = speculate_with_cache(
            cfg,
            cache,
            draft,
            fc,
            torch.randn(1, 24),
            torch.tensor([[7]]),
            CacheKey(0, -1, 7),
        )
        self.assertEqual(int(result.cache_hits[0].item()), 1)


if __name__ == "__main__":
    unittest.main()
