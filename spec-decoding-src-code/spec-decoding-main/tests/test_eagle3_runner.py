# SPDX-License-Identifier: Apache-2.0
"""Tests for Eagle3GreedyGenerator config guards and build_eagle3_fc."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from spec_decoding.eagle3 import Eagle3Config, build_eagle3_fc
from spec_decoding.eagle3.runner import Eagle3GreedyGenerator


class _CfgModel(nn.Module):
    def __init__(self, hidden: int, n_layers: int = 4, vocab: int = 64):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden, "num_hidden_layers": n_layers})()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_key_values=None, use_cache=False, output_hidden_states=False, return_dict=True, **kwargs):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed(input_ids)
        logits = self.lm(h)
        hs = (h,) if output_hidden_states else None
        from types import SimpleNamespace
        return SimpleNamespace(
            logits=logits,
            hidden_states=hs,
            past_key_values=past_key_values,
        )


class _Target(_CfgModel):
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, output_hidden_states=False, return_dict=True, **kwargs):
        h = self.embed(input_ids)
        logits = self.lm(h)
        from types import SimpleNamespace
        hs = tuple(h for _ in range(self.config.num_hidden_layers + 1)) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hs, past_key_values=past_key_values)


class TestEagle3Runner(unittest.TestCase):
    def test_build_fc_shape(self):
        target = _Target(hidden=16, n_layers=8)
        draft = _CfgModel(hidden=8)
        cfg = Eagle3Config(speculate_k=2)
        fc = build_eagle3_fc(target, draft, cfg)
        self.assertEqual(fc.in_features, 3 * 16)
        self.assertEqual(fc.out_features, 8)

    def test_rejects_async_cfg(self):
        target = _Target(hidden=8, n_layers=4)
        draft = _CfgModel(hidden=8)
        cfg = Eagle3Config(draft_async=True)
        fc = build_eagle3_fc(target, draft, cfg)
        with self.assertRaises(ValueError):
            Eagle3GreedyGenerator(target, draft, cfg, fc)

    def test_rejects_spec_cache_cfg(self):
        target = _Target(hidden=8, n_layers=4)
        draft = _CfgModel(hidden=8)
        cfg = Eagle3Config(enable_spec_cache=True)
        fc = build_eagle3_fc(target, draft, cfg)
        with self.assertRaises(ValueError):
            Eagle3GreedyGenerator(target, draft, cfg, fc)


if __name__ == "__main__":
    unittest.main()
