# SPDX-License-Identifier: Apache-2.0
"""Tests for spec_decoding.ssd.runner."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from spec_decoding.ssd import Eagle3Config, SSDGenerator, build_ssd_fc


class _CfgModel(nn.Module):
    def __init__(self, hidden: int, n_layers: int = 4, vocab: int = 64):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden, "num_hidden_layers": n_layers})()
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


class _Target(_CfgModel):
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, output_hidden_states=False, return_dict=True, **kwargs):
        h = self.embed(input_ids)
        from types import SimpleNamespace
        hs = tuple(h for _ in range(self.config.num_hidden_layers + 1)) if output_hidden_states else None
        return SimpleNamespace(logits=self.lm(h), hidden_states=hs, past_key_values=past_key_values)


class TestSSDRunner(unittest.TestCase):
    def test_build_fc_shape(self):
        target = _Target(hidden=16, n_layers=8)
        draft = _CfgModel(hidden=8)
        cfg = Eagle3Config(speculate_k=2)
        fc = build_ssd_fc(target, draft, cfg)
        self.assertEqual(fc.in_features, 3 * 16)

    def test_rejects_async_cfg(self):
        target = _Target(hidden=8, n_layers=4)
        draft = _CfgModel(hidden=8)
        cfg = Eagle3Config(draft_async=True)
        fc = build_ssd_fc(target, draft, cfg)
        with self.assertRaises(ValueError):
            SSDGenerator(target, draft, cfg, fc)

    def test_accepts_spec_cache_cfg(self):
        target = _Target(hidden=8, n_layers=4)
        draft = _CfgModel(hidden=8)
        cfg = Eagle3Config(enable_spec_cache=True, speculate_k=2)
        fc = build_ssd_fc(target, draft, cfg)
        gen = SSDGenerator(target, draft, cfg, fc)
        self.assertIsNotNone(gen.spec_cache)


if __name__ == "__main__":
    unittest.main()
