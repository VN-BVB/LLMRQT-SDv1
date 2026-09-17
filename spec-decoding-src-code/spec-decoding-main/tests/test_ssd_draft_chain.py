# SPDX-License-Identifier: Apache-2.0
"""CPU tests for eagle3.draft / eagle3.chain with a tiny mock draft."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from spec_decoding.ssd.chain import make_eagle3_draft_topk_fn
from spec_decoding.ssd.draft import draft_forward_step, resolve_conditioning


class _TinyDraft(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, past_key_values=None, use_cache=False, output_hidden_states=False, return_dict=True, **kwargs):
        h = inputs_embeds
        logits = self.lm(h)
        from types import SimpleNamespace
        return SimpleNamespace(
            logits=logits,
            hidden_states=(h,) if output_hidden_states else None,
            past_key_values=past_key_values,
        )


class TestEagle3DraftChain(unittest.TestCase):
    def test_resolve_conditioning(self):
        fc = nn.Linear(12, 8, bias=False)
        raw = torch.randn(2, 12)
        proj = resolve_conditioning(raw, fc)
        self.assertEqual(proj.shape, (2, 8))
        prenorm = torch.randn(2, 8)
        same = resolve_conditioning(prenorm, fc)
        self.assertTrue(torch.allclose(same, prenorm))

    def test_draft_forward_and_topk_fn(self):
        hidden, target_dim, vocab = 8, 12, 32
        draft = _TinyDraft(vocab=vocab, hidden=hidden)
        fc = nn.Linear(target_dim, hidden, bias=False)
        parent = torch.tensor([[5]])
        acts = torch.randn(1, target_dim)
        logits, prenorm, _ = draft_forward_step(draft, fc, parent, acts)
        self.assertEqual(logits.shape, (1, vocab))
        self.assertEqual(prenorm.shape, (1, hidden))

        fn = make_eagle3_draft_topk_fn(draft, fc, topk=3)
        ids, lps = fn(acts, parent)
        self.assertEqual(ids.shape, (1, 3))
        self.assertEqual(lps.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
