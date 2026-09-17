# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the EAGLE3Pro + Speculative Speculative Decoding layer."""

from __future__ import annotations

import unittest

import torch


class _MappedDraft:
    def topk_target_ids(self, logits: torch.Tensor, k: int):
        scores, ids = torch.topk(logits, k=k, dim=-1)
        return ids, scores


class TestEagle3ProSSDPolicy(unittest.TestCase):
    def test_config_rejects_empty_fan_out(self) -> None:
        from spec_decoding.eagle3pro.ssd import Eagle3ProSSDConfig

        with self.assertRaises(ValueError):
            Eagle3ProSSDConfig(fan_out=0)

    def test_recovery_fork_excludes_current_path_token(self) -> None:
        from spec_decoding.eagle3pro.ssd import _select_recovery_tokens

        logits = torch.tensor([[0.0, 9.0, 8.0, 7.0, 6.0]])
        selected = _select_recovery_tokens(
            _MappedDraft(), logits, 3, excluded_token=1
        )
        self.assertEqual(selected, [2, 3, 4])

    def test_cache_key_separates_accept_length(self) -> None:
        from spec_decoding.eagle3pro.ssd import SSDCacheKey

        cache = {
            SSDCacheKey(0, 7): "miss-first-token",
            SSDCacheKey(1, 7): "full-match",
        }
        self.assertEqual(cache[SSDCacheKey(0, 7)], "miss-first-token")
        self.assertEqual(cache[SSDCacheKey(1, 7)], "full-match")


class TestEagle3ProSSDE2E(unittest.TestCase):
    def test_matches_base_eagle3pro_on_tiny_llama(self) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM

        from spec_decoding.eagle3pro import (
            Eagle3ProSSDConfig,
            Eagle3ProSSDGenerator,
            Eagle3SglConfig,
            Eagle3SglGenerator,
            build_eagle3_sgl_draft,
            make_eagle3_sgl_draft_topk_fn,
        )

        torch.manual_seed(1234)
        target = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=64,
                eos_token_id=None,
            )
        ).eval()
        cfg = Eagle3SglConfig(
            topk=1,
            num_steps=1,
            max_tree_nodes=1,
            eagle_layers=[0, 2, 3],
            verify_mode="full_model_tree",
        )
        torch.manual_seed(7)
        draft = build_eagle3_sgl_draft(
            target,
            cfg.eagle_layers,
            num_layers=1,
            pack_projections=False,
        )
        topk_fn = make_eagle3_sgl_draft_topk_fn(draft, topk=1)
        input_ids = torch.tensor([[1, 7, 13, 21, 30]], dtype=torch.long)

        base = Eagle3SglGenerator(
            target, cfg, topk_fn, draft_model=draft
        ).generate(input_ids, max_new_tokens=8)
        ssd_generator = Eagle3ProSSDGenerator(
            target,
            cfg,
            topk_fn,
            draft_model=draft,
            ssd_cfg=Eagle3ProSSDConfig(fan_out=3, overlap=False),
        )
        actual = ssd_generator.generate(input_ids, max_new_tokens=8)
        stats = ssd_generator.ssd_stats
        ssd_generator.close()

        self.assertTrue(torch.equal(actual, base))
        self.assertGreater(stats.cache_hits + stats.cache_misses, 0)
        self.assertEqual(stats.forked_branches % (2 * 3), 0)


if __name__ == "__main__":
    unittest.main()
