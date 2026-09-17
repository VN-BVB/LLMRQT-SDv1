# SPDX-License-Identifier: Apache-2.0
"""
EAGLE2 with the lightweight EAGLE draft model (reuse target embedding/lm_head + fc +
1 decoder layer). Gated behind RUN_SPEC_E2E (loads a tiny HF model).

    RUN_SPEC_E2E=1 PYTHONPATH=. python3 -m unittest tests.test_eagle2_draft_model -v
"""

from __future__ import annotations

import os
import unittest

import torch

_RUN = os.environ.get("RUN_SPEC_E2E") == "1"
_MODEL = os.environ.get("SPEC_E2E_TARGET_SMALL", "sshleifer/tiny-gpt2")


@unittest.skipUnless(_RUN, "set RUN_SPEC_E2E=1 to run the eagle2 draft-model test")
class TestEagle2DraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import GPT2LMHeadModel

        cls.target = GPT2LMHeadModel.from_pretrained(_MODEL).eval()

    def test_draft_topk_fn_shapes(self) -> None:
        from spec_decoding.eagle2 import build_eagle2_draft, make_eagle2_draft_topk_fn

        draft = build_eagle2_draft(self.target, num_layers=1)
        topk = 3
        fn = make_eagle2_draft_topk_fn(draft, topk=topk)
        hidden = torch.zeros(1, int(self.target.config.hidden_size))
        parent = torch.tensor([[5]], dtype=torch.long)
        idx, lp = fn(hidden, parent)
        self.assertEqual(idx.shape, (1, topk))
        self.assertEqual(lp.shape, (1, topk))
        # 分数应为 log 概率（<= 0）
        self.assertTrue(bool((lp <= 1e-4).all()))

    def test_generate_lossless_vs_greedy(self) -> None:
        # reference_paths 用 _verify_path_clean（已与 EAGLE1 对齐），应与贪心一致（无损）。
        from spec_decoding.eagle2 import (
            Eagle2TreeConfig,
            Eagle2Generator,
            build_eagle2_draft,
            make_eagle2_draft_topk_fn,
        )

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        max_new = 8
        with torch.inference_mode():
            base = self.target.generate(
                input_ids, max_new_tokens=max_new, do_sample=False, num_beams=1,
                use_cache=True, pad_token_id=self.target.config.eos_token_id,
            )
        base_cont = base[0, input_ids.shape[1]:]

        draft = build_eagle2_draft(self.target, num_layers=1)
        cfg = Eagle2TreeConfig(
            topk=2, num_steps=2, max_tree_nodes=8,
            tree_expand_mode="cumulative", verify_mode="reference_paths",
        )
        gen = Eagle2Generator(self.target, cfg, make_eagle2_draft_topk_fn(draft, topk=2))
        out = gen.generate(input_ids, max_new_tokens=max_new)
        spec_cont = out[0, input_ids.shape[1]:]

        m = min(int(base_cont.numel()), int(spec_cont.numel()))
        self.assertGreater(m, 0)
        self.assertTrue(
            torch.equal(base_cont[:m].cpu(), spec_cont[:m].cpu()),
            f"eagle2 reference_paths diverged from greedy:\n base={base_cont[:m].tolist()}\n spec={spec_cont[:m].tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
