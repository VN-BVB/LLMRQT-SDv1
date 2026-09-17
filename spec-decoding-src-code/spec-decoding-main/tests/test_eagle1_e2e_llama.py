# SPDX-License-Identifier: Apache-2.0
"""
End-to-end EAGLE1 on a Llama-series target with the lightweight EAGLE1 draft
(reuse target embedding/lm_head + 1 decoder layer, skip layer-0 input_layernorm)
and a static top-k tree.

The draft is randomly initialized (no trained EAGLE1 checkpoint), so accept rate is
low — but greedy tree verify keeps the output lossless vs. plain greedy.

Gated behind an env var (downloads + GPU helpful):

    RUN_SPEC_E2E=1 PYTHONPATH=. python3 -m unittest tests.test_eagle1_e2e_llama -v
"""

from __future__ import annotations

import os
import unittest

import torch

from spec_decoding.eagle import (
    Eagle1Config,
    Eagle1Generator,
    build_eagle1_draft,
    make_eagle1_draft_topk_fn,
)

_RUN = os.environ.get("RUN_SPEC_E2E") == "1"
_TARGET_ID = os.environ.get("SPEC_E2E_TARGET", "JackFram/llama-160m")
_PROMPT = os.environ.get("SPEC_E2E_PROMPT", "The capital of France is")
_MAX_NEW = int(os.environ.get("SPEC_E2E_MAX_NEW", "24"))


@unittest.skipUnless(_RUN, "set RUN_SPEC_E2E=1 to run the EAGLE1 Llama end-to-end test")
class TestEagle1E2ELlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if cls.device.type == "cuda" else torch.float32
        cls.tok = AutoTokenizer.from_pretrained(_TARGET_ID)
        cls.target = (
            AutoModelForCausalLM.from_pretrained(_TARGET_ID, torch_dtype=dtype)
            .to(cls.device)
            .eval()
        )
        cls.draft = build_eagle1_draft(cls.target, num_layers=1)

    def test_lossless_vs_greedy(self) -> None:
        enc = self.tok(_PROMPT, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        plen = input_ids.shape[1]

        with torch.inference_mode():
            base = self.target.generate(
                input_ids,
                max_new_tokens=_MAX_NEW,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id,
            )
        base_cont = base[0, plen:]

        cfg = Eagle1Config(
            topk=2, num_steps=2, max_tree_nodes=8, verify_mode="reference_paths"
        )
        fn = make_eagle1_draft_topk_fn(self.draft, topk=cfg.topk)
        gen = Eagle1Generator(self.target, cfg, fn)
        out = gen.generate(input_ids, max_new_tokens=_MAX_NEW, eos_token_id=self.tok.eos_token_id)
        spec_cont = out[0, plen:]

        m = min(int(base_cont.numel()), int(spec_cont.numel()))
        self.assertGreater(m, 0)
        if not torch.equal(base_cont[:m].cpu(), spec_cont[:m].cpu()):
            self.fail(
                "EAGLE1 output diverged from greedy baseline:\n"
                f"  base: {base_cont[:m].tolist()}\n  spec: {spec_cont[:m].tolist()}"
            )


if __name__ == "__main__":
    unittest.main()
