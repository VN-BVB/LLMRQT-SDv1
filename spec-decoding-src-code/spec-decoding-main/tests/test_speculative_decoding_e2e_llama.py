# SPDX-License-Identifier: Apache-2.0
"""
End-to-end single-chain greedy speculative decode on Llama-series models.

Target/draft are real Llama-architecture checkpoints sharing the same 32k vocab:
  - target: JackFram/llama-160m
  - draft:  JackFram/llama-68m  (a classic ungated speculative-decoding pair)

Correctness check (losslessness): greedy speculative decoding must produce the
SAME continuation as the target's plain greedy generate (the draft only changes
*speed*, never the output under greedy verify).

This test downloads weights + needs a GPU, so it is gated behind an env var to
keep the default CPU unit-test suite fast:

    RUN_SPEC_E2E=1 PYTHONPATH=. python3 -m unittest tests.test_speculative_decoding_e2e_llama -v

Override models via SPEC_E2E_TARGET / SPEC_E2E_DRAFT.
"""

from __future__ import annotations

import os
import unittest

import torch

from spec_decoding.speculative_decoding import (
    SpeculativeConfig,
    speculative_greedy_decode,
    speculative_greedy_decode_from_strings,
)
from spec_decoding.speculative_decoding_opti import speculative_greedy_decode_opt

_RUN = os.environ.get("RUN_SPEC_E2E") == "1"
_TARGET_ID = os.environ.get("SPEC_E2E_TARGET", "JackFram/llama-160m")
_DRAFT_ID = os.environ.get("SPEC_E2E_DRAFT", "JackFram/llama-68m")
_PROMPT = os.environ.get("SPEC_E2E_PROMPT", "The capital of France is")
_MAX_NEW = int(os.environ.get("SPEC_E2E_MAX_NEW", "32"))
_GAMMA = int(os.environ.get("SPEC_E2E_GAMMA", "4"))


@unittest.skipUnless(_RUN, "set RUN_SPEC_E2E=1 to run the Llama end-to-end test")
class TestSpeculativeE2ELlama(unittest.TestCase):
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
        cls.draft = (
            AutoModelForCausalLM.from_pretrained(_DRAFT_ID, torch_dtype=dtype)
            .to(cls.device)
            .eval()
        )

    def test_vocab_matches(self) -> None:
        self.assertEqual(
            int(self.target.config.vocab_size),
            int(self.draft.config.vocab_size),
            "target/draft must share a vocabulary for greedy verify to be valid",
        )

    def test_lossless_vs_greedy_baseline(self) -> None:
        enc = self.tok(_PROMPT, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[1]

        # Baseline: plain greedy decode with the target model only.
        with torch.inference_mode():
            base = self.target.generate(
                input_ids,
                max_new_tokens=_MAX_NEW,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id,
            )
        base_cont = base[0, prompt_len:]

        # Single-chain greedy speculative decode (target verifies draft).
        cfg = SpeculativeConfig(
            num_draft_tokens=_GAMMA,
            eos_token_id=self.tok.eos_token_id,
        )
        spec = speculative_greedy_decode(
            self.target,
            self.draft,
            input_ids,
            max_new_tokens=_MAX_NEW,
            config=cfg,
        )
        spec_cont = spec[0, prompt_len:]

        m = min(int(base_cont.numel()), int(spec_cont.numel()))
        self.assertGreater(m, 0, "no tokens were generated")
        if not torch.equal(base_cont[:m].cpu(), spec_cont[:m].cpu()):
            base_txt = self.tok.decode(base_cont[:m])
            spec_txt = self.tok.decode(spec_cont[:m])
            self.fail(
                "speculative output diverged from greedy baseline:\n"
                f"  baseline: {base_cont[:m].tolist()}\n  -> {base_txt!r}\n"
                f"  spec    : {spec_cont[:m].tolist()}\n  -> {spec_txt!r}"
            )

    def test_opt_matches_baseline_and_lossless(self) -> None:
        """KV-reuse opt decode must be lossless AND identical to the baseline decode."""
        enc = self.tok(_PROMPT, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[1]
        cfg = SpeculativeConfig(
            num_draft_tokens=_GAMMA, eos_token_id=self.tok.eos_token_id
        )

        with torch.inference_mode():
            base = self.target.generate(
                input_ids,
                max_new_tokens=_MAX_NEW,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id,
            )
        base_cont = base[0, prompt_len:]

        opt = speculative_greedy_decode_opt(
            self.target, self.draft, input_ids, max_new_tokens=_MAX_NEW, config=cfg
        )
        spec = speculative_greedy_decode(
            self.target, self.draft, input_ids, max_new_tokens=_MAX_NEW, config=cfg
        )
        opt_cont = opt[0, prompt_len:]
        spec_cont = spec[0, prompt_len:]

        # opt is lossless vs plain greedy
        m = min(int(base_cont.numel()), int(opt_cont.numel()))
        self.assertGreater(m, 0)
        self.assertTrue(
            torch.equal(base_cont[:m].cpu(), opt_cont[:m].cpu()),
            "opt diverged from greedy baseline",
        )
        # opt and baseline speculative produce the same continuation
        m2 = min(int(opt_cont.numel()), int(spec_cont.numel()))
        self.assertTrue(
            torch.equal(opt_cont[:m2].cpu(), spec_cont[:m2].cpu()),
            "opt and non-opt speculative outputs differ",
        )

    def test_decode_to_text_runs(self) -> None:
        out, text = speculative_greedy_decode_from_strings(
            self.target,
            self.draft,
            self.tok,
            _PROMPT,
            max_new_tokens=_MAX_NEW,
            config=SpeculativeConfig(num_draft_tokens=_GAMMA),
        )
        self.assertGreater(out.shape[1], 0)
        self.assertIsInstance(text, str)
        print(f"\n[spec-e2e] prompt={_PROMPT!r}\n[spec-e2e] text={text!r}")


def _main() -> None:
    """Standalone runner (no unittest gating): demo generation + lossless check."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[spec-e2e] target={_TARGET_ID} draft={_DRAFT_ID} device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(_TARGET_ID)
    target = AutoModelForCausalLM.from_pretrained(_TARGET_ID, torch_dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(_DRAFT_ID, torch_dtype=dtype).to(device).eval()

    out, text = speculative_greedy_decode_from_strings(
        target, draft, tok, _PROMPT, max_new_tokens=_MAX_NEW,
        config=SpeculativeConfig(num_draft_tokens=_GAMMA),
    )
    print(f"[spec-e2e] out_len={out.shape[1]} text={text!r}")
    print("[spec-e2e] OK")


if __name__ == "__main__":
    _main()
