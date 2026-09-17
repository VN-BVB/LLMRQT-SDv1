# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3-SGL（EAGLE3 多层特征 + EAGLE1 静态树 + tree verify）。
Gated behind RUN_SPEC_E2E（loads a tiny HF model）。

    RUN_SPEC_E2E=1 PYTHONPATH=. python3 -m unittest tests.test_eagle3_sgl -v
"""

from __future__ import annotations

import os
import unittest

import torch

_RUN = os.environ.get("RUN_SPEC_E2E") == "1"
_MODEL = os.environ.get("SPEC_E2E_TARGET_SMALL", "sshleifer/tiny-gpt2")


class TestEagle3SglConfigCpu(unittest.TestCase):
    """轻量 CPU 单测（无需模型/下载）：config 解析。"""

    def test_resolve_eagle_layers_explicit(self) -> None:
        from spec_decoding.eagle3_sgl import Eagle3SglConfig

        cfg = Eagle3SglConfig(eagle_layers=[0, 1])
        self.assertEqual(cfg.resolve_eagle_layers(2), [0, 1])

    def test_invalid_verify_mode(self) -> None:
        from spec_decoding.eagle3_sgl import Eagle3SglConfig

        with self.assertRaises(ValueError):
            Eagle3SglConfig(verify_mode="nope")


@unittest.skipUnless(_RUN, "set RUN_SPEC_E2E=1 to run the eagle3_sgl model test")
class TestEagle3Sgl(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import GPT2LMHeadModel

        cls.target = GPT2LMHeadModel.from_pretrained(_MODEL).eval()
        cls.n_layers = int(
            getattr(cls.target.config, "num_hidden_layers", getattr(cls.target.config, "n_layer", 2))
        )
        cls.eagle_layers = [0, 1] if cls.n_layers <= 2 else None

    def test_draft_structure_aligned_with_sglang(self) -> None:
        from spec_decoding.eagle3_sgl import build_eagle3_sgl_draft, make_eagle3_sgl_draft_topk_fn

        layers = [0, 1] if self.n_layers <= 2 else list(range(min(3, self.n_layers)))
        draft = build_eagle3_sgl_draft(self.target, layers, num_layers=1)
        H = int(self.target.config.hidden_size)
        # (1) 多层 aux：fc 输入 = n*H（降维多层 act，与 embedding 分开）
        self.assertEqual(draft.fc.in_features, len(layers) * H)
        self.assertEqual(draft.fc.out_features, H)
        # (3) midlayer：qkv 输入 = 2H，且有独立的 hidden_norm
        self.assertEqual(draft.midlayer.q_proj.in_features, 2 * H)
        self.assertTrue(hasattr(draft.midlayer, "hidden_norm"))
        fn = make_eagle3_sgl_draft_topk_fn(draft, topk=3)
        acts = torch.zeros(1, len(layers) * H)
        idx, sc = fn(acts, torch.tensor([[5]], dtype=torch.long))
        self.assertEqual(idx.shape, (1, 3))

    def test_independent_draft_vocab_and_d2t(self) -> None:
        # (2) 独立 draft 词表 + d2t 映射 + 不共享 lm_head
        from spec_decoding.eagle3_sgl import build_eagle3_sgl_draft, make_eagle3_sgl_draft_topk_fn

        layers = [0, 1] if self.n_layers <= 2 else list(range(min(3, self.n_layers)))
        H = int(self.target.config.hidden_size)
        dv = 16
        # d2t 全 0 -> hot_token_id = arange(dv)，draft id 即落在 target [0,dv) 上
        draft = build_eagle3_sgl_draft(
            self.target, layers, draft_vocab_size=dv, d2t=torch.zeros(dv, dtype=torch.long)
        )
        self.assertIsNotNone(draft.lm_head)
        self.assertEqual(draft.lm_head.out_features, dv)        # 独立 draft 词表大小
        self.assertNotEqual(dv, int(self.target.config.vocab_size))  # 不等于 target 词表
        fn = make_eagle3_sgl_draft_topk_fn(draft, topk=3)
        idx, sc = fn(torch.zeros(1, len(layers) * H), torch.tensor([[5]], dtype=torch.long))
        self.assertEqual(idx.shape, (1, 3))
        self.assertTrue(int(idx.max()) < dv)  # 经 d2t 映射后落在 target 词表前 dv 个 id

    def test_load_weights_roundtrip(self) -> None:
        # draft_weights_path=None -> 随机初始化；给定路径 -> 加载权重。
        # 这里 round-trip 验证加载机制：保存一个 draft 的权重再装回新 draft，输出应一致。
        import tempfile

        from spec_decoding.eagle3_sgl import (
            build_eagle3_sgl_draft,
            load_eagle3_draft_weights,
            make_eagle3_sgl_draft_topk_fn,
        )

        layers = [0, 1] if self.n_layers <= 2 else list(range(min(3, self.n_layers)))
        H = int(self.target.config.hidden_size)
        dv = 16
        d2t = torch.zeros(dv, dtype=torch.long)

        src = build_eagle3_sgl_draft(self.target, layers, draft_vocab_size=dv, d2t=d2t)
        path = tempfile.mktemp(suffix=".pt")
        torch.save(src.state_dict(), path)

        # 新建一个（随机权重）draft，再从文件加载
        dst = build_eagle3_sgl_draft(self.target, layers, draft_vocab_size=dv, d2t=d2t)
        report = load_eagle3_draft_weights(dst, path)
        self.assertEqual(report["unexpected"], [])
        self.assertGreater(len(report["loaded"]), 0)

        # 加载后两者输出应一致
        acts = torch.randn(1, len(layers) * H)
        parent = torch.tensor([[7]], dtype=torch.long)
        with torch.inference_mode():
            self.assertTrue(torch.allclose(src(parent, acts), dst(parent, acts), atol=1e-5))

        # 也验证「build 时直接传 draft_weights_path」这条路
        dst2 = build_eagle3_sgl_draft(
            self.target, layers, draft_vocab_size=dv, d2t=d2t, draft_weights_path=path
        )
        with torch.inference_mode():
            self.assertTrue(torch.allclose(src(parent, acts), dst2(parent, acts), atol=1e-5))

    def test_lossless_vs_greedy(self) -> None:
        from spec_decoding.eagle3_sgl import (
            Eagle3SglConfig,
            Eagle3SglGenerator,
            build_eagle3_sgl_draft,
            make_eagle3_sgl_draft_topk_fn,
        )

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        max_new = 8
        with torch.inference_mode():
            base = self.target.generate(
                input_ids, max_new_tokens=max_new, do_sample=False, num_beams=1,
                use_cache=True, pad_token_id=self.target.config.eos_token_id,
            )
        base_cont = base[0, input_ids.shape[1]:]

        cfg = Eagle3SglConfig(
            topk=2, num_steps=2, max_tree_nodes=8,
            eagle_layers=self.eagle_layers, verify_mode="reference_paths",
        )
        layers = cfg.resolve_eagle_layers(self.n_layers)
        draft = build_eagle3_sgl_draft(self.target, layers, num_layers=1)
        gen = Eagle3SglGenerator(
            self.target,
            cfg,
            make_eagle3_sgl_draft_topk_fn(draft, topk=2),
            draft_model=draft,
            autoregressive_draft=True,
        )
        out = gen.generate(input_ids, max_new_tokens=max_new)
        spec_cont = out[0, input_ids.shape[1]:]

        m = min(int(base_cont.numel()), int(spec_cont.numel()))
        self.assertGreater(m, 0)
        self.assertTrue(
            torch.equal(base_cont[:m].cpu(), spec_cont[:m].cpu()),
            f"eagle3_sgl diverged from greedy:\n base={base_cont[:m].tolist()}\n spec={spec_cont[:m].tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
