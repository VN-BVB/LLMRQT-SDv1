# SPDX-License-Identifier: Apache-2.0
"""
SSD × EAGLE3-SGL：eagle3_sgl 树投机 + 官方 SSD 投机缓存集成。

包含：
- 纯 CPU 单测（无需模型/下载）：config 校验、spec_cache、fork（top-F 含 argmax、屏蔽路径 token）。
- Gated（RUN_SPEC_E2E=1）模型端到端：用一个 小随机 LlamaForCausalLM（token 多样）验证
  (a) 无损 vs 贪心；(b) SSD 缓存版输出 == 基座 Eagle3SglGenerator；(c) cache 确有命中。

    PYTHONPATH=. python3 -m unittest tests.test_ssd_eagle3sgl_style -v
    RUN_SPEC_E2E=1 PYTHONPATH=. python3 -m unittest tests.test_ssd_eagle3sgl_style -v
"""

from __future__ import annotations

import os
import unittest

import torch

_RUN = os.environ.get("RUN_SPEC_E2E") == "1"


class TestSsdConfigCpu(unittest.TestCase):
    """config：SSD 字段校验（不依赖模型）。"""

    def test_ssd_fields_defaults(self) -> None:
        from spec_decoding.ssd_eagle3sgl_style import Eagle3SglConfig

        cfg = Eagle3SglConfig(enable_spec_cache=True, fan_out=3)
        self.assertTrue(cfg.enable_spec_cache)
        self.assertEqual(cfg.fan_out, 3)
        self.assertTrue(cfg.jit_speculate)

    def test_invalid_fan_out(self) -> None:
        from spec_decoding.ssd_eagle3sgl_style import Eagle3SglConfig

        with self.assertRaises(ValueError):
            Eagle3SglConfig(fan_out=0)


class TestSpecCacheCpu(unittest.TestCase):
    """spec_cache：replace_all / lookup / clear / root_token。"""

    def _entry(self, tok: int):
        from spec_decoding.ssd_eagle3sgl_style.spec_cache import CacheEntry
        from spec_decoding.ssd_eagle3sgl_style.tree_draft import TreeDraftResult

        tree = TreeDraftResult(token_ids=[tok], parent=[-1], node_depth=[1], bfs_index=[0])
        return CacheEntry(tree=tree, root_token=tok)

    def test_replace_lookup_clear(self) -> None:
        from spec_decoding.ssd_eagle3sgl_style.spec_cache import CacheKey, SpeculationCache

        cache = SpeculationCache()
        keys = [CacheKey(0, 0, 11), CacheKey(0, 1, 22)]
        entries = [self._entry(11), self._entry(22)]
        cache.replace_all(keys, entries)
        self.assertEqual(len(cache), 2)
        hit = cache.lookup(CacheKey(0, 1, 22))
        self.assertIsNotNone(hit)
        self.assertEqual(hit.root_token, 22)
        self.assertIsNone(cache.lookup(CacheKey(0, 1, 23)))
        # replace_all 是整表替换
        cache.replace_all([CacheKey(0, 0, 99)], [self._entry(99)])
        self.assertEqual(len(cache), 1)
        self.assertIsNone(cache.lookup(CacheKey(0, 1, 22)))
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_length_mismatch_raises(self) -> None:
        from spec_decoding.ssd_eagle3sgl_style.spec_cache import CacheKey, SpeculationCache

        with self.assertRaises(ValueError):
            SpeculationCache().replace_all([CacheKey(0, 0, 1)], [])


class TestForkCpu(unittest.TestCase):
    """fork：top-F 候选含 argmax；中途失配位屏蔽路径 token；hypo 前缀正确。"""

    def test_fork_includes_argmax_and_masks_path(self) -> None:
        from spec_decoding.ssd_eagle3sgl_style import Eagle3SglConfig
        from spec_decoding.ssd_eagle3sgl_style.fork import fork_recovery_from_path_logits

        cfg = Eagle3SglConfig(fan_out=2)
        V = 6
        best_path = [3, 5]  # P=2 -> P+1=3 行
        logits = torch.full((3, V), -10.0)
        # 行0（keep_idx=0，失配位）：path[0]=3 本是最大，但应被屏蔽 -> 取次大 1
        logits[0, 3] = 100.0
        logits[0, 1] = 50.0
        logits[0, 2] = 40.0
        # 行1（keep_idx=1，失配位）：path[1]=5 屏蔽 -> 取 4
        logits[1, 5] = 100.0
        logits[1, 4] = 50.0
        logits[1, 0] = 40.0
        # 行2（keep_idx=2，全中 bonus 位，不屏蔽）：argmax=2
        logits[2, 2] = 100.0
        logits[2, 0] = 50.0

        branches = fork_recovery_from_path_logits(cfg, logits, best_path)
        bd = {}
        for (k, r) in branches:
            bd.setdefault(k, []).append(r)
        # keep_idx=0：屏蔽 3 后 top-2 = [1,2]
        self.assertEqual(bd[0][:2], [1, 2])
        self.assertNotIn(3, bd[0])
        # keep_idx=1：屏蔽 5 后 top-2 = [4,0]
        self.assertEqual(bd[1][:2], [4, 0])
        self.assertNotIn(5, bd[1])
        # keep_idx=2：bonus 位含 argmax=2
        self.assertIn(2, bd[2])

    def test_hypo_prefix(self) -> None:
        from spec_decoding.ssd_eagle3sgl_style.fork import hypo_prefix_for_fork

        out_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
        best_path = [7, 8, 9]
        # keep_idx=2, recovery=99 -> out_ids + path[:2] + [99]
        hypo = hypo_prefix_for_fork(out_ids, best_path, 2, 99)
        self.assertEqual(hypo[0].tolist(), [10, 11, 12, 7, 8, 99])
        # keep_idx=0 -> out_ids + [99]
        hypo0 = hypo_prefix_for_fork(out_ids, best_path, 0, 99)
        self.assertEqual(hypo0[0].tolist(), [10, 11, 12, 99])

    def test_cache_keys(self) -> None:
        from spec_decoding.ssd_eagle3sgl_style.fork import cache_keys_for_branches

        keys = cache_keys_for_branches(7, [(0, 5), (1, 9)])
        self.assertEqual([k.as_tuple() for k in keys], [(7, 0, 5), (7, 1, 9)])


def _tiny_llama(vocab: int = 64, n_layers: int = 4, hidden: int = 32):
    """构造一个小随机 LlamaForCausalLM（token 多样，CPU 可跑）。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1234)
    m = LlamaForCausalLM(cfg).eval()
    return m


@unittest.skipUnless(_RUN, "set RUN_SPEC_E2E=1 to run the SSD x EAGLE3-SGL e2e test")
class TestSsdEagle3SglE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = _tiny_llama()
        cls.n_layers = int(cls.target.config.num_hidden_layers)

    def _make(self, enable_cache: bool):
        from spec_decoding.ssd_eagle3sgl_style import (
            Eagle3SglConfig,
            Eagle3SglGenerator,
            SsdEagle3SglGenerator,
            build_eagle3_sgl_draft,
            make_eagle3_sgl_draft_topk_fn,
        )

        cfg = Eagle3SglConfig(
            topk=2,
            num_steps=3,
            max_tree_nodes=12,
            verify_mode="reference_paths",
            enable_spec_cache=enable_cache,
            fan_out=2,
        )
        layers = cfg.resolve_eagle_layers(self.n_layers)
        torch.manual_seed(7)
        draft = build_eagle3_sgl_draft(self.target, layers, num_layers=1)
        fn = make_eagle3_sgl_draft_topk_fn(draft, topk=cfg.topk)
        if enable_cache:
            return SsdEagle3SglGenerator(self.target, cfg, fn), fn
        return Eagle3SglGenerator(self.target, cfg, fn), fn

    def test_lossless_and_cache_hits(self) -> None:
        input_ids = torch.tensor([[1, 7, 13, 21, 30]], dtype=torch.long)
        max_new = 16

        with torch.inference_mode():
            base_greedy = self.target.generate(
                input_ids, max_new_tokens=max_new, do_sample=False, num_beams=1,
                use_cache=True, pad_token_id=self.target.config.eos_token_id,
            )
        greedy_cont = base_greedy[0, input_ids.shape[1]:]

        base_gen, fn = self._make(enable_cache=False)
        out_base = base_gen.generate(input_ids, max_new_tokens=max_new)

        ssd_gen, _ = self._make(enable_cache=True)
        out_ssd = ssd_gen.generate(input_ids, max_new_tokens=max_new)

        gc = greedy_cont.tolist()
        bc = out_base[0, input_ids.shape[1]:].tolist()
        sc = out_ssd[0, input_ids.shape[1]:].tolist()

        m = min(len(gc), len(bc), len(sc))
        self.assertGreater(m, 0)
        # (a) 基座树投机无损 vs 贪心
        self.assertEqual(bc[:m], gc[:m], f"base vs greedy diverged\n base={bc[:m]}\n grdy={gc[:m]}")
        # (b) SSD 缓存版无损 == 基座
        self.assertEqual(sc[:m], bc[:m], f"ssd vs base diverged\n ssd={sc[:m]}\n base={bc[:m]}")
        # (c) cache 确有命中（greedy 下第 0 轮 miss，之后应稳定命中）
        self.assertGreater(ssd_gen.n_cache_hit, 0, "expected at least one speculation-cache hit")


if __name__ == "__main__":
    unittest.main()
