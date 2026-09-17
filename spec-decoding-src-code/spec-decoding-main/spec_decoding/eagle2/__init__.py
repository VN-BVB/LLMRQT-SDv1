# SPDX-License-Identifier: Apache-2.0
"""
spec_decoding.eagle2 —— 完整 EAGLE2 思想（相对链式 greedy 的三块补齐）

与 eagle_speculative.speculative_greedy_decode 零交叉导入；本包可单独使用。

--------------------------------------------------------------------
EAGLE2 在工程上相对draft 只猜一条链多出什么？
--------------------------------------------------------------------

1. Target 条件 Draft（hidden）  
   小模型不只吃 token id，还吃 大模型在当前前缀末 token 上的某层 hidden
   （常见取最后一层、进 lm_head 之前），使 draft 与 target 表征空间对齐。
   实现入口：hidden.forward_target_last_hidden；draft 侧两种接法：
     - draft_model.build_eagle2_draft + make_eagle2_draft_topk_fn：轻量 EAGLE draft
       （复用 target embedding/lm_head + fc(2H→H) + 1 层 decoder，首层跳过 input_layernorm）；
     - runner.make_hidden_residual_draft_topk_fn：用外部完整模型 + proj（emb + proj(h)）。

2. Top-k > 1 → 树形候选  
   每一步 draft 输出多个子 token，形成 树 而非单链；一步可并行探索多条
   延续。实现入口：tree_draft.expand_draft_tree_topk、Eagle2TreeConfig.topk。

3. 专用 Verify Attention（树掩码）  
   若把整棵树摊成一次 target forward 里的多个 query 位置，标准 三角因果掩码
   不够：兄弟分支会互相看见，未选中路径的信息会泄漏进 logits。EAGLE2
   要求位置 i 只能 attend 自己到根路径上的祖先（外加 KV cache 里的前缀）。
   实现入口：tree_verify_mask.TreeVerifyLayout、
   tree_verify_mask.build_tree_sdpa_attn_bias（仅树内 [L,L]）；
   带前缀的并行 SDPA 掩码用 tree_verify_mask.build_tree_cross_attn_bias_with_prefix。

--------------------------------------------------------------------
本仓库里verify几条路（见 Eagle2TreeConfig.verify_mode）
--------------------------------------------------------------------

- reference_paths：对每个 叶子路径 用 HF 逐步 causal forward 与 target
  greedy 对齐（正确、易对接；多几次 forward）。
- full_model_tree：tree_verify_full 对 全层 target 一次 forward + 树掩码
  （对齐官方 EAGLE verify，推荐）。

外层调度：runner.Eagle2Generator；基类便捷入口：
BaseModelForCausalLM.generate_eagle2。
"""

from __future__ import annotations

from typing import Any

# 轻量子模块在 import 时加载；依赖 transformers 的 hidden / runner 惰性加载，
# 便于在无 HF 环境下跑 tree_verify_mask / tree_verify_parallel 的单元测试。
from .config import Eagle2TreeConfig
from .tree_draft import (
    TreeDraftResult,
    expand_draft_tree_bfs,
    expand_draft_tree_topk,
    tree_draft_to_verify_layout,
)
from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
    build_tree_sdpa_attn_bias,
)
__all__ = [
    "Eagle2TreeConfig",
    "Eagle2Generator",
    "Eagle2DraftModel",
    "build_eagle2_draft",
    "load_eagle2_draft_weights",
    "make_eagle2_draft_topk_fn",
    "TreeDraftResult",
    "TreeVerifyLayout",
    "build_tree_cross_attn_bias_with_prefix",
    "build_tree_sdpa_attn_bias",
    "expand_draft_tree_bfs",
    "expand_draft_tree_topk",
    "forward_target_last_hidden",
    "make_hidden_residual_draft_topk_fn",
    "full_model_tree_verify_logits",
    "select_append_from_full_tree_logits",
    "tree_verify_attention_context",
    "tree_draft_to_verify_layout",
    "resolve_verify_attn_backend",
    "flash_attn_available",
    "triton_extend_available",
]


def __getattr__(name: str) -> Any:
    if name == "forward_target_last_hidden":
        from .hidden import forward_target_last_hidden

        return forward_target_last_hidden
    if name == "Eagle2Generator":
        from .runner import Eagle2Generator

        return Eagle2Generator
    if name == "make_hidden_residual_draft_topk_fn":
        from .runner import make_hidden_residual_draft_topk_fn

        return make_hidden_residual_draft_topk_fn
    if name in (
        "Eagle2DraftModel",
        "build_eagle2_draft",
        "load_eagle2_draft_weights",
        "make_eagle2_draft_topk_fn",
    ):
        from . import draft_model as dm

        return getattr(dm, name)
    if name == "full_model_tree_verify_logits":
        from .tree_verify_full import full_model_tree_verify_logits

        return full_model_tree_verify_logits
    if name == "tree_verify_attention_context":
        from .tree_verify_full import tree_verify_attention_context

        return tree_verify_attention_context
    if name == "select_append_from_full_tree_logits":
        from .tree_verify_full import select_append_from_full_tree_logits

        return select_append_from_full_tree_logits
    if name in ("resolve_verify_attn_backend", "flash_attn_available", "triton_extend_available"):
        from . import tree_verify_attn as tva

        return getattr(tva, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
