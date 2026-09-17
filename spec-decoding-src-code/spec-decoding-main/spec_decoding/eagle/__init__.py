# SPDX-License-Identifier: Apache-2.0
"""
spec_decoding.eagle —— EAGLE1（在链式投机上补三块，但用 *静态* top-k 树）。

相对普通链式投机（spec_decoding.speculative_decoding）EAGLE1 多出：

1. Target 特征条件 Draft（feature-level）
   轻量 draft 复用 target 的 embedding/lm_head，输入是 fc(concat(token_emb, target_hidden))，
   通常 1 层 decoder 且首层跳过 input_layernorm。实现：draft_model.Eagle1DraftModel /
   build_eagle1_draft / make_eagle1_draft_topk_fn。

2. Top-k > 1 → 静态树形候选
   每个节点固定展开 top-k 子节点（tree_draft.expand_draft_tree）。
   与 EAGLE2 的区别：不做累计概率 beam / 全局按分剪枝——树形状只由
   topk / num_steps / max_tree_nodes 静态决定。

3. 专用 Verify Attention（树掩码）
   一次 target forward 验证整棵树，树掩码保证兄弟分支互不泄漏。
   实现：tree_verify_mask / tree_verify_full 等。

外层调度：runner.Eagle1Generator。
"""

from __future__ import annotations

from typing import Any

from .config import Eagle1Config
from .tree_draft import (
    TreeDraftResult,
    expand_draft_tree,
    tree_draft_to_verify_layout,
)
from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
    build_tree_sdpa_attn_bias,
)

__all__ = [
    "Eagle1Config",
    "Eagle1Generator",
    "Eagle1DraftModel",
    "build_eagle1_draft",
    "make_eagle1_draft_topk_fn",
    "load_eagle1_draft_weights",
    "TreeDraftResult",
    "TreeVerifyLayout",
    "build_tree_cross_attn_bias_with_prefix",
    "build_tree_sdpa_attn_bias",
    "expand_draft_tree",
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
    if name == "Eagle1Generator":
        from .runner import Eagle1Generator

        return Eagle1Generator
    if name == "make_hidden_residual_draft_topk_fn":
        from .runner import make_hidden_residual_draft_topk_fn

        return make_hidden_residual_draft_topk_fn
    if name in (
        "Eagle1DraftModel",
        "build_eagle1_draft",
        "make_eagle1_draft_topk_fn",
        "load_eagle1_draft_weights",
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
