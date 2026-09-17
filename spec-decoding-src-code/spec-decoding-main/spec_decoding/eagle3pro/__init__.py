# SPDX-License-Identifier: Apache-2.0
"""
spec_decoding.eagle3pro —— 从只读 eagle3_sgl 基线复制出的独立优化工作包。

EAGLE3PRO PATH：调用方必须从 ``spec_decoding.eagle3pro`` 导入；本目录的实验和
性能修改不会回写 ``spec_decoding.eagle3_sgl`` 或它的日期备份。

= EAGLE1 的静态 top-k 树 + full_model_tree/reference_paths tree verify+ EAGLE3 的target 低/中/高多层 hidden 拼接作为 draft 条件特征。

相对 eagle(EAGLE1) 的唯一区别在 feature 侧：
- draft 条件特征 = target 多层 act 拼接（eagle_layers），经 fc((1+n)H→H) 融合，
  而非单层 hidden（fc(2H→H)）；
- 取特征用 hidden.forward_target_eagle_acts，配置用 eagle_layers / resolve_eagle_layers。

树与 verify 与 eagle 完全一致（topk 静态展开 + 树掩码一次性 verify / 逐叶 verify）。

外层调度：runner.Eagle3SglGenerator。
draft：draft_model.build_eagle3_sgl_draft + make_eagle3_sgl_draft_topk_fn（轻量、复用
target embedding/lm_head + fc + 1 层 decoder、首层跳过 input_layernorm）。
"""

from __future__ import annotations

from typing import Any

from .config import Eagle3SglConfig
from .tree_draft import (
    TreeDraftResult,
    expand_draft_tree,
    expand_draft_tree_ar,
    expand_draft_tree_topk,
    tree_draft_to_verify_layout,
)
from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
    build_tree_sdpa_attn_bias,
)

__all__ = [
    "Eagle3SglConfig",
    "Eagle3SglGenerator",
    "Eagle3SglDraftModel",
    "build_eagle3_sgl_draft",
    "make_eagle3_sgl_draft_topk_fn",
    "load_eagle3_draft_weights",
    "TreeDraftResult",
    "TreeVerifyLayout",
    "build_tree_cross_attn_bias_with_prefix",
    "build_tree_sdpa_attn_bias",
    "expand_draft_tree",
    "expand_draft_tree_topk",
    "expand_draft_tree_ar",
    "forward_target_eagle_acts",
    "make_hidden_residual_draft_topk_fn",
    "full_model_tree_verify_logits",
    "select_append_from_full_tree_logits",
    "tree_verify_attention_context",
    "tree_draft_to_verify_layout",
    "resolve_verify_attn_backend",
    "flash_attn_available",
    "triton_extend_available",
    "enable_qwen3_packed_projections",
    "enable_vllm_fused_kernels",
    "Eagle3ProSSDConfig",
    "Eagle3ProSSDGenerator",
    "Eagle3ProSSDStats",
    "SSDCacheEntry",
    "SSDCacheKey",
]


def __getattr__(name: str) -> Any:
    if name == "forward_target_eagle_acts":
        from .hidden import forward_target_eagle_acts

        return forward_target_eagle_acts
    if name == "Eagle3SglGenerator":
        from .runner import Eagle3SglGenerator

        return Eagle3SglGenerator
    if name == "make_hidden_residual_draft_topk_fn":
        from .runner import make_hidden_residual_draft_topk_fn

        return make_hidden_residual_draft_topk_fn
    if name in (
        "Eagle3SglDraftModel",
        "build_eagle3_sgl_draft",
        "make_eagle3_sgl_draft_topk_fn",
        "load_eagle3_draft_weights",
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
    if name == "enable_qwen3_packed_projections":
        from .target_optim import enable_qwen3_packed_projections

        # EAGLE3PRO: 延迟导入，普通路径不承担 Qwen3 专用优化模块的启动成本。
        return enable_qwen3_packed_projections
    if name == "enable_vllm_fused_kernels":
        from .target_optim import enable_vllm_fused_kernels

        # EAGLE3PRO: vLLM 是显式可选依赖；未选择该路径时不会导入它。
        return enable_vllm_fused_kernels
    if name in (
        "Eagle3ProSSDConfig",
        "Eagle3ProSSDGenerator",
        "Eagle3ProSSDStats",
        "SSDCacheEntry",
        "SSDCacheKey",
    ):
        from . import ssd

        return getattr(ssd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
