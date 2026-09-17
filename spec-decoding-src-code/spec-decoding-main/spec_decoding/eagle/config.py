# SPDX-License-Identifier: Apache-2.0
"""
EAGLE1 超参：树宽/树深/取哪层 hidden/怎么验。
EAGLE1 使用 静态 top-k 树，按 BFS顺序填满 max_tree_nodes 为止。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Eagle1Config:
    """EAGLE1 树投机相关超参。"""

    topk: int = 4
    """每层 draft 展开的分支数（>1 时形成树）。"""
    num_steps: int = 3
    """从根出发的 draft 深度。"""
    max_tree_nodes: int = 32
    """draft 节点上限（按 BFS 顺序截断，无按分剪枝）。"""
    hidden_layer_index: int = -1
    """取 target 哪一层 hidden_states 作为 draft 条件；-1 表示最后一层。"""
    verify_mode: str = "full_model_tree"
    """
    full_model_tree（推荐）：一次 全层 target forward + 树掩码（tree_verify_full）。
    reference_paths：对每个叶子路径用标准 causal forward 做 HF greedy 校验（速度慢、仅作对照）。
    """
    verify_attn_backend: str = "auto"
    """
    full_model_tree 时 target attention 后端：

    - auto：topk==1 且Flash attention可用 → flash_attn；topk>1 且 Triton 可用 → triton_tree；否则 eager。
    - flash_attn：HF flash_attention_2
    - triton_tree：SGLang extend_attention_fwd + custom_mask。
    - eager：target_m.forward + 树 mask patch + 临时 eager。
    """

    def __post_init__(self) -> None:
        """校验超参合法性（topk/num_steps >= 1，verify_mode / backend 取值在白名单内）。"""
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.verify_mode not in (
            "full_model_tree",
            "reference_paths",
        ):
            raise ValueError(
                "verify_mode must be full_model_tree or reference_paths"
            )
        if self.verify_attn_backend not in (
            "auto",
            "eager",
            "flash_attn",
            "triton_tree",
        ):
            raise ValueError(
                "verify_attn_backend must be auto, eager, flash_attn, or triton_tree"
            )
