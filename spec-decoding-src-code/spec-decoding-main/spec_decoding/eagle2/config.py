# SPDX-License-Identifier: Apache-2.0
"""
EAGLE2 超参：把论文/系统里的树宽、树深、取哪层 hidden、怎么验落成可配项。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 完整 EAGLE2 里各字段扮演的角色（读配置时对照下面理解）
# ---------------------------------------------------------------------------
# - topk / num_steps / max_tree_nodes：控制 draft 树 的形状与上限。
# - hidden_layer_index：控制 target → draft 条件信号 来自哪一层 hidden。
# - verify_mode：控制 verify 阶段用哪个backend kernel。
# ---------------------------------------------------------------------------
@dataclass
class Eagle2TreeConfig:
    """EAGLE2 树投机相关超参。"""

    topk: int = 4
    """每层 draft 展开的分支数（>1 时形成树）。"""
    num_steps: int = 3
    """从根出发的 draft 深度（不含根上的下一步预测层数语义依 draft 定义）。"""
    max_tree_nodes: int = 32
    """剪枝后保留的 draft 节点上限（对齐官方 total_tokens 量级）。"""
    tree_expand_mode: str = "cumulative"
    """
    cumulative：EAGLE-2 累计 log 概率 beam + 全局按分剪枝（默认）。
    bfs：仅 BFS 展开、不做 beam 剪枝（调试对照）。
    """
    hidden_layer_index: int = -1
    """取 target 哪一层 hidden_states 作为 draft 条件；-1 表示最后一层。"""
    verify_mode: str = "full_model_tree"
    """
    full_model_tree（推荐）：一次 全层 target forward + 树掩码（tree_verify_full）。
    reference_paths：对每个叶子路径用标准 causal forward 做贪心串行校验（慢、作对照）。
    """
    verify_attn_backend: str = "auto"
    """
    full_model_tree 时 target attention 后端：

    - auto：topk==1 且可用 → flash_attn；topk>1 且 Triton 可用 → triton_tree；否则 eager。
    - flash_attn：HF flash_attention_2。
    - triton_tree：SGLang extend_attention_fwd + custom_mask。
    - eager：target_m.forward + 树 mask patch + 临时 eager。
    """

    def __post_init__(self) -> None:
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.tree_expand_mode not in ("cumulative", "bfs"):
            raise ValueError("tree_expand_mode must be cumulative or bfs")
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
