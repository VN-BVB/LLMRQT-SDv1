# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3-SGL 超参：EAGLE3 的多层 act 特征 + EAGLE1 的静态 top-k 树 + 树掩码 verify。

相对 eagle(EAGLE1)：
- draft 的条件特征改为 target 的 低/中/高 多层 hidden 拼接（eagle_layers），而非单层 hidden；
- 因此用 eagle_layers / resolve_eagle_layers 取代 EAGLE1 的 hidden_layer_index。

树与 verify（topk/num_steps/max_tree_nodes/verify_mode/verify_attn_backend）与 EAGLE1 一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


def default_eagle_layers(num_hidden_layers: int) -> List[int]:
    """默认 EAGLE3 特征层（早/中/晚三层）。"""
    L = num_hidden_layers
    return [2, L // 2, L - 3]


@dataclass
class Eagle3SglConfig:
    """EAGLE3-SGL 树投机超参。"""

    topk: int = 8
    """每层 draft 展开的分支数（>1 时形成树）。"""
    num_steps: int = 8
    """从根出发的 draft 深度。"""
    max_tree_nodes: int = 64
    """draft 节点上限：cumulative 模式按累计分剪枝、static 模式按 BFS 顺序截断。"""
    tree_expand_mode: str = "cumulative"
    """
    cumulative（默认，EAGLE-2 能力）：每层累计 log 概率 + 全局 top-K beam，末尾按累计分剪枝，
        在相同节点预算下把算力集中到 draft 更有把握的分支（接受率更高）。
    static：EAGLE-1 静态 top-k BFS 树（不看分数、按 BFS 顺序填满 max_tree_nodes）。
    """
    eagle_layers: Optional[List[int]] = None
    """作为 draft 条件的 target 多层 hidden 下标；None 时用 default_eagle_layers。"""
    verify_mode: str = "full_model_tree"
    """
    full_model_tree（推荐）：一次 全层 target forward + 树掩码（tree_verify_full）。
    reference_paths：对每个叶子路径用标准 causal forward 做贪心串行校验（慢、作对照）。
    """
    verify_attn_backend: str = "auto"
    """
    full_model_tree 时 target attention 后端：

    - auto：topk==1 且可用 → flash_attn；topk>1 且 Triton 可用 → triton_tree；否则 eager。
    - flash_attn：HF flash_attention_2（链式因果，不叠树 mask）。
    - triton_tree：SGLang extend_attention_fwd + custom_mask。
    - eager：target_m.forward + 树 mask patch + 临时 eager。
    """
    decode_cuda_graph: bool = False
    """Step5/6：target decode 走 static KV 缓存 + 固定 shape masked SDPA，并对固定
    q_len 的 verify 前向做 CUDA graph 捕获/replay（暂不做 paged KV/融合注意力）。
    仅 full_model_tree + autoregressive_draft 支持。"""
    graph_capture: bool = True
    """decode_cuda_graph 下是否真正捕获/replay CUDA graph；False 时走同一 static KV + SDPA
    eager 执行"""
    eliminate_replay: bool = True
    """Step6：让 target KV 落后一个 pending token，用同一次固定形状 verify forward
    提交已接受 KV，取消每轮 target replay。topk=1 直接 crop 逻辑长度；多分支树把
    非连续 accepted-path KV gather/compact 到连续正式 slots 后再推进逻辑长度。"""
    graph_cache_margin: int = 16
    """static KV 缓存额外预留长度（容纳 verify 临时写入的 L 个树节点 slot 等）。"""

    def __post_init__(self) -> None:
        """校验超参合法性（topk/num_steps >= 1，verify_mode / backend 取值在白名单内）。"""
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.tree_expand_mode not in ("cumulative", "static"):
            raise ValueError("tree_expand_mode must be cumulative or static")
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

    def resolve_eagle_layers(self, num_hidden_layers: int) -> List[int]:
        """返回配置的或默认的 eagle_layers（并校验下标在 [0, num_hidden_layers) 内）。"""
        if self.eagle_layers is not None:
            for idx in self.eagle_layers:
                if idx < 0 or idx >= num_hidden_layers:
                    raise ValueError(
                        f"eagle_layers index {idx} out of range for num_hidden_layers={num_hidden_layers}"
                    )
            return list(self.eagle_layers)
        return default_eagle_layers(num_hidden_layers)
