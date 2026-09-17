# SPDX-License-Identifier: Apache-2.0
"""Tree Verify Attention专用的树形掩码

一次 target forward 里若要把 多个树节点 当作序列上的多个位置一起算
self-attention，不能用标准下三角 mask：那样会允许兄弟子树互相可见,
未选中的 draft 分支会污染当前位置的表示

做法：位置 i 的 Q 只能可见树意义上从根到 i 路径上的token
（含 i 自身）；KV cache 里的历史前缀照常全局可见。

- 仅 树上 L 个新节点 互看时，用 build_tree_sdpa_attn_bias（[L,L]）。
- L 个 draft query 对前缀 + L整段 K 的并行 SDPA，用
  build_tree_cross_attn_bias_with_prefix（[L, past+L]），见
  tree_verify_parallel。

这里输出 SDPA additive bias（禁止处为 -inf），与
torch.nn.functional.scaled_dot_product_attention(..., attn_mask=...) 对接。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch


# -----------------------------------------------------------------------------
# 摊平后节点 i 的语义：draft 树里的第 i 个待 verify 位置，parent[i] 指向
# 父节点下标（-1 为根）。token_ids[i] 为该位置上的 draft token。与一次 verify
# forward 里新 token 段的第 i 列对齐后即可套 build_tree_sdpa_attn_bias。
# -----------------------------------------------------------------------------
@dataclass
class TreeVerifyLayout:
    """
    一棵 draft 树摊平成 L 个 verify 位置所需的索引结构。

    token_ids/parent 与 expand_draft_tree_topk 输出一致即可；掩码只
    依赖 父子指针（祖先链），与节点在数组中的遍历顺序无强绑定。dfs_order
    预留与外部 DFS 重排对齐；当前默认 range(L) 时与 BFS 建树的节点下标一致。
    """

    token_ids: List[int]
    """摊平后第 i 个 verify 位置上的 draft token id（与 parent[i] 同属节点 i）。"""
    parent: List[int]
    """parent[i] 为节点 i 的父节点下标；-1 表示根（整段序列中第一个节点）。"""
    dfs_order: List[int]
    """通常为 range(L)；预留与外部重排对齐。"""

    def __len__(self) -> int:
        return len(self.token_ids)


def node_depth_from_root(parent: List[int], node: int) -> int:
    """Root-to-node depth (edge count); used as tree position_ids offset."""
    d = 0
    k = node
    while parent[k] >= 0:
        d += 1
        k = parent[k]
    return d


def layout_tree_position_ids(layout: TreeVerifyLayout, past_len: int) -> torch.LongTensor:
    """树 verify 用的绝对 position_ids [L]，即深度。

    Parameters
    ----------
    layout
        draft 树布局；同深度兄弟共享 depth，故 position = past_len + depth。
    past_len
        已接受前缀长度 P。

    同一 BFS 层的节点 position 相同（树注意力按深度而非 BFS 下标计 RoPE）。
    """
    depths = [node_depth_from_root(layout.parent, i) for i in range(len(layout))]
    return torch.tensor(depths, dtype=torch.long) + int(past_len)


def build_tree_allow_mask_4d(
    layout: TreeVerifyLayout,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    EAGLE-style draft×draft block: [1, 1, L, L], 1.0 = allow, 0.0 = block.
    """
    n = len(layout)
    if n == 0:
        return torch.zeros(1, 1, 0, 0, device=device, dtype=dtype)
    m = torch.zeros(1, 1, n, n, device=device, dtype=dtype)
    for i in range(n):
        for j in range(n):
            if _on_tree_path(layout.parent, j, i):
                m[0, 0, i, j] = 1.0
    return m


def _on_tree_path(parent: Sequence[int], anc: int, desc: int) -> bool:
    """若 anc 是 desc 的祖先或 anc==desc（沿 parent 指针向上走），返回 True。"""
    k = desc
    while True:
        if k == anc:
            return True
        if k < 0:
            break
        k = parent[k]
    return False


# -----------------------------------------------------------------------------
# 完整 EAGLE2 verify（单次 forward 版）的核心张量：bias[i,j]=0 当且仅当 j 在
# i 的祖先链上；否则为 -inf。拼到 past_len+L 的大 mask 时注意只覆盖新 L 列对
# 新 L 行子块（对 past 的列通常全 0 允许 attend）。
# -----------------------------------------------------------------------------
def build_tree_sdpa_attn_bias(
    layout: TreeVerifyLayout,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    构造 [L, L] 的 additive bias，满足 树因果：

    - bias[i, j] == 0：节点 i 允许 attend 节点 j；
    - bias[i, j] == -inf：禁止。

    与 torch.nn.functional.scaled_dot_product_attention(is_causal=False, attn_mask=...)
    配合：传入 attn_mask 形状 [B, H, L, L] 时需 broadcast。
    """
    n = len(layout)
    if n == 0:
        return torch.zeros(0, 0, device=device, dtype=dtype)

    neg = torch.finfo(dtype).min
    bias = torch.full((n, n), neg, device=device, dtype=dtype)
    for i in range(n):
        for j in range(n):
            if _on_tree_path(layout.parent, j, i):
                bias[i, j] = 0.0
    return bias


def build_tree_sdpa_attn_bias_4d(
    layout: TreeVerifyLayout,
    *,
    batch: int = 1,
    num_heads: int = 1,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """[B, H, L, L] 广播用 4D bias。"""
    b2 = build_tree_sdpa_attn_bias(layout, device=device, dtype=dtype)
    return b2.view(1, 1, len(layout), len(layout)).expand(batch, num_heads, len(layout), len(layout))

# 用于tree attention verify，attention_bias即表示attention_mask
def build_tree_cross_attn_bias_with_prefix(
    layout: TreeVerifyLayout,
    past_seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    并行 tree verify（SDPA）用的 cross 掩码：Q 只有树上 L 个 draft 位置，
    K/V 对应 整段 past_seq_len + L（前缀 token + 各树节点 token 与 x 对齐）。

    第 i 个 draft query（0 <= i < L）允许 attend 的 key 下标 j：

    - j < past_seq_len：前缀任意位置（整段 prompt 可见）；
    - j >= past_seq_len：令 k = j - past_seq_len，仅当 k 在树上是 i
      的祖先或 k==i 时允许（与 build_tree_sdpa_attn_bias 的 [L,L] 子块一致）。

    返回 [L, past_seq_len + L]，可直接广播到
    scaled_dot_product_attention 的 attn_mask（additive，禁止处为 -inf）。
    """
    n = len(layout)
    if n == 0:
        return torch.zeros(0, past_seq_len, device=device, dtype=dtype)

    neg = torch.finfo(dtype).min
    bias = torch.full((n, past_seq_len + n), neg, device=device, dtype=dtype)
    for i in range(n):
        for j in range(past_seq_len + n):
            if j < past_seq_len:
                bias[i, j] = 0.0
            else:
                k = j - past_seq_len
                if _on_tree_path(layout.parent, k, i):
                    bias[i, j] = 0.0
    return bias


def build_tree_cross_attn_bias_with_prefix_4d(
    layout: TreeVerifyLayout,
    past_seq_len: int,
    *,
    batch: int = 1,
    num_heads: int = 1,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """[B, H, L, past_seq_len+L] 广播用。"""
    b2 = build_tree_cross_attn_bias_with_prefix(
        layout, past_seq_len, device=device, dtype=dtype
    )
    L, t = b2.shape
    return b2.view(1, 1, L, t).expand(batch, num_heads, L, t)
