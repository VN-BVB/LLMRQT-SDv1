# SPDX-License-Identifier: Apache-2.0
"""
树形 verify 的 叶子路径 greedy 对齐 辅助函数。

给定一次树 verify forward 得到的逐节点 logits [1, L, V]（每行对应 layout
里的一个树节点），对某条根→叶路径做与链式投机相同的 greedy 比对：逐位用
target argmax 与 draft 提案比较，返回 (match_len, next_token)。

被 tree_verify_full.select_append_from_full_tree_logits 复用，用于从单次
full_model_tree forward 的 logits 中挑出最优接受路径。
"""

from __future__ import annotations

from typing import List, Tuple

import torch  # noqa: F401  (用于类型注解 torch.Tensor)


def path_node_indices_root_to_leaf(parent: List[int], leaf: int) -> List[int]:
    """从根到 leaf 的节点下标（含叶），根在 parent[i]==-1 的链上。"""
    out: List[int] = []
    k = leaf
    while k >= 0:
        out.append(k)
        k = parent[k]
    out.reverse()  # 回溯是叶→根，反转成根→叶
    return out


def verify_leaf_path_against_parallel_logits(
    logits: torch.Tensor,
    parent: List[int],
    token_ids: List[int],
    leaf: int,
) -> Tuple[int, int]:
    """
    用树 verify 的 [1,L,V] 节点 logits，对一条根到叶路径做 greedy 对齐。

    - 路径上第 j 个节点（下标 node_idx）的 target greedy = argmax(logits[0, node_idx])；
    - 与 draft 提案 token_ids[node_idx] 比较，遇首个不一致即停（返回该处 target token）；
    - 全中则返回 (len, bonus)，bonus 为叶子节点处的 argmax（再多走一步）。

    Returns (match_len, next_token)。
    """
    indices = path_node_indices_root_to_leaf(parent, leaf)
    if not indices:
        return 0, 0
    path_toks = [token_ids[i] for i in indices]
    for j, node_idx in enumerate(indices):
        pred = int(logits[0, node_idx, :].argmax(dim=-1).item())
        if pred != path_toks[j]:
            return j, pred  # 首个不匹配位：接受前 j 个，纠正为 target 的 pred
    last = indices[-1]
    bonus = int(logits[0, last, :].argmax(dim=-1).item())  # 全中 -> bonus
    return len(indices), bonus
