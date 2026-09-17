# SPDX-License-Identifier: Apache-2.0
"""
Top-k > 1 → 静态树形 Draft（多分支并行猜）

链式投机（topk=1）每步只保留 一个 下一 token；EAGLE1 每步保留 K 个 子节点，
按 BFS 顺序 展开成一棵 静态树，直到深度 num_steps 或节点数达到
max_tree_nodes。

树形状只由 topk / num_steps / max_tree_nodes 这几个超参静态决定。

draft_topk_fn：(target last-token hidden, 父节点 token id) → topk ids + 分数。
分数（第二个返回值）在 EAGLE1 中不参与建树，仅为与 EAGLE2 接口兼容而保留。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from .config import Eagle1Config


@dataclass
class TreeDraftResult:
    """静态展开后的树，摊平为 TreeVerifyLayout 所需字段。"""

    token_ids: List[int]
    parent: List[int]
    node_depth: List[int]
    bfs_index: List[int]
    """节点创建顺序（调试用）。"""


DraftTopkFn = Callable[
    [torch.Tensor, torch.LongTensor],
    Tuple[torch.Tensor, torch.Tensor],
]
"""(hidden[B,H], parent_token_id[B,1]) -> (topk_ids[B,K], topk_scores[B,K])

EAGLE1 只用 topk_ids；topk_scores 不参与建树。
"""


def _topk_ids(topk_ids: torch.Tensor, k: int) -> List[int]:
    """取 draft_topk_fn 返回的前 k 个候选 token id（忽略分数）。"""
    return [int(t) for t in topk_ids[0, :k].tolist()]


def _expand_hidden(
    hidden: torch.Tensor,
    path: List[int],
    *,
    target_hidden_refresh: Optional[Callable[[torch.LongTensor], torch.Tensor]],
    initial_prefix_ids: Optional[torch.LongTensor],
) -> torch.Tensor:
    """扩展子节点时的 draft 条件 hidden。

    若提供了 target_hidden_refresh：用前缀 + 当前路径重新前向 target，取末
    token 的真实 hidden（正确性优先，接受率更高）；否则沿用父节点 hidden（更省算力）。
    """
    if target_hidden_refresh is not None and initial_prefix_ids is not None:
        device = hidden.device
        ext = torch.cat(
            [
                initial_prefix_ids,
                torch.tensor([path], device=device, dtype=torch.long),
            ],
            dim=1,
        )
        return target_hidden_refresh(ext)
    return hidden


def expand_draft_tree(
    cfg: Eagle1Config,
    root_hidden: torch.Tensor,
    root_parent_token_id: torch.LongTensor,
    draft_topk_fn: DraftTopkFn,
    *,
    target_hidden_refresh: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
    initial_prefix_ids: Optional[torch.LongTensor] = None,
) -> TreeDraftResult:
    """
    EAGLE1 静态 top-k 树：每个节点固定展开 top-k 子节点，按 BFS 顺序填到
    max_tree_nodes 为止。

    Parameters
    ----------
    root_hidden
        Target 在 当前前缀最后一个 token 处的 hidden，[1, H]。
    root_parent_token_id
        当前序列 最后一个已生成 token id，[1, 1]。
    draft_topk_fn
        见 DraftTopkFn（EAGLE1 仅使用其 token id 输出）。
    target_hidden_refresh / initial_prefix_ids
        若给定：扩展子节点时用前缀 + 路径刷新 target hidden。
    """
    K = cfg.topk
    if K < 1:
        raise ValueError("topk invalid")
    if cfg.max_tree_nodes < 1:
        raise ValueError("max_tree_nodes invalid")

    device = root_hidden.device
    token_ids: List[int] = []
    parent: List[int] = []
    depth: List[int] = []
    bfs_ix: List[int] = []
    # root_parent_token_id = is,对其作topk spec
    # root_toks = [Paris, the]
    topk_ids, _ = draft_topk_fn(root_hidden, root_parent_token_id)
    root_toks = _topk_ids(topk_ids, K)

    # 队列元素：(tree_idx, depth, hidden, path)
    # 设 cfg.topk=2、num_steps=3、max_tree_nodes=16，前缀是 “The capital of France is”，根 token = “is”。
    # token_ids = [Paris, the, ., ,, capital, city, .., .., .., .., .., .., .., ..]
    # parent    = [ -1,  -1,  0, 0,   1,    1,   2,  2,  3,  3,  4,  4,  5,  5 ]
    # node_depth= [  1,   1,  2, 2,   2,    2,   3,  3,  3,  3,  3,  3,  3,  3 ]
    # bfs_index = [  0,   1,  2, 3,   4,    5,   6,  7,  8,  9, 10, 11, 12, 13 ]
    # 把paris和the这俩由prompt末端token生成的topK添加到队列
    # q=((0,1,root hidden,[Paris]),(1,1,root hidden,[the]))
    # parent=(-1,-1)
    q: deque = deque()
    for t in root_toks:
        if len(token_ids) >= cfg.max_tree_nodes:
            break
        idx = len(token_ids)
        token_ids.append(t)
        parent.append(-1)
        depth.append(1)
        bfs_ix.append(idx)
        q.append((idx, 1, root_hidden, [t]))

    while q and len(token_ids) < cfg.max_tree_nodes:
        pidx, d, ph, path_p = q.popleft()
        if d >= cfg.num_steps:
            continue
        h = _expand_hidden(
            ph,
            path_p,
            target_hidden_refresh=target_hidden_refresh,
            initial_prefix_ids=initial_prefix_ids,
        )
        tid = torch.tensor([[path_p[-1]]], device=device, dtype=torch.long)
        # 对paris作spec K，对the作spec K
        child_ids, _ = draft_topk_fn(h, tid)
        # 对paris和the生成的topk添加在q里，对该topk继续bfs生成下一个topk，直到超过了num steps或max tree nodes
        for ct in _topk_ids(child_ids, K):
            if len(token_ids) >= cfg.max_tree_nodes:
                break
            cidx = len(token_ids)
            token_ids.append(ct)
            parent.append(pidx)
            depth.append(d + 1)
            bfs_ix.append(cidx)
            q.append((cidx, d + 1, h, path_p + [ct]))
    # token_ids = [Paris, the, ., ,, capital, city, .., .., .., .., .., .., .., ..]
    # parent    = [ -1,  -1,  0, 0,   1,    1,   2,  2,  3,  3,  4,  4,  5,  5 ]
    # node_depth= [  1,   1,  2, 2,   2,    2,   3,  3,  3,  3,  3,  3,  3,  3 ]
    # bfs_index = [  0,   1,  2, 3,   4,    5,   6,  7,  8,  9, 10, 11, 12, 13 ]
    return TreeDraftResult(
        token_ids=token_ids, # bfs式地保存了树里所有token id
        parent=parent, # bfs式地保存了树里所有token的parent id
        node_depth=depth,
        bfs_index=bfs_ix, # bfs式地保存了树里所有token的创建顺序
    )


def tree_draft_to_verify_layout(
    draft: TreeDraftResult,
) -> "TreeVerifyLayout":
    from .tree_verify_mask import TreeVerifyLayout

    n = len(draft.token_ids)
    return TreeVerifyLayout(
        token_ids=list(draft.token_ids),
        parent=list(draft.parent),
        dfs_order=list(range(n)),
    )
