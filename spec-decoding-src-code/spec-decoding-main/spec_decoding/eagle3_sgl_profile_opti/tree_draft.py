# SPDX-License-Identifier: Apache-2.0
"""
与eagle1大致相同，没有应用eagle2的树剪枝

EAGLE1 要素：Top-k > 1 → 静态树形 Draft（多分支并行猜）

链式投机（topk=1）每步只保留 一个 下一 token；EAGLE1 每步保留 K 个 子节点，
按 BFS 顺序 展开成一棵 静态树，直到深度 num_steps 或节点数达到
max_tree_nodes。

树形状只由 topk / num_steps / max_tree_nodes 这几个超参静态决定。

draft_topk_fn：(target last-token hidden, 父节点 token id) → topk ids + 分数。
分数（第二个返回值）在 EAGLE1 中不参与建树，仅为与 EAGLE2 接口兼容而保留。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Eagle3SglConfig


@dataclass
class TreeDraftResult:
    """展开后的树，摊平为 TreeVerifyLayout 所需字段。"""

    token_ids: List[int]
    parent: List[int]
    node_depth: List[int]
    bfs_index: List[int]
    """节点创建顺序（调试用）。"""
    cum_log_probs: List[float] = field(default_factory=list)
    """每个节点的累计 log 概率（cumulative 模式用；static 模式留空）。"""


DraftTopkFn = Callable[
    [torch.Tensor, torch.LongTensor],
    Tuple[torch.Tensor, torch.Tensor],
]
"""(hidden[B,H], parent_token_id[B,1]) -> (topk_ids[B,K], topk_scores[B,K])

EAGLE1 只用 topk_ids；topk_scores 不参与建树, 其将在eagle2用到。
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
    cfg: Eagle3SglConfig,
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


# =============================================================================
# EAGLE-2 风格 累计概率剪枝（cumulative beam）—— 相对上面的 EAGLE1 静态树多出来的能力：
# 每层把 cum_logp(child) = cum_logp(parent) + log p(child|parent) 作为分数，只保留全局
# 累计分最高的 K 条活跃路径（beam）；展开完再按累计分把节点数压到 max_tree_nodes（保留祖先，
# 保证 parent 链有效）。这样在相同节点预算下把算力集中到draft 更有把握的分支，接受率更高。
# =============================================================================
@dataclass
class _FrontierNode:
    tree_idx: int
    cum_logp: float
    hidden: torch.Tensor
    path: List[int]


def _topk_log_probs(
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    k: int,
) -> Tuple[List[int], List[float]]:
    """把 draft_topk_fn 返回的分数规范为 log 概率。

    EAGLE3-SGL 的 make_eagle3_sgl_draft_topk_fn 返回的是 原始 logits 的 top-k 分数，
    故 max>0 时对这 k 个分数做 log_softmax（在 sibling 子集上归一化，得到可累加的条件 log 概率）；
    若分数已是 log-softmax 值（<=0）则直接使用（兼容返回 log 概率的 topk_fn）。
    """
    ids = topk_ids[0, :k].tolist()
    sc = topk_scores[0, :k].float()
    if sc.numel() == 0:
        return [], []
    if sc.max() > 0:
        logps = F.log_softmax(sc, dim=-1).tolist()
    else:
        logps = sc.tolist()
    return [int(t) for t in ids], [float(x) for x in logps]


def _compact_tree_by_score(
    token_ids: List[int],
    parent: List[int],
    node_depth: List[int],
    cum_log_probs: List[float],
    max_nodes: int,
) -> Tuple[List[int], List[int], List[int], List[float]]:
    """按累计分保留至多 max_nodes 个节点，并带上其祖先（保证 parent 链有效）。"""
    n = len(token_ids)
    if n <= max_nodes:
        return token_ids, parent, node_depth, cum_log_probs

    ranked = sorted(range(n), key=lambda i: cum_log_probs[i], reverse=True)
    keep: set[int] = set()
    for i in ranked[:max_nodes]:
        j = i
        while j >= 0:
            keep.add(j)
            j = parent[j]

    order = sorted(keep)
    old_to_new = {old: new for new, old in enumerate(order)}
    return (
        [token_ids[i] for i in order],
        [old_to_new[parent[i]] if parent[i] >= 0 else -1 for i in order],
        [node_depth[i] for i in order],
        [cum_log_probs[i] for i in order],
    )


def expand_draft_tree_topk(
    cfg: Eagle3SglConfig,
    root_hidden: torch.Tensor,
    root_parent_token_id: torch.LongTensor,
    draft_topk_fn: DraftTopkFn,
    *,
    target_hidden_refresh: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
    initial_prefix_ids: Optional[torch.LongTensor] = None,
) -> TreeDraftResult:
    """
    EAGLE-2 风格 draft 树：每层累计 log 概率 + 全局 top-K beam，最后再按分数截断。

    Parameters 同 expand_draft_tree。要点：

    - 每步扩展：cum_logp(child) = cum_logp(parent) + log p(child|parent)；
    - 每步只保留累计分最高的 topk 条活跃路径（beam）；
    - 全部步结束后，若节点数超过 max_tree_nodes，按累计分做全局剪枝（保留祖先）。

    root_hidden 此处即 EAGLE3 的多层 act 特征（[1, n*H]）；draft_topk_fn 内部会用它做条件化。
    """
    K = min(cfg.topk, cfg.max_tree_nodes) if cfg.max_tree_nodes > 0 else cfg.topk
    if K < 1:
        raise ValueError("topk or max_tree_nodes invalid")

    device = root_hidden.device
    token_ids: List[int] = []
    parent: List[int] = []
    node_depth: List[int] = []
    cum_log_probs: List[float] = []
    bfs_ix: List[int] = []

    topk_ids, topk_scores = draft_topk_fn(root_hidden, root_parent_token_id)
    root_toks, root_lps = _topk_log_probs(topk_ids, topk_scores, K)

    frontier: List[_FrontierNode] = []
    for t, lp in zip(root_toks, root_lps):
        idx = len(token_ids)
        token_ids.append(t)
        parent.append(-1)
        node_depth.append(1)
        cum_log_probs.append(lp)
        bfs_ix.append(idx)
        path = [t]
        h = _expand_hidden(
            root_hidden,
            path,
            target_hidden_refresh=target_hidden_refresh,
            initial_prefix_ids=initial_prefix_ids,
        )
        frontier.append(_FrontierNode(idx, lp, h, path))

    for _step in range(1, cfg.num_steps):
        if not frontier:
            break
        candidates: List[_FrontierNode] = []
        for node in frontier:
            if node_depth[node.tree_idx] >= cfg.num_steps:
                continue
            tid = torch.tensor([[node.path[-1]]], device=device, dtype=torch.long)
            child_ids, child_scores = draft_topk_fn(node.hidden, tid)
            ctoks, clps = _topk_log_probs(child_ids, child_scores, K)
            for t, dlp in zip(ctoks, clps):
                new_lp = node.cum_logp + dlp
                path_c = node.path + [t]
                h = _expand_hidden(
                    node.hidden,
                    path_c,
                    target_hidden_refresh=target_hidden_refresh,
                    initial_prefix_ids=initial_prefix_ids,
                )
                idx = len(token_ids)
                token_ids.append(t)
                parent.append(node.tree_idx)
                node_depth.append(node_depth[node.tree_idx] + 1)
                cum_log_probs.append(new_lp)
                bfs_ix.append(idx)
                candidates.append(_FrontierNode(idx, new_lp, h, path_c))

        candidates.sort(key=lambda n: n.cum_logp, reverse=True)
        frontier = candidates[:K]

    token_ids, parent, node_depth, cum_log_probs = _compact_tree_by_score(
        token_ids, parent, node_depth, cum_log_probs, cfg.max_tree_nodes
    )
    bfs_ix = list(range(len(token_ids)))

    return TreeDraftResult(
        token_ids=token_ids,
        parent=parent,
        node_depth=node_depth,
        bfs_index=bfs_ix,
        cum_log_probs=cum_log_probs,
    )


def expand_draft_tree_ar(
    cfg: Eagle3SglConfig,
    draft_model,
    root_logits: torch.Tensor,
    root_hidden: torch.Tensor,
    root_kv,
    past_len: int,
    *,
    device: torch.device,
) -> TreeDraftResult:
    """自回归（KV cache + RoPE + feature 递归）静态 top-k 树展开。

    逐层批量版（Step 2）：每个 BFS 深度的所有节点合成一个 batch，做一次 draft.step
    前向（替代每节点一次微型前向）；child token 全程保持 GPU 张量在层间传递，最后一次
    .tolist() 物化整棵树的 token（替代每节点 .tolist() + 每节点 torch.tensor(...,device=)）。
    与原逐节点 BFS 数值等价（同层各节点 batch 行间互不影响，topk 一致；树结构、parent/depth
    完全相同），因此无损。
    """
    K = cfg.topk
    max_nodes = cfg.max_tree_nodes
    if K < 1 or max_nodes < 1:
        raise ValueError("invalid topk/max_tree_nodes")

    token_id_chunks: List[torch.Tensor] = []  # 各层 token（GPU 张量），末尾一次性拼接 + tolist
    parent: List[int] = []
    depth: List[int] = []

    # ---- 第 1 层：root 直接来自 root_logits ----
    root_idx, _ = draft_model.topk_target_ids(root_logits, K)
    root_row = root_idx.reshape(-1)[:K] 
    n_roots = min(int(root_row.shape[0]), max_nodes)
    root_row = root_row[:n_roots]
    token_id_chunks.append(root_row)
    parent.extend([-1] * n_roots)
    depth.extend([1] * n_roots)

    # frontier：当前层节点的 batch 状态（token / 条件 hidden / KV / 全局节点下标）
    frontier_tokens = root_row.view(n_roots, 1)                       # [N,1]
    frontier_cond = root_hidden.expand(n_roots, -1).contiguous()      # [N,H]
    fk, fv = root_kv
    frontier_k = fk.expand(n_roots, *fk.shape[1:]).contiguous()       # [N,...]
    frontier_v = fv.expand(n_roots, *fv.shape[1:]).contiguous()
    frontier_global_idx = list(range(n_roots))

    total = n_roots
    d = 1
    while total < max_nodes and d < cfg.num_steps and frontier_tokens.shape[0] > 0:
        N = int(frontier_tokens.shape[0])
        # 一次前向展开整层（position 同层相同；与原 past_len-1+d 一致）
        logits, h, kv = draft_model.step(
            frontier_tokens, frontier_cond, (frontier_k, frontier_v), past_len - 1 + d
        )
        child_idx, _ = draft_model.topk_target_ids(logits, K)  # [N, Kc]
        Kc = int(child_idx.shape[-1])
        # 行优先展平 = BFS（parent-major, child-minor）顺序；
        child_flat = child_idx.reshape(N * Kc)
        keep = min(max_nodes - total, N * Kc)
        child_flat = child_flat[:keep]
        token_id_chunks.append(child_flat)

        parent_rows = torch.arange(keep, device=device) // Kc  # [keep]，各 child 的父行号
        pr = parent_rows.tolist()  # 每层一次 D2H
        parent.extend(frontier_global_idx[r] for r in pr)
        depth.extend([d + 1] * keep)

        # 下一层 frontier = children
        frontier_cond = h.index_select(0, parent_rows)
        frontier_k = kv[0].index_select(0, parent_rows)
        frontier_v = kv[1].index_select(0, parent_rows)
        frontier_tokens = child_flat.view(keep, 1)
        frontier_global_idx = list(range(total, total + keep))
        total += keep
        d += 1

    token_ids = torch.cat(token_id_chunks).tolist() if token_id_chunks else []
    bfs_ix = list(range(len(token_ids)))
    return TreeDraftResult(
        token_ids=token_ids, parent=parent, node_depth=depth, bfs_index=bfs_ix
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
