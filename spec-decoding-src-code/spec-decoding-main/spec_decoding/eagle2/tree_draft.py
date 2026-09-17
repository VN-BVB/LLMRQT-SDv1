# SPDX-License-Identifier: Apache-2.0
"""
EAGLE2 要素之二：Top-k > 1 → 树形 Draft（多分支并行猜）

链式投机（topk=1）每步只保留 一个 下一 token；EAGLE2 每步保留 K 个，
用 累计 log 概率 做 beam 剪枝，再输出parent / token_ids 供 verify。

draft_topk_fn：(target last-token hidden, 父节点 token id) → topk ids + log 分数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Eagle2TreeConfig


@dataclass
class TreeDraftResult:
    """展开并剪枝后的树，摊平为 TreeVerifyLayout 所需字段。"""

    token_ids: List[int]
    parent: List[int]
    node_depth: List[int]
    cum_log_probs: List[float]
    """每个节点的累计 log 概率（根层为单步 log p）。"""
    bfs_index: List[int]
    """节点创建顺序（调试用）。"""


DraftTopkFn = Callable[
    [torch.Tensor, torch.LongTensor],
    Tuple[torch.Tensor, torch.Tensor],
]
"""(hidden[B,H], parent_token_id[B,1]) -> (topk_ids[B,K], topk_log_probs[B,K])"""


@dataclass
class _FrontierNode: # 用来保存topk node的每个node信息
    tree_idx: int
    cum_logp: float
    hidden: torch.Tensor
    path: List[int]


def _topk_log_probs(
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    k: int,
) -> Tuple[List[int], List[float]]:
    """
    将 draft_topk_fn 返回的分数规范为 log 概率。

    若 topk_scores 已是 log-softmax 采样值，直接使用；否则对
    topk_scores 在 top-k 维上做 log_softmax。
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


def _expand_hidden(
    hidden: torch.Tensor,
    path: List[int],
    *,
    target_hidden_refresh: Optional[Callable[[torch.LongTensor], torch.Tensor]],
    initial_prefix_ids: Optional[torch.LongTensor],
) -> torch.Tensor:
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


def expand_draft_tree_topk(
    cfg: Eagle2TreeConfig,
    root_hidden: torch.Tensor,
    root_parent_token_id: torch.LongTensor,
    draft_topk_fn: DraftTopkFn,
    *,
    target_hidden_refresh: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
    initial_prefix_ids: Optional[torch.LongTensor] = None,
) -> TreeDraftResult:
    """
    EAGLE-2 风格 draft 树：每层累计 log 概率 + 全局 top-K beam，最后再按分数截断。
    Parameters
    ----------
    root_hidden
        Target 在 当前前缀最后一个 token 处的 hidden，[1, H]。
    root_parent_token_id
        当前序列 最后一个已生成 token id，[1, 1]。
    draft_topk_fn
        见 DraftTopkFn。
    target_hidden_refresh
        若给定：输入 prefix_ids [1,L]，返回该前缀末 token 的 target hidden [1,H]。
    initial_prefix_ids
        与 target_hidden_refresh 联用：当前完整 input_ids（用于刷新子节点 hidden）。
    与官方 topK_genrate 对齐的要点：

    - 每一步扩展：cum_logp(child) = cum_logp(parent) + log p(child|parent)；
    - 每步只保留累计分最高的 topk 条 活跃路径（beam）；
    - 全部步结束后，若节点数超过 max_tree_nodes，按累计分做全局剪枝（保留祖先）。
    """
    # 同时最多保留 K 条活跃路径（也限制根层分支数）
    K = min(cfg.topk, cfg.max_tree_nodes) if cfg.max_tree_nodes > 0 else cfg.topk
    if K < 1:
        raise ValueError("topk or max_tree_nodes invalid")

    device = root_hidden.device
    # 树摊平为数组：token_ids[i] 是节点 i 的 draft token；parent[i] 是父节点下标（根为 -1）
    token_ids: List[int] = []
    parent: List[int] = []
    node_depth: List[int] = []       # 从根算起的路径长度（根层节点 depth=1）
    cum_log_probs: List[float] = []  # 根→该节点的累计 log 概率
    bfs_ix: List[int] = []           # 节点创建顺序（剪枝后会重编号）

    # 对前缀末 token draft 一次，取 log 概率最高的 K 个下一 token 作为根节点
    topk_ids, topk_scores = draft_topk_fn(root_hidden, root_parent_token_id)
    root_toks, root_lps = _topk_log_probs(topk_ids, topk_scores, K)

    # frontier = 当前 beam 上的活跃叶节点，类似于先保存expand阶段的所有topk node到frontier
    frontier: List[_FrontierNode] = []
    for t, lp in zip(root_toks, root_lps):
        idx = len(token_ids)
        token_ids.append(t)
        parent.append(-1)
        node_depth.append(1)
        cum_log_probs.append(lp)     # 树的每一层的累计分数 = 单步 log p(token|prefix)
        bfs_ix.append(idx)
        path = [t]                   # 从根到该节点的 token 序列（供 refresh hidden 用）
        h = _expand_hidden(          # 若开启 refresh：用 prefix+path 重算 target hidden；否则沿用根 hidden
            root_hidden,
            path,
            target_hidden_refresh=target_hidden_refresh,
            initial_prefix_ids=initial_prefix_ids,
        )
        frontier.append(_FrontierNode(idx, lp, h, path))

    # ── 第 1..num_steps-1 步：对 frontier 上每条路径各 draft 一步，再按累计分 beam 剪枝 ──
    for _step in range(1, cfg.num_steps):
        if not frontier:
            break
        candidates: List[_FrontierNode] = []  # 本层所有父节点展开出的子节点候选
        for node in frontier:
            if node_depth[node.tree_idx] >= cfg.num_steps:
                continue  # 已达最大深度，不再扩展
            tid = torch.tensor([[node.path[-1]]], device=device, dtype=torch.long)
            # draft forward，得到每个topk node的子节点logits
            child_ids, child_scores = draft_topk_fn(node.hidden, tid)
            ctoks, clps = _topk_log_probs(child_ids, child_scores, K)
            # 遍历之前每个topk node的child node，保留新累积概率分数、新的target末token hidden以及path
            for t, dlp in zip(ctoks, clps):
                new_lp = node.cum_logp + dlp  # 累计 log 概率：log p(path) = Σ log p(token|parent)
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

        # beam 剪枝：全层候选按累计分排序，只保留 top-K 条路径进入下一轮 frontier
        candidates.sort(key=lambda n: n.cum_logp, reverse=True)
        # 每个step生成了新的topk node后经过排序后的新topk
        frontier = candidates[:K]

    # ── 全局剪枝：beam 过程中可能产生 > max_tree_nodes 的节点，按 cum_log_probs 保留高分节点及其祖先 ──
    token_ids, parent, node_depth, cum_log_probs = _compact_tree_by_score(
        token_ids, parent, node_depth, cum_log_probs, cfg.max_tree_nodes
    )
    bfs_ix = list(range(len(token_ids)))  # 剪枝后按新下标 0..L-1 顺序编号

    return TreeDraftResult(
        token_ids=token_ids,
        parent=parent,
        node_depth=node_depth,
        cum_log_probs=cum_log_probs,
        bfs_index=bfs_ix,
    )


def expand_draft_tree_bfs(
    cfg: Eagle2TreeConfig,
    root_hidden: torch.Tensor,
    root_parent_token_id: torch.LongTensor,
    draft_topk_fn: DraftTopkFn,
    *,
    target_hidden_refresh: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
    initial_prefix_ids: Optional[torch.LongTensor] = None,
) -> TreeDraftResult:
    """无累计剪枝的 BFS 展开（对照 / 调试， eagle1版本）。"""
    K = min(cfg.topk, cfg.max_tree_nodes - 1) if cfg.max_tree_nodes > 0 else cfg.topk
    if K < 1:
        raise ValueError("topk or max_tree_nodes invalid")

    token_ids: List[int] = []
    parent: List[int] = []
    depth: List[int] = []
    cum_log_probs: List[float] = []
    bfs_ix: List[int] = []

    topk_ids, topk_scores = draft_topk_fn(root_hidden, root_parent_token_id)
    root_toks, root_lps = _topk_log_probs(topk_ids, topk_scores, K)

    from collections import deque

    q: deque = deque()
    for t, lp in zip(root_toks, root_lps):
        if len(token_ids) >= cfg.max_tree_nodes:
            break
        idx = len(token_ids)
        token_ids.append(t)
        parent.append(-1)
        depth.append(1)
        cum_log_probs.append(lp)
        bfs_ix.append(idx)
        path = [t]
        q.append((idx, 1, int(t), root_hidden, path, lp))

    while q and len(token_ids) < cfg.max_tree_nodes:
        pidx, d, _ptok, ph, path_p, p_lp = q.popleft()
        if d >= cfg.num_steps:
            continue
        tid = torch.tensor([[path_p[-1]]], device=root_hidden.device, dtype=torch.long)
        h = _expand_hidden(
            ph,
            path_p,
            target_hidden_refresh=target_hidden_refresh,
            initial_prefix_ids=initial_prefix_ids,
        )
        child_ids, child_scores = draft_topk_fn(h, tid)
        ctoks, clps = _topk_log_probs(child_ids, child_scores, K)
        for ct, dlp in zip(ctoks, clps):
            if len(token_ids) >= cfg.max_tree_nodes:
                break
            cidx = len(token_ids)
            token_ids.append(ct)
            parent.append(pidx)
            depth.append(d + 1)
            cum_log_probs.append(p_lp + dlp)
            bfs_ix.append(cidx)
            path_c = path_p + [ct]
            q.append((cidx, d + 1, ct, h, path_c, p_lp + dlp))

    return TreeDraftResult(
        token_ids=token_ids,
        parent=parent,
        node_depth=depth,
        cum_log_probs=cum_log_probs,
        bfs_index=bfs_ix,
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
