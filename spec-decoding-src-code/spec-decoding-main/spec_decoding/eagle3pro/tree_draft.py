# SPDX-License-Identifier: Apache-2.0
"""
目前保持与EAGLE1一致：Top-k > 1 → 静态树形 Draft（多分支并行猜）

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

EAGLE1 只用 topk_ids；topk_scores 不参与建树, 其在eagle2用到。
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
    """扩展子节点时的 draft 条件 hidden（非 AR 建树用）。

    Parameters
    ----------
    hidden
        父节点沿用的 act [1, n*H]（或末 token act）。
    path
        从根到当前节点的 token 路径（用于拼 initial_prefix_ids + path）。
    target_hidden_refresh
        若给定：对 prefix + path 重算 target 末 token act（接受率更高）。
    initial_prefix_ids
        真实已接受前缀 [1, P]；与 target_hidden_refresh 成对使用。

    Returns
    -------
    Tensor
        子节点扩展用的条件 hidden [1, n*H]。
    """
    if target_hidden_refresh is not None and initial_prefix_ids is not None:
        device = hidden.device
        # 用所有已有tokens + draft tree path作为输入到target推理，取末token多层 act
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
    # 根层：对前缀末 token做一次 draft_topk_fn（decoding阶段），得到 K 个候选子节点
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

    # 逐层 BFS：弹出父节点 → 可选刷新 target act → 对父末 token 再 topk → 入队子节点
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
        # 对paris作draft推理生成topk个候选，对the作draft推理生成topk个候选
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
# eagle2剪枝暂时未被用到
# EAGLE-2 风格 累计概率剪枝（cumulative beam）—— 相对EAGLE1 静态树多出来的能力：
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

    EAGLE3 的 make_eagle3_sgl_draft_topk_fn 返回的是 原始 logits 的 top-k 分数，暂时还没有集成eagle2的剪枝，目前此函数仅适用eagle2
    """
    ids = topk_ids[0, :k].tolist()
    sc = topk_scores[0, :k].float()
    if sc.numel() == 0:
        return [], []
    if sc.max() > 0:
        # 原始 logits：在 top-k 子集上做 log_softmax，得到可沿路径累加的条件 log 概率
        logps = F.log_softmax(sc, dim=-1).tolist()
    else:
        # 已是 log 概率（如 log_softmax 输出）
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

    # 先按 cum_log_prob 取 top max_nodes 个点，再把每个点的整条祖先链并入 keep
    ranked = sorted(range(n), key=lambda i: cum_log_probs[i], reverse=True)
    keep: set[int] = set()
    for i in ranked[:max_nodes]:
        j = i
        while j >= 0:
            keep.add(j)
            j = parent[j]

    # 重编号 parent，满足一个能正常表示树的顺序，使剪枝后仍是一棵合法树
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

    # frontier：当前每条 beam 的路径末端叶节点（每步只保留累计分最高的 K 条路径）
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

    # 第 2..num_steps 层：每条 beam 再扩 K 个子节点，摊平进 token_ids；全局只留 cum_logp 最高的 K 条 beam
    for _step in range(1, cfg.num_steps):
        if not frontier:
            break
        # 只留K条路径
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

    # 步数跑完后若节点超预算：按分数剪枝并保留祖先链
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
    """自回归（KV cache + RoPE + feature 递归）静态 top-k BFS 树展开。

    与 expand_draft_tree 树形状相同（静态 BFS），但 draft step()带状态，
    条件 hidden 在 draft 内部递归更新，而非每节点回调 draft_topk_fn。

    Parameters
    ----------
    cfg
        树超参：topk / num_steps / max_tree_nodes。
    draft_model
        Eagle3SglDraftModel；提供 step / topk_target_ids。
    root_logits
        draft prefill 末步 logits [1, V]；根层 topk 来源。
    root_hidden
        draft prefill 末步 hidden [1, H]；各 BFS 分支初始 cond。
    root_kv
        draft prefill 后的 KV (k, v)；BFS 分支 fork 时各路径各自 extend。
    past_len
        target 前缀长度 P 的末位置索引（RoPE 基准）；树内 depth=d 的 step 位置 = past_len-1+d。
    device
        计算设备。
    """
    K = cfg.topk
    if K < 1 or cfg.max_tree_nodes < 1:
        raise ValueError("invalid topk/max_tree_nodes")

    token_id_chunks: List[torch.Tensor] = []
    parent: List[int] = []
    depth: List[int] = []

    # Root candidates stay device-resident.  Tokens are materialized as one
    # Python list only after the complete tree has been built.
    root_idx, _ = draft_model.topk_target_ids(root_logits, K)

    # Benchmark-only compatibility path: reproduce the pre-optimization
    # num_steps=1 materialization exactly, including the root D2H copy.  The
    # normal/default path keeps this tensor on device and bypasses tree
    # materialization in Eagle3ProRunner.generate().
    if cfg.num_steps == 1 and not cfg.gpu_resident_top1:
        root_toks = [int(t) for t in root_idx[0, :K].tolist()]
        root_toks = root_toks[: cfg.max_tree_nodes]
        return TreeDraftResult(
            token_ids=root_toks,
            parent=[-1] * len(root_toks),
            node_depth=[1] * len(root_toks),
            bfs_index=list(range(len(root_toks))),
        )

    root_row = root_idx.reshape(-1)[:K]
    n_roots = min(int(root_row.shape[0]), cfg.max_tree_nodes)
    root_row = root_row[:n_roots]
    token_id_chunks.append(root_row)
    parent.extend([-1] * n_roots)
    depth.extend([1] * n_roots)

    total = n_roots
    if total < cfg.max_tree_nodes and cfg.num_steps > 1 and total > 0:
        # All nodes at one BFS depth share the same RoPE position and have
        # independent batch rows, so one batched step is numerically equivalent
        # to N batch-1 steps. Deeper/branching expansion always receives a
        # forkable tuple; CUDA-Graph mode obtains it as a read-only Flash-KV view.
        frontier_tokens = root_row.view(n_roots, 1)
        frontier_cond = root_hidden.expand(n_roots, -1).contiguous()
        root_k, root_v = root_kv
        frontier_k = root_k.expand(n_roots, *root_k.shape[1:]).contiguous()
        frontier_v = root_v.expand(n_roots, *root_v.shape[1:]).contiguous()
        frontier_global_idx = list(range(n_roots))

        current_depth = 1
        while (
            total < cfg.max_tree_nodes
            and current_depth < cfg.num_steps
            and frontier_tokens.shape[0] > 0
        ):
            frontier_size = int(frontier_tokens.shape[0])
            logits, hidden, kv = draft_model.step(
                frontier_tokens,
                frontier_cond,
                (frontier_k, frontier_v),
                past_len - 1 + current_depth,
            )
            child_idx, _ = draft_model.topk_target_ids(logits, K)
            children_per_parent = int(child_idx.shape[-1])
            child_flat = child_idx.reshape(frontier_size * children_per_parent)
            keep = min(
                cfg.max_tree_nodes - total,
                frontier_size * children_per_parent,
            )
            child_flat = child_flat[:keep]
            token_id_chunks.append(child_flat)

            parent_rows = torch.arange(keep, device=device) // children_per_parent
            # The row pattern is deterministic (0 repeated K times, then 1,
            # ...), so parent metadata needs no GPU-to-CPU readback.
            parent_row_list = [
                row
                for row in range(frontier_size)
                for _ in range(children_per_parent)
            ][:keep]
            parent.extend(frontier_global_idx[row] for row in parent_row_list)
            depth.extend([current_depth + 1] * keep)

            frontier_cond = hidden.index_select(0, parent_rows)
            frontier_k = kv[0].index_select(0, parent_rows)
            frontier_v = kv[1].index_select(0, parent_rows)
            frontier_tokens = child_flat.view(keep, 1)
            frontier_global_idx = list(range(total, total + keep))
            total += keep
            current_depth += 1

    token_ids = torch.cat(token_id_chunks).tolist() if token_id_chunks else []
    bfs_ix = list(range(len(token_ids)))
    return TreeDraftResult(
        token_ids=token_ids, parent=parent, node_depth=depth, bfs_index=bfs_ix
    )


@torch.inference_mode()
def expand_draft_chain_ar_gpu(
    cfg: Eagle3SglConfig,
    draft_model,
    root_logits: torch.Tensor,
    root_hidden: torch.Tensor,
    root_kv,
    past_len: int,
) -> torch.LongTensor:
    """Build a top-1 AR chain without materializing token ids on the CPU.

    Unlike :func:`expand_draft_tree_ar`, this fixed single-path helper does not
    need Python parent/depth metadata.  Every proposed id therefore remains in
    CUDA memory until target verification, removing one ``.tolist()`` sync and
    the following CPU-to-CUDA reconstruction from every speculative round.
    """
    steps = min(int(cfg.num_steps), int(cfg.max_tree_nodes))
    if int(cfg.topk) != 1 or steps < 1:
        raise ValueError("GPU-resident chain requires topk=1 and at least one step")

    greedy_ids = getattr(draft_model, "greedy_target_ids", None)
    token = (
        greedy_ids(root_logits)
        if greedy_ids is not None
        else draft_model.topk_target_ids(root_logits, 1)[0]
    )
    token = token[:, :1]
    chunks = [token]
    cond = root_hidden
    kv = root_kv
    flash_cache = getattr(kv, "_eagle3pro_draft_flash_cache", False)
    original_length = int(kv.length) if flash_cache else -1
    for depth in range(1, steps):
        if flash_cache:
            logits, cond, kv = draft_model.step_flash_cache(
                token, cond, kv, past_len - 1 + depth
            )
        else:
            logits, cond, kv = draft_model.step(
                token,
                cond,
                kv,
                past_len - 1 + depth,
            )
        token = (
            greedy_ids(logits)
            if greedy_ids is not None
            else draft_model.topk_target_ids(logits, 1)[0]
        )
        token = token[:, :1]
        chunks.append(token)
    if flash_cache:
        # Proposal KV is speculative scratch in the unused cache tail.  Restore
        # the canonical accepted length; commit will overwrite the accepted tail.
        kv.set_length(original_length)
    return torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]


def tree_draft_to_verify_layout(
    draft: TreeDraftResult,
) -> "TreeVerifyLayout":
    """TreeDraftResult → TreeVerifyLayout（verify / 掩码构造用）。

    Parameters
    ----------
    draft
        expand_draft_tree* 的输出；dfs_order 默认 range(L) 与 BFS 下标一致。
    """
    from .tree_verify_mask import TreeVerifyLayout

    n = len(draft.token_ids)
    return TreeVerifyLayout(
        token_ids=list(draft.token_ids),
        parent=list(draft.parent),
        dfs_order=list(range(n)),
    )
