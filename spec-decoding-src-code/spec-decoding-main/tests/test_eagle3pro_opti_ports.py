import random
from types import SimpleNamespace

import pytest
import torch

from spec_decoding.eagle3_sgl_llama.tree_draft import (
    expand_draft_tree_ar as expand_draft_tree_ar_reference,
)
from spec_decoding.eagle3pro.config import Eagle3SglConfig
from spec_decoding.eagle3pro.tree_draft import (
    expand_draft_chain_ar_gpu,
    expand_draft_tree_ar,
)
from spec_decoding.eagle3pro.tree_verify_full import _select_append_root_child
from spec_decoding.eagle3pro.tree_verify_mask import (
    TreeVerifyLayout,
    _on_tree_path,
    build_tree_allow_mask_4d,
    build_tree_cross_attn_bias_with_prefix,
    build_tree_sdpa_attn_bias,
)
from spec_decoding.eagle3pro.target_optim import (
    Eagle3DraftFlashKVCache,
    Eagle3FlashKVCache,
)
from spec_decoding.eagle3pro.tree_verify_full import _fork_cache_metadata


def test_cuda_graph_config_validates_workspace_and_warmup():
    cfg = Eagle3SglConfig(cuda_graph=True, cuda_graph_max_seq_len=256)
    assert cfg.cuda_graph
    assert cfg.cuda_graph_max_seq_len == 256
    with pytest.raises(ValueError, match="cuda_graph_max_seq_len"):
        Eagle3SglConfig(cuda_graph_max_seq_len=0)
    with pytest.raises(ValueError, match="cuda_graph_warmup_iters"):
        Eagle3SglConfig(cuda_graph_warmup_iters=0)


def test_tree_verify_forks_cache_metadata_without_copying_prefix_storage():
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn_like(key)
    cache.update(key, value, 0)
    forked = _fork_cache_metadata(cache)

    assert forked is not cache
    assert forked.layers[0] is not cache.layers[0]
    assert forked.layers[0].keys.data_ptr() == cache.layers[0].keys.data_ptr()

    forked.update(torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4), 0)
    assert forked.get_seq_length() == 5
    assert cache.get_seq_length() == 3


def test_graph_safe_flash_caches_reuse_storage_addresses():
    first_layer = SimpleNamespace(
        keys=torch.randn(1, 2, 3, 4), values=torch.randn(1, 2, 3, 4)
    )
    target = Eagle3FlashKVCache.from_dynamic(
        SimpleNamespace(layers=[first_layer]), 8, graph_safe=True
    )
    target_key_ptr = target.key_cache[0].data_ptr()
    second_layer = SimpleNamespace(
        keys=torch.randn(1, 2, 5, 4), values=torch.randn(1, 2, 5, 4)
    )
    reused_target = Eagle3FlashKVCache.from_dynamic(
        SimpleNamespace(layers=[second_layer]),
        8,
        reuse=target,
        graph_safe=True,
    )
    assert reused_target is target
    assert reused_target.key_cache[0].data_ptr() == target_key_ptr
    assert reused_target.get_seq_length() == 5
    assert reused_target.cache_seqlens.tolist() == [5]

    first_kv = (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
    draft = Eagle3DraftFlashKVCache.from_tuple(
        first_kv, max_cache_len=8, inv_freq=torch.randn(2), graph_safe=True
    )
    draft_key_ptr = draft.key_cache.data_ptr()
    second_kv = (torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    reused_draft = Eagle3DraftFlashKVCache.from_tuple(
        second_kv,
        max_cache_len=8,
        inv_freq=torch.randn(2),
        reuse=draft,
        graph_safe=True,
    )
    assert reused_draft is draft
    assert reused_draft.key_cache.data_ptr() == draft_key_ptr
    assert reused_draft.length == 5
    assert reused_draft.cache_seqlens.tolist() == [5]
    tuple_k, tuple_v = reused_draft.as_tuple()
    assert tuple_k.shape == (1, 2, 5, 4)
    assert tuple_v.shape == (1, 2, 5, 4)
    assert (
        tuple_k.untyped_storage().data_ptr()
        == reused_draft.key_cache.untyped_storage().data_ptr()
    )


def test_vectorized_masks_match_path_reference():
    rng = random.Random(20260912)
    for _ in range(50):
        n = rng.randint(1, 30)
        parent = [-1] + [rng.randrange(i) for i in range(1, n)]
        layout = TreeVerifyLayout(list(range(n)), parent, list(range(n)))
        expected = torch.tensor(
            [
                [_on_tree_path(parent, key, query) for key in range(n)]
                for query in range(n)
            ],
            dtype=torch.bool,
        )

        allow = build_tree_allow_mask_4d(
            layout, device=torch.device("cpu"), dtype=torch.float32
        )[0, 0]
        bias = build_tree_sdpa_attn_bias(
            layout, device=torch.device("cpu"), dtype=torch.float32
        )
        past = rng.randint(0, 20)
        cross = build_tree_cross_attn_bias_with_prefix(
            layout, past, device=torch.device("cpu"), dtype=torch.float32
        )

        assert torch.equal(allow == 1, expected)
        assert torch.equal(bias == 0, expected)
        assert torch.equal(cross[:, :past] == 0, torch.ones(n, past, dtype=torch.bool))
        assert torch.equal(cross[:, past:] == 0, expected)


def _select_reference(node_logits, recovery_logits, parent, token_ids, leaves):
    best_m, best_li, best_next, best_leaf = -1, 10**9, 0, 0
    for li, leaf in enumerate(leaves):
        path = []
        current = leaf
        while current >= 0:
            path.append(current)
            current = parent[current]
        path.reverse()
        matched = 0
        correction = None
        for offset, node in enumerate(path):
            source = recovery_logits if offset == 0 else node_logits[path[offset - 1]]
            prediction = int(source.argmax(dim=-1).item())
            if prediction != token_ids[node]:
                correction = prediction
                break
            matched += 1
        next_token = (
            int(node_logits[path[-1]].argmax(dim=-1).item())
            if matched == len(path)
            else correction
        )
        if matched > best_m or (matched == best_m and li < best_li):
            best_m, best_li = matched, li
            best_next, best_leaf = next_token, leaf
    path = []
    current = best_leaf
    while current >= 0:
        path.append(current)
        current = parent[current]
    path.reverse()
    path_tokens = [token_ids[node] for node in path]
    append = path_tokens + [best_next] if best_m == len(path) else path_tokens[:best_m] + [best_next]
    return append, best_m, best_leaf, path_tokens


def test_batched_accept_selection_matches_scalar_reference():
    parent = [-1, 0, 0, 1, 1, 2]
    leaves = [3, 4, 5]
    token_ids = [4, 2, 3, 1, 5, 0]
    generator = torch.Generator().manual_seed(17)
    for _ in range(20):
        node_logits = torch.randn(6, 11, generator=generator)
        recovery_logits = torch.randn(11, generator=generator)
        assert _select_append_root_child(
            node_logits, recovery_logits, parent, token_ids, leaves
        ) == _select_reference(
            node_logits, recovery_logits, parent, token_ids, leaves
        )


class _FakeDraft:
    def topk_target_ids(self, logits, k):
        scores, indices = torch.topk(logits, k=k, dim=-1)
        return indices, scores

    def step(self, token_ids, hidden, past_kv, position):
        batch = token_ids.shape[0]
        vocab = 13
        centers = (token_ids[:, 0] + int(position) + 3) % vocab
        vocab_ids = torch.arange(vocab).view(1, -1)
        logits = -(vocab_ids - centers.view(-1, 1)).abs().float()
        next_hidden = hidden + token_ids.to(hidden.dtype)
        key, value = past_kv
        extension = token_ids.view(batch, 1, 1, 1).to(key.dtype)
        return (
            logits,
            next_hidden,
            (torch.cat([key, extension], dim=2), torch.cat([value, extension], dim=2)),
        )


def test_depth_batched_ar_tree_matches_nodewise_reference():
    cfg = Eagle3SglConfig(topk=3, num_steps=3, max_tree_nodes=13)
    draft = _FakeDraft()
    root_logits = torch.arange(13, dtype=torch.float32).view(1, -1)
    root_hidden = torch.zeros(1, 2)
    root_kv = (torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1))
    args = (cfg, draft, root_logits, root_hidden, root_kv, 4)

    actual = expand_draft_tree_ar(*args, device=torch.device("cpu"))
    expected = expand_draft_tree_ar_reference(*args, device=torch.device("cpu"))

    assert actual.token_ids == expected.token_ids
    assert actual.parent == expected.parent
    assert actual.node_depth == expected.node_depth
    assert actual.bfs_index == expected.bfs_index


def test_gpu_resident_top1_chain_matches_tree_tokens():
    cfg = Eagle3SglConfig(topk=1, num_steps=4, max_tree_nodes=8)
    draft = _FakeDraft()
    root_logits = torch.arange(13, dtype=torch.float32).view(1, -1)
    root_hidden = torch.zeros(1, 2)
    root_kv = (torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1))
    args = (cfg, draft, root_logits, root_hidden, root_kv, 4)

    chain = expand_draft_chain_ar_gpu(*args)
    tree = expand_draft_tree_ar(*args, device=torch.device("cpu"))

    assert chain.shape == (1, 4)
    assert chain.tolist()[0] == tree.token_ids
