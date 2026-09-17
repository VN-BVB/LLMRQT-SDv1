# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for profile-opti correctness compatibility fixes."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from spec_decoding.eagle3_sgl_profile_opti.target_graph import (
    StaticGraphVerifier,
    build_pending_tree_mask,
)
from spec_decoding.eagle3_sgl_profile_opti.tree_verify_mask import TreeVerifyLayout


def test_static_verifier_restores_recent_hf_cache_lengths() -> None:
    tensor_layer = SimpleNamespace(cumulative_length=torch.tensor(17))
    integer_layer = SimpleNamespace(cumulative_length=17)
    legacy_layer = SimpleNamespace()
    verifier = object.__new__(StaticGraphVerifier)
    verifier.sc = SimpleNamespace(
        layers=[tensor_layer, integer_layer, legacy_layer]
    )

    verifier._set_cache_length(9)

    assert tensor_layer.cumulative_length.item() == 9
    assert integer_layer.cumulative_length == 9
    assert not hasattr(legacy_layer, "cumulative_length")


def test_pending_tree_mask_keeps_branches_isolated() -> None:
    layout = TreeVerifyLayout(
        token_ids=[10, 20, 11, 21],
        parent=[-1, -1, 0, 1],
        dfs_order=[0, 1, 2, 3],
    )
    mask = build_pending_tree_mask(
        layout, before=2, max_len=9, device=torch.device("cpu"), dtype=torch.float32
    )[0, 0]

    # pending sees prefix + itself, but no tree slot
    assert torch.equal(mask[0] == 0, torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool))
    # child 11 sees prefix, pending, root 10 and itself; sibling branch stays masked
    assert torch.equal(mask[3] == 0, torch.tensor([1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool))


def test_multibranch_compact_preserves_static_cache_addresses() -> None:
    keys = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
    values = (100 + torch.arange(10, dtype=torch.float32)).view(1, 1, 10, 1)
    layer = SimpleNamespace(keys=keys.clone(), values=values.clone())
    verifier = object.__new__(StaticGraphVerifier)
    verifier.sc = SimpleNamespace(layers=[layer])
    verifier.device = torch.device("cpu")
    key_ptr = layer.keys.data_ptr()
    value_ptr = layer.values.data_ptr()

    # before=2: pending is slot 2; tree nodes start at slot 3.
    # Accepted node indices [2, 5] live in slots [5, 8] and compact to [3, 4].
    verifier._compact_tree_path(before=2, path_indices=[2, 5], kept_draft=2)

    assert layer.keys.data_ptr() == key_ptr
    assert layer.values.data_ptr() == value_ptr
    assert layer.keys[0, 0, 3:5, 0].tolist() == [5.0, 8.0]
    assert layer.values[0, 0, 3:5, 0].tolist() == [105.0, 108.0]
