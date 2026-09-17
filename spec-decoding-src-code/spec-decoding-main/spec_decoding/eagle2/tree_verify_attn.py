# SPDX-License-Identifier: Apache-2.0
"""
Verify attention backends for full_model_tree:

- flash_attn (topk==1): HF flash_attention_2，标准链式因果，不叠树 mask。
- triton_tree (topk>1): SGLang extend_attention_fwd + custom_mask。
- eager：tree_verify_attention_context + HF forward（回退 / 对照）。
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
)

_TRITON_SUPPORTED_ATTN = frozenset(
    {
        "LlamaAttention",
        "Qwen2Attention",
        "Qwen3Attention",
        "MistralAttention",
        "Gemma2Attention",
    }
)


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


def triton_extend_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        from .kernels.extend_attention import extend_attention_fwd  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_verify_attn_backend(
    backend: Optional[str],
    *,
    topk: int,
    layout_len: int,
) -> str:
    """
    auto: topk==1 + FA → flash_attn；topk>1 + Triton → triton_tree；否则 eager。
    """
    b = (backend or "auto").lower()
    if b == "auto":
        if topk <= 1 and layout_len > 0 and flash_attn_available():
            return "flash_attn"
        if topk > 1 and layout_len > 0 and triton_extend_available():
            return "triton_tree"
        return "eager"
    if b not in ("eager", "flash_attn", "triton_tree"):
        raise ValueError(
            "verify_attn_backend must be auto, eager, flash_attn, or triton_tree"
        )
    if b == "flash_attn" and not flash_attn_available():
        raise RuntimeError("flash_attn requested but flash-attn is not available")
    if b == "triton_tree" and not triton_extend_available():
        raise RuntimeError("triton_tree requested but triton extend kernel is unavailable")
    return b


@dataclass
class TritonTreeVerifyMetadata:
    """Indptr / indices / bool mask for extend_attention_fwd (batch=1)."""

    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    max_len_extend: int
    past_len: int
    extend_len: int


def build_triton_tree_verify_metadata(
    layout: TreeVerifyLayout,
    past_len: int,
    *,
    device: torch.device,
) -> TritonTreeVerifyMetadata:
    """Build metadata for one tree-verify extend step.

    关键约定：kv_indptr / kv_indices 只描述 前缀 KV（长度 past_len），
    kernel 用 cur_seq_len_prefix = past_len、stage-1 attend 前缀；extend 的 L 个树节点在 K_Extend 里。
    custom_mask / mask_indptr 行 stride = past_len+L = total_kv。
    """
    l_n = len(layout)
    total_kv = int(past_len) + l_n
    cross = build_tree_cross_attn_bias_with_prefix(
        layout, past_len, device=device, dtype=torch.float32
    )
    custom_mask = (cross == 0).reshape(-1).to(torch.bool)
    qo_indptr = torch.tensor([0, l_n], dtype=torch.int32, device=device)
    # 前缀 KV 长度 = past_len（不含 extend）
    kv_indptr = torch.tensor([0, int(past_len)], dtype=torch.int32, device=device)
    kv_indices = torch.arange(int(past_len), dtype=torch.int32, device=device)
    # mask 行 stride = total_kv（前缀 + extend），与 kernel 的 cur_seq_len 一致
    mask_indptr = torch.tensor(
        [0, l_n * total_kv], dtype=torch.int32, device=device
    )
    return TritonTreeVerifyMetadata(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        custom_mask=custom_mask,
        mask_indptr=mask_indptr,
        max_len_extend=l_n,
        past_len=int(past_len),
        extend_len=l_n,
    )


def tree_custom_mask_bool_flat(
    layout: TreeVerifyLayout,
    past_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Flat bool mask (True = allow), length L * (past_len + L)."""
    return build_triton_tree_verify_metadata(layout, past_len, device=device).custom_mask


@contextmanager
def tree_verify_attention_context_flash(decoder: nn.Module):
    """Use HF flash_attention_2 without tree-mask patch (chain / topk=1)."""
    config = getattr(decoder, "config", None)
    saved_impl: Optional[str] = None
    if config is not None and hasattr(config, "_attn_implementation"):
        saved_impl = config._attn_implementation
        config._attn_implementation = "flash_attention_2"
    try:
        yield
    finally:
        if config is not None and saved_impl is not None:
            config._attn_implementation = saved_impl


def _apply_rotary(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_embeddings is not None:
        cos, sin = position_embeddings
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            return apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except ImportError:
            pass
    rotary = getattr(module, "rotary_emb", None)
    if rotary is None:
        return query_states, key_states
    raise RuntimeError(
        "triton_tree verify needs position_embeddings from the decoder layer"
    )


def _layer_idx(module: nn.Module) -> int:
    idx = getattr(module, "layer_idx", None)
    if idx is not None:
        return int(idx)
    raise AttributeError("self_attn.layer_idx is required for triton tree verify")

# 植入了triton custom mask attn的attn layer
def _triton_tree_attn_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Any = None,
    attention_mask: Any = None,
    past_key_values: Any = None,
    cache_position: Any = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    from .kernels.extend_attention import extend_attention_fwd

    # transformers 5.x 把 attention 的 cache kwarg 从 past_key_value 改名为 past_key_values（复数）；
    # 两种命名都兜底，否则前缀 KV 不会传进来 → 树节点只能 attend 自己 → verify 全错。
    past_key_value = past_key_values
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_value")

    ctx: TritonTreeVerifyMetadata = module._llmqrt_triton_meta
    bsz, q_len, _ = hidden_states.shape
    if bsz != 1:
        raise NotImplementedError("triton_tree verify supports batch_size=1 only")

    # transformers 5.x 把头数从 attention 模块挪到了 config，这里两种命名都兼容。
    mcfg = getattr(module, "config", None)
    num_heads = getattr(module, "num_heads", None)
    if num_heads is None:
        num_heads = int(getattr(mcfg, "num_attention_heads"))
    num_kv = getattr(module, "num_key_value_heads", None)
    if num_kv is None:
        num_kv = int(getattr(mcfg, "num_key_value_heads", num_heads))
    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        head_dim = module.q_proj.out_features // num_heads
    query_states = (
        module.q_proj(hidden_states)
        .view(bsz, q_len, num_heads, head_dim)
        .transpose(1, 2)
    )
    key_states = (
        module.k_proj(hidden_states)
        .view(bsz, q_len, num_kv, head_dim)
        .transpose(1, 2)
    )
    value_states = (
        module.v_proj(hidden_states)
        .view(bsz, q_len, num_kv, head_dim)
        .transpose(1, 2)
    )

    query_states, key_states = _apply_rotary(
        module, query_states, key_states, position_embeddings
    )

    if past_key_value is not None:
        cache_kwargs = {}
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        key_states, value_states = past_key_value.update(
            key_states, value_states, _layer_idx(module), cache_kwargs
        )

    # extend_attention_fwd expects Q with all heads, K/V with kv heads (GQA inside kernel).
    q = query_states.transpose(1, 2).reshape(-1, num_heads, head_dim).contiguous()
    k_all = key_states.transpose(1, 2).reshape(-1, num_kv, head_dim).contiguous()
    v_all = value_states.transpose(1, 2).reshape(-1, num_kv, head_dim).contiguous()
    l_ext = ctx.extend_len
    k_extend = k_all[-l_ext:]
    v_extend = v_all[-l_ext:]
    o = torch.empty_like(q)

    extend_attention_fwd(
        q,
        k_extend,
        v_extend,
        o,
        k_all,
        v_all,
        ctx.qo_indptr,
        ctx.kv_indptr,
        ctx.kv_indices,
        ctx.custom_mask,
        False,
        ctx.mask_indptr,
        ctx.max_len_extend,
        sm_scale=head_dim**-0.5,
        skip_prefix_custom_mask=True,
    )

    attn_output = o.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = module.o_proj(attn_output)
    # transformers 5.x 的 attention 返回 (attn_output, attn_weights) 两元组（KV cache 已就地更新）。
    return attn_output, None


@contextmanager
def tree_verify_triton_attention_context(
    decoder: nn.Module,
    metadata: TritonTreeVerifyMetadata,
):
    """Patch each decoder self_attn.forward to call extend_attention_fwd."""
    originals: List[Tuple[nn.Module, Callable[..., Any]]] = []
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("decoder.layers not found for triton_tree verify")

    for layer in layers:
        attn = layer.self_attn
        if attn.__class__.__name__ not in _TRITON_SUPPORTED_ATTN:
            raise NotImplementedError(
                f"triton_tree verify unsupported attention: {attn.__class__.__name__}"
            )
        attn._llmqrt_triton_meta = metadata
        if not hasattr(attn, "_llmqrt_orig_forward"):
            attn._llmqrt_orig_forward = attn.forward
        originals.append((attn, attn._llmqrt_orig_forward))

        def _wrapped(
            self: nn.Module,
            hidden_states: torch.Tensor,
            *args: Any,
            **kw: Any,
        ) -> Any:
            return _triton_tree_attn_forward(self, hidden_states, *args, **kw)

        attn.forward = types.MethodType(_wrapped, attn)

    try:
        yield
    finally:
        for attn, orig in originals:
            attn.forward = orig
            if hasattr(attn, "_llmqrt_triton_meta"):
                del attn._llmqrt_triton_meta
