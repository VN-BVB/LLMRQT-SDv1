# SPDX-License-Identifier: Apache-2.0
"""EAGLE3Pro 的 Qwen3 batch-1 推理优化。

依次提供 packed QKV/gate-up、vLLM RMSNorm/SwiGLU/QK-RoPE，以及 target/draft
预分配 Flash KV cache。转换必须在 checkpoint 完整加载后、创建 generator 前执行；
它只改变推理调度和 cache 表示，不改变 target 贪心接受规则。

这些转换会替换内存中的模块结构，因此只面向推理；如需保存标准 Transformers
checkpoint，应保存转换前模型。适用范围和完整调用路径见 ``EAGLE3PRO_WORKFLOW.md``。
"""

from __future__ import annotations

import types
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PackedLinear(nn.Module):
    """持有按输出维拼接的权重，并通过一次 ``F.linear`` 计算全部投影。"""

    def __init__(self, linears: Sequence[nn.Linear]) -> None:
        super().__init__()
        if not linears:
            raise ValueError("linears must be non-empty")
        in_features = {int(layer.in_features) for layer in linears}
        if len(in_features) != 1:
            raise ValueError("all packed linears must have the same in_features")
        bias_modes = {layer.bias is not None for layer in linears}
        if len(bias_modes) != 1:
            raise ValueError("all packed linears must consistently use or omit bias")

        self.in_features = in_features.pop()
        self.split_sizes = tuple(int(layer.out_features) for layer in linears)
        self.out_features = sum(self.split_sizes)
        with torch.no_grad():
            weight = torch.cat([layer.weight.detach() for layer in linears], dim=0)
            bias = (
                torch.cat([layer.bias.detach() for layer in linears], dim=0)
                if linears[0].bias is not None
                else None
            )
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = (
            nn.Parameter(bias, requires_grad=False) if bias is not None else None
        )
        # EAGLE3PRO: 拼接顺序与原 forward 完全一致，split 后仍分别得到 Q/K/V 或 gate/up。

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


class Eagle3FlashKVCache:
    """Batch-1、全 attention 层的预分配 FlashAttention KV cache。"""

    _eagle3pro_flash_cache = True

    def __init__(
        self,
        key_cache: list[torch.Tensor],
        value_cache: list[torch.Tensor],
        lengths: list[int],
        *,
        graph_safe: bool = False,
    ) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.lengths = lengths
        self.max_cache_len = int(key_cache[0].shape[1])
        self.graph_safe = bool(graph_safe)
        self.cache_seqlens = (
            torch.full(
                (len(key_cache),),
                int(lengths[0]),
                device=key_cache[0].device,
                dtype=torch.int32,
            )
            if graph_safe
            else None
        )

    @classmethod
    def from_dynamic(
        cls,
        dynamic_cache: Any,
        max_cache_len: int,
        *,
        reuse: "Eagle3FlashKVCache | None" = None,
        graph_safe: bool = False,
    ):
        """把 prefill 的 HF `[B,H,S,D]` cache 一次性转成 `[B,S,H,D]`。"""
        layers = list(dynamic_cache.layers)
        if reuse is not None and reuse._can_reuse(
            layers, max_cache_len=max_cache_len, graph_safe=graph_safe
        ):
            reuse._load_dynamic_layers(layers)
            return reuse

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        lengths: list[int] = []
        for layer in layers:
            key = layer.keys
            value = layer.values
            if key.shape[0] != 1:
                raise NotImplementedError("Flash KV fast path currently supports batch_size=1")
            sequence_length = int(key.shape[2])
            if sequence_length > max_cache_len:
                raise ValueError("prefill sequence exceeds requested Flash KV capacity")
            key_storage = torch.empty(
                (1, max_cache_len, key.shape[1], key.shape[3]),
                device=key.device,
                dtype=key.dtype,
            )
            value_storage = torch.empty_like(key_storage)
            key_storage[:, :sequence_length].copy_(key.transpose(1, 2))
            value_storage[:, :sequence_length].copy_(value.transpose(1, 2))
            keys.append(key_storage)
            values.append(value_storage)
            lengths.append(sequence_length)
        # EAGLE3PRO: 只在 prefill 后转置一次；decode 不再逐轮 torch.cat 整段 KV。
        return cls(keys, values, lengths, graph_safe=graph_safe)

    def _can_reuse(
        self,
        layers: list[Any],
        *,
        max_cache_len: int,
        graph_safe: bool,
    ) -> bool:
        if self.graph_safe != bool(graph_safe) or self.max_cache_len < max_cache_len:
            return False
        if len(layers) != len(self.key_cache):
            return False
        return all(
            layer.keys.shape[0] == storage.shape[0]
            and layer.keys.shape[1] == storage.shape[2]
            and layer.keys.shape[3] == storage.shape[3]
            and layer.keys.device == storage.device
            and layer.keys.dtype == storage.dtype
            for layer, storage in zip(layers, self.key_cache)
        )

    def _load_dynamic_layers(self, layers: list[Any]) -> None:
        lengths: list[int] = []
        for layer, key_storage, value_storage in zip(
            layers, self.key_cache, self.value_cache
        ):
            sequence_length = int(layer.keys.shape[2])
            if sequence_length > self.max_cache_len:
                raise ValueError("prefill sequence exceeds persistent Flash KV capacity")
            key_storage[:, :sequence_length].copy_(layer.keys.transpose(1, 2))
            value_storage[:, :sequence_length].copy_(layer.values.transpose(1, 2))
            lengths.append(sequence_length)
        self.lengths = lengths
        if self.cache_seqlens is not None:
            self.cache_seqlens.fill_(lengths[0])

    def set_length(self, length: int, *, update_device: bool = True) -> None:
        """Set every target layer to one logical length without reallocating storage."""
        length = int(length)
        if length < 0 or length > self.max_cache_len:
            raise ValueError(f"target cache length {length} outside capacity")
        self.lengths = [length] * len(self.lengths)
        if update_device and self.cache_seqlens is not None:
            self.cache_seqlens.fill_(length)

    def mark_replayed(self, length: int) -> None:
        """Update Python metadata after replay; graph kernels updated device lengths."""
        self.set_length(length, update_device=False)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.lengths[layer_idx])

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(0, self.get_seq_length() + max_length)
        new_length = min(self.get_seq_length(), int(max_length))
        self.set_length(new_length)

    def attend_and_update(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        softmax_scale: float,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_with_kvcache

        before = self.get_seq_length(layer_idx)
        after = before + int(key.shape[1])
        if after > self.max_cache_len:
            raise RuntimeError(
                f"Flash KV cache capacity {self.max_cache_len} < required {after}"
            )
        cache_seqlens = (
            self.cache_seqlens[layer_idx : layer_idx + 1]
            if self.cache_seqlens is not None
            else before
        )
        output = flash_attn_with_kvcache(
            query,
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            k=key,
            v=value,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            rotary_interleaved=False,
        )
        self.lengths[layer_idx] = after
        if self.cache_seqlens is not None:
            self.cache_seqlens[layer_idx : layer_idx + 1].fill_(after)
            # 这一 fill 会被录入 Graph，使下一轮前 device 长度与实际 KV 写入同步。
        # EAGLE3PRO: 每层独立推进长度；一轮结束后所有层自然保持同一逻辑长度。
        return output


class Eagle3DraftFlashKVCache:
    """单层 EAGLE3 draft 的预分配、原地更新 KV cache。"""

    _eagle3pro_draft_flash_cache = True

    def __init__(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        length: int,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        *,
        graph_safe: bool = False,
    ) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.length = int(length)
        self.rotary_cos = rotary_cos
        self.rotary_sin = rotary_sin
        self.max_cache_len = int(key_cache.shape[1])
        self.graph_safe = bool(graph_safe)
        self.cache_seqlens = (
            torch.tensor([self.length], device=key_cache.device, dtype=torch.int32)
            if graph_safe
            else None
        )

    @classmethod
    def from_tuple(
        cls,
        kv: tuple[torch.Tensor, torch.Tensor],
        *,
        max_cache_len: int,
        inv_freq: torch.Tensor,
        reuse: "Eagle3DraftFlashKVCache | None" = None,
        graph_safe: bool = False,
    ) -> "Eagle3DraftFlashKVCache":
        key, value = kv
        sequence_length = int(key.shape[2])
        if reuse is not None and reuse._can_reuse(
            key, max_cache_len=max_cache_len, graph_safe=graph_safe
        ):
            reuse.key_cache[:, :sequence_length].copy_(key.transpose(1, 2))
            reuse.value_cache[:, :sequence_length].copy_(value.transpose(1, 2))
            reuse.set_length(sequence_length)
            return reuse
        key_storage = torch.empty(
            (key.shape[0], max_cache_len, key.shape[1], key.shape[3]),
            device=key.device,
            dtype=key.dtype,
        )
        value_storage = torch.empty_like(key_storage)
        key_storage[:, :sequence_length].copy_(key.transpose(1, 2))
        value_storage[:, :sequence_length].copy_(value.transpose(1, 2))
        positions = torch.arange(
            max_cache_len, device=key.device, dtype=torch.float32
        )
        frequencies = positions[:, None] * inv_freq.to(
            device=key.device, dtype=torch.float32
        )[None, :]
        # EAGLE3PRO: draft 的非持久 inv_freq 可能仍在 CPU，建表时显式迁移到 cache device。
        # EAGLE3PRO: draft 使用非 interleaved/NeoX RoPE；FlashAttention 接收半维 cos/sin 表。
        return cls(
            key_storage,
            value_storage,
            sequence_length,
            frequencies.cos().to(key.dtype),
            frequencies.sin().to(key.dtype),
            graph_safe=graph_safe,
        )

    def _can_reuse(
        self,
        key: torch.Tensor,
        *,
        max_cache_len: int,
        graph_safe: bool,
    ) -> bool:
        storage = self.key_cache
        return (
            self.graph_safe == bool(graph_safe)
            and self.max_cache_len >= max_cache_len
            and key.shape[0] == storage.shape[0]
            and key.shape[1] == storage.shape[2]
            and key.shape[3] == storage.shape[3]
            and key.device == storage.device
            and key.dtype == storage.dtype
            and int(key.shape[2]) <= self.max_cache_len
        )

    def set_length(self, length: int, *, update_device: bool = True) -> None:
        length = int(length)
        if length < 0 or length > self.max_cache_len:
            raise ValueError(f"draft cache length {length} outside capacity")
        self.length = length
        if update_device and self.cache_seqlens is not None:
            self.cache_seqlens.fill_(length)

    def mark_replayed(self, length: int) -> None:
        self.set_length(length, update_device=False)

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a read-only logical-prefix view for top-1 draft-chain expansion."""
        return (
            self.key_cache[:, : self.length].transpose(1, 2),
            self.value_cache[:, : self.length].transpose(1, 2),
        )

    def attend_and_update(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        start_position: int,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_with_kvcache

        if int(start_position) != self.length:
            raise ValueError(
                f"draft cache length {self.length} != start_position {start_position}"
            )
        after = self.length + int(key.shape[1])
        if after > int(self.key_cache.shape[1]):
            raise RuntimeError("draft Flash KV cache capacity exceeded")
        output = flash_attn_with_kvcache(
            query,
            self.key_cache,
            self.value_cache,
            k=key,
            v=value,
            rotary_cos=self.rotary_cos,
            rotary_sin=self.rotary_sin,
            cache_seqlens=(
                self.cache_seqlens if self.cache_seqlens is not None else self.length
            ),
            causal=True,
            rotary_interleaved=False,
        )
        self.length = after
        if self.cache_seqlens is not None:
            # Graph replay starts from a different logical prefix each round.
            # Relative advancement remains valid inside a captured graph,
            # whereas fill_(capture_time_after) would bake in one stale length.
            self.cache_seqlens.add_(int(key.shape[1]))
        # EAGLE3PRO: 消除 draft 每轮 cat/repeat_interleave，并把 RoPE、写 cache、GQA attention 合成一次。
        return output


def _qwen3_attention_forward_packed(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
):
    """Transformers Qwen3Attention.forward 的等价 packed-QKV 版本。"""
    from transformers.models.qwen3 import modeling_qwen3 as qwen3_mod

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    qkv = self.qkv_proj(hidden_states)
    if getattr(self, "_eagle3pro_vllm_qk_rope", False):
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            raise ValueError("fused QK-norm+RoPE requires position_ids")
        batch_size, sequence_length = hidden_states.shape[:2]
        flat_positions = position_ids.expand(batch_size, sequence_length).reshape(-1)
        torch.ops._C.fused_qk_norm_rope(
            qkv.view(-1, qkv.shape[-1]),
            int(self.config.num_attention_heads),
            int(self.config.num_key_value_heads),
            int(self.config.num_key_value_heads),
            int(self.head_dim),
            float(self.q_norm.variance_epsilon),
            self.q_norm.weight,
            self.k_norm.weight,
            self._eagle3pro_rope_cache,
            True,
            flat_positions,
        )
        # EAGLE3PRO: vLLM 原地融合 Q/K RMSNorm 与 NeoX RoPE；V 区域保持不变。
        query_states, key_states, value_states = qkv.split(
            self.qkv_proj.split_sizes, dim=-1
        )
        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)
    else:
        query_states, key_states, value_states = qkv.split(
            self.qkv_proj.split_sizes, dim=-1
        )
        # EAGLE3PRO: 非融合回退仍保留 Q/K norm；漏掉会导致 Qwen3 有损输出。
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qwen3_mod.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if getattr(past_key_values, "_eagle3pro_flash_cache", False):
        if not getattr(self, "_eagle3pro_vllm_qk_rope", False):
            raise RuntimeError("Flash KV cache requires fused QK-norm+RoPE")
        if self.sliding_window not in (None, -1):
            raise NotImplementedError("Flash KV fast path does not support sliding attention")
        attn_output = past_key_values.attend_and_update(
            int(self.layer_idx),
            query_states,
            key_states,
            value_states,
            softmax_scale=float(self.scaling),
        )
        # EAGLE3PRO: FlashAttention 同一 kernel 原地写 KV 并完成 GQA causal attention。
        attn_output = attn_output.reshape(*input_shape, -1)
        return self.o_proj(attn_output), None

    if getattr(self, "_eagle3pro_vllm_qk_rope", False):
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = qwen3_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        qwen3_mod.eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _qwen3_mlp_forward_packed(self, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_up = self.gate_up_proj(hidden_states)
    if getattr(self, "_eagle3pro_vllm_silu", False):
        activated = _vllm_silu_and_mul(gate_up)
        # EAGLE3PRO: 可选复用 vLLM 已编译的单 kernel SwiGLU，避免分散的 SiLU 与乘法。
    else:
        gate, up = gate_up.split(self.gate_up_proj.split_sizes, dim=-1)
        activated = self.act_fn(gate) * up
    # EAGLE3PRO: gate/up 共用一次 GEMM；激活、乘法、down projection 顺序保持不变。
    return self.down_proj(activated)


def _vllm_silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (*gate_up.shape[:-1], gate_up.shape[-1] // 2),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    torch.ops._C.silu_and_mul(out, gate_up)
    return out


def _vllm_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(hidden_states)
    epsilon = getattr(self, "variance_epsilon", getattr(self, "eps", None))
    if epsilon is None:
        raise AttributeError("RMSNorm module has neither variance_epsilon nor eps")
    torch.ops._C.rms_norm(out, hidden_states, self.weight, float(epsilon))
    return out


def _decoder_layers(target: nn.Module) -> Iterable[nn.Module]:
    decoder = getattr(target, "model", target)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError(
            "packed projections require a Qwen3-style target with model.layers"
        )
    return layers


@torch.inference_mode()
def enable_qwen3_packed_projections(target: nn.Module) -> dict[str, int]:
    """原地把 Qwen3 target 的 QKV 与 gate/up 权重转换为推理用 packed 布局。

    返回 ``{"attention_layers": N, "mlp_layers": N}``。重复调用是幂等的；若
    模型不是标准 Qwen3 层布局则立即报错，避免静默修改不兼容模型。
    """
    attention_count = 0
    mlp_count = 0
    for layer in _decoder_layers(target):
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attention is None or mlp is None:
            raise TypeError("every Qwen3 decoder layer must expose self_attn and mlp")
        if "Qwen3" not in type(attention).__name__:
            raise TypeError(
                f"expected Qwen3 attention, got {type(attention).__name__}"
            )

        if not hasattr(attention, "qkv_proj"):
            attention.qkv_proj = _PackedLinear(
                [attention.q_proj, attention.k_proj, attention.v_proj]
            )
            del attention.q_proj
            del attention.k_proj
            del attention.v_proj
            attention.forward = types.MethodType(
                _qwen3_attention_forward_packed, attention
            )
            attention_count += 1
            # EAGLE3PRO: 删除旧投影引用，packed 权重不会额外长期占用一份显存。

        if not hasattr(mlp, "gate_up_proj"):
            mlp.gate_up_proj = _PackedLinear([mlp.gate_proj, mlp.up_proj])
            del mlp.gate_proj
            del mlp.up_proj
            mlp.forward = types.MethodType(_qwen3_mlp_forward_packed, mlp)
            mlp_count += 1
            # EAGLE3PRO: 只改变投影调度，down_proj 与激活数学顺序保持不变。

    return {"attention_layers": attention_count, "mlp_layers": mlp_count}


@torch.inference_mode()
def enable_vllm_fused_kernels(
    target: nn.Module,
    draft_model: nn.Module | None = None,
    *,
    rms_norm: bool = False,
    qk_norm_rope: bool = False,
    flash_kv_cache: bool = False,
) -> dict[str, int]:
    """为 packed Qwen3 target（及可选 draft）启用本机 vLLM 融合算子。

    这是显式可选路径：调用会要求当前环境已安装可用的 vLLM CUDA 扩展。
    ``rms_norm=False`` 时只启用数值逐位一致的 fused SwiGLU。
    """
    import vllm._custom_ops  # noqa: F401  # 注册 torch.ops._C CUDA 算子

    target_mlp = 0
    for layer in _decoder_layers(target):
        mlp = layer.mlp
        if not hasattr(mlp, "gate_up_proj"):
            raise RuntimeError("enable packed projections before vLLM fused kernels")
        mlp._eagle3pro_vllm_silu = True
        target_mlp += 1

    draft_mlp = 0
    if draft_model is not None:
        midlayer = getattr(draft_model, "midlayer", None)
        if midlayer is None or not hasattr(midlayer, "gate_up_proj"):
            raise RuntimeError("draft midlayer must be packed before vLLM fused kernels")
        midlayer._eagle3pro_vllm_silu = True
        draft_mlp = 1

    if flash_kv_cache and not qk_norm_rope:
        raise ValueError("flash_kv_cache requires qk_norm_rope=True")

    qk_rope_layers = 0
    if qk_norm_rope:
        decoder = getattr(target, "model", target)
        rotary = getattr(decoder, "rotary_emb", None)
        if rotary is None or getattr(rotary, "rope_type", "default") != "default":
            raise NotImplementedError("fused QK-norm+RoPE currently supports default RoPE")
        if float(getattr(rotary, "attention_scaling", 1.0)) != 1.0:
            raise NotImplementedError("scaled RoPE is not supported by this fused path")
        inv_freq = rotary.inv_freq.detach().to(device=next(target.parameters()).device)
        positions = torch.arange(
            int(target.config.max_position_embeddings),
            device=inv_freq.device,
            dtype=torch.float32,
        )
        frequencies = positions[:, None] * inv_freq.float()[None, :]
        rope_cache = torch.cat(
            [frequencies.cos(), frequencies.sin()], dim=-1
        ).to(dtype=next(target.parameters()).dtype)
        decoder._eagle3pro_rope_cache = rope_cache
        # EAGLE3PRO: 全层共享约 10 MiB 的 RoPE cache；不会为 28 层重复分配。
        for layer in _decoder_layers(target):
            layer.self_attn._eagle3pro_vllm_qk_rope = True
            layer.self_attn._eagle3pro_rope_cache = rope_cache
            qk_rope_layers += 1

    if flash_kv_cache:
        target._eagle3pro_flash_kv_enabled = True
        if draft_model is not None:
            draft_model._eagle3pro_flash_kv_enabled = True
        # EAGLE3PRO: runner 只在 top-1 快路径检测此标记并转换 cache；其它 verify 模式不变。

    norm_count = 0
    if rms_norm:
        modules = list(target.modules())
        if draft_model is not None:
            modules.extend(draft_model.modules())
        for module in modules:
            if type(module).__name__ not in ("Qwen3RMSNorm", "_RMSNorm"):
                continue
            module.forward = types.MethodType(_vllm_rms_norm_forward, module)
            norm_count += 1
            # EAGLE3PRO: 只在显式 rms_norm=True 时替换，便于独立回归性能与数值。

    return {
        "vllm_silu_target_layers": target_mlp,
        "vllm_silu_draft_layers": draft_mlp,
        "vllm_rms_norms": norm_count,
        "vllm_qk_norm_rope_layers": qk_rope_layers,
        "flash_kv_cache": int(flash_kv_cache),
    }
