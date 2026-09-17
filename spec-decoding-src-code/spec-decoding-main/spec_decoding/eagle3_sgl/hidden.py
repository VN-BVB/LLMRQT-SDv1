# SPDX-License-Identifier: Apache-2.0
"""
取 target 的多层（低/中/高）hidden states，拼成 EAGLE3 风格的 draft 条件特征。

forward_target_eagle_acts：前向 target 一次，取 eagle_layers 各层末 token 的 hidden 并拼接，
返回 [batch, len(eagle_layers) * hidden_dim]。供 EAGLE3 的 draft 做特征条件化。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


@torch.inference_mode()
def forward_target_selected_eagle_acts(
    target: nn.Module,
    eagle_layers: Sequence[int],
    *,
    last_token_only: bool,
    **forward_kwargs: Any,
) -> Tuple[Any, torch.Tensor]:
    """只捕获指定 decoder 层的输入残差流，并返回 ``(outputs, acts)``。

    Transformers 5 的 ``output_hidden_states=True`` 会给所有 decoder 层收集输出；
    EAGLE3 实际只需要三层，因此用临时 pre-hook 避免保留整套 hidden states。
    """
    decoder = getattr(target, "model", target)
    # 既支持 CausalLM（层在 target.model.layers），也支持直接传入 backbone（层在 target.layers）。
    layers = getattr(decoder, "layers", None)
    if layers is None:
        outputs = target(output_hidden_states=True, **forward_kwargs)
        acts = select_eagle_acts_from_hidden_states(
            outputs.hidden_states,
            eagle_layers,
            last_token_only=last_token_only,
        )
        return outputs, acts
        # GPT-2 等非 Llama/Qwen 布局保留通用全 hidden fallback，避免破坏原 CPU smoke 用法。

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in eagle_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(
                f"eagle layer index {layer_idx} out of range for {len(layers)} layers"
            )

        def capture_input(_module, args, *, idx=layer_idx):
            captured[idx] = args[0]
            # decoder layer 的第一个位置参数就是进入该层前的 residual hidden state。

        handles.append(layers[layer_idx].register_forward_pre_hook(capture_input))

    try:
        outputs = target(output_hidden_states=False, **forward_kwargs)
        # 关闭全层输出捕获；临时 hook 已保存训练 EAGLE3 所需的三层输入。
    finally:
        for handle in handles:
            handle.remove()

    missing = [idx for idx in eagle_layers if idx not in captured]
    if missing:
        raise RuntimeError(f"target forward did not visit EAGLE layers: {missing}")
    chunks = []
    for layer_idx in eagle_layers:
        hidden = captured[layer_idx]
        if last_token_only:
            hidden = hidden[:, -1, :]
        chunks.append(hidden.contiguous())
    return outputs, torch.cat(chunks, dim=-1)
    # 拼接顺序严格沿用 eagle_layers，数值语义与 hidden_states[layer_idx] 相同。


def _unwrap_hf(m: Any) -> Any:
    inner = getattr(m, "model", None)
    from transformers import PreTrainedModel

    if isinstance(m, PreTrainedModel):
        return m
    if inner is not None:
        return inner
    return m


def select_eagle_acts_from_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    eagle_layers: Sequence[int],
    *,
    last_token_only: bool = True,
) -> torch.Tensor:
    """
    从 HF output_hidden_states 元组里取多层激活并拼接，供 EAGLE3 draft 吸取作为输入。

    Parameters
    ----------
    hidden_states
        HF forward 返回的 hidden_states 元组。hidden_states[0] 为 embedding 输出；
        hidden_states[i]（i>=1）为第 i-1 层 decoder 输出 = 第 i 层输入残差流。
    eagle_layers
        要采集的层下标列表（如 [2, L//2, L-3]）。取 hidden_states[layer_idx]，
        即进入该层之前的残差流（对齐官方 EAGLE3 训练时的 fc 输入）。
    last_token_only
        True：每层只取末 token [:, -1, :]，输出 [B, n*H]（建树根条件 / recovery）。
        False：保留全序列 [B, S, H] 逐层 cat 成 [B, S, n*H]（draft prefill / acts_all）。

    Returns
    -------
    Tensor
        last_token_only=True → [B, len(eagle_layers) * hidden_dim]；
        False → [B, S, len(eagle_layers) * hidden_dim]。
    """
    if not hidden_states:
        raise ValueError("hidden_states must be non-empty")
    # hidden_states[i] = 进入第 i 层 transformer 前的残差流
    n_layers = len(hidden_states) - 1
    chunks: List[torch.Tensor] = []
    for layer_idx in eagle_layers:
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            raise ValueError(
                f"eagle layer index {layer_idx} out of range for "
                f"{n_layers} transformer layers"
            )
        h = hidden_states[layer_idx]  # 第 layer_idx 层的输入残差流
        if last_token_only:
            h = h[:, -1, :]  # 只关心前缀最后一个位置的多层特征
        chunks.append(h.contiguous())
    # 输出 [B, n*H] 或 [B, S, n*H]：draft 的 fc 入口维 = len(eagle_layers) * H
    return torch.cat(chunks, dim=-1)


@torch.inference_mode()
def forward_target_eagle_acts(
    target: Union[Any, nn.Module],
    input_ids: torch.LongTensor,
    eagle_layers: Sequence[int],
    attention_mask: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    *,
    use_cache: bool = True,
    last_token_only: bool = True,
) -> Tuple[torch.Tensor, Any, Any]:
    """
    前向 target，返回拼接后的多层 act。

    Parameters
    ----------
    target
        HF CausalLM 或其 .model 包装。
    input_ids
        输入 token [B, S]。
    eagle_layers
        要采集的层下标（见 select_eagle_acts_from_hidden_states）。
    attention_mask
        可选；默认全 1。
    past_key_values
        可选增量 KV；generate() 通常不用此路径取 act（prefill/replay 自带 hidden_states）。
    use_cache
        是否返回/更新 KV cache。
    last_token_only
        True → [B, n*H]（末 token）；False → [B, S, n*H]（全序列，供 acts_all）。

    Returns
    -------
    eagle_acts
        拼接后的多层 act。
    past_key_values
        若 use_cache=True。
    outputs
        HF model output（含 logits / hidden_states 等）。
    """
    from transformers import PreTrainedModel

    model = _unwrap_hf(target)
    if not isinstance(model, PreTrainedModel):
        raise TypeError("target must be a HuggingFace PreTrainedModel or wrap one in .model")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    # 一次 target forward：要留住 hidden 就要 output_hidden_states=True
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_hidden_states=True,
        return_dict=True,
    )
    if out.hidden_states is None:
        raise RuntimeError(
            "Model returned no hidden_states; use a CausalLM with output_hidden_states=True."
        )
    # 从整段 hidden 里按 eagle_layers 切片并拼接 → draft 条件向量（或每位置 acts_all）
    eagle_acts = select_eagle_acts_from_hidden_states(
        out.hidden_states, eagle_layers, last_token_only=last_token_only
    )
    return eagle_acts, out.past_key_values, out
