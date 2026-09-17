# SPDX-License-Identifier: Apache-2.0
"""
取 target 的多层（低/中/高）hidden states，拼成 EAGLE3 风格的 draft 条件特征。

forward_target_eagle_acts：前向 target 一次，取 eagle_layers 各层末 token 的 hidden 并拼接，
返回 [batch, len(eagle_layers) * hidden_dim]。供 EAGLE3-SGL 的 draft 做特征条件化。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


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
    从 HF output_hidden_states 元组里取多层 last-token 激活并拼接。

    hidden_states[0]=embedding 输出；hidden_states[i]=第 i-1 层输出 = 第 i 层的输入残差流。
    LlamaModel.forward对 eagle_layers=[2,16,29] 采集的是
    进入这些层之前的残差流（hidden_states + residual），即 HF 的 hidden_states[layer_idx]
    （= 第 layer_idx-1 层输出 = 第 layer_idx 层输入），而非该层的输出。
    因此这里取 hidden_states[layer_idx]（bug: 之前误用 layer_idx+1，会喂给训练好的 fc 错位特征）。
    """
    if not hidden_states:
        raise ValueError("hidden_states must be non-empty")
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
            h = h[:, -1, :]
        chunks.append(h.contiguous())
    return torch.cat(chunks, dim=-1)


def _decoder_layers(target: Any) -> nn.ModuleList:
    """取 HF CausalLM 的 decoder layer 列表（LlamaForCausalLM: .model.layers）。"""
    inner = getattr(target, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        layers = getattr(target, "layers", None)
    if layers is None:
        raise TypeError("target has no decoder .model.layers (expected Llama/Qwen-like CausalLM)")
    return layers


class EagleActCapture:
    """用 forward_pre_hook 捕获 eagle_layers 各层输入残差流，等价于 HF
    output_hidden_states 元组里的 hidden_states[layer_idx]，但只采集需要的几层，
    避免 output_hidden_states=True 收集并返回全部 num_layers+1 层。

    用法::

        with EagleActCapture(target, eagle_layers) as cap:
            out = target(input_ids, attention_mask=..., use_cache=True)  # 不要 output_hidden_states
        acts = cap.acts(last_token_only=False)  # [B, S, n*H]
    """

    def __init__(self, target: Any, eagle_layers: Sequence[int]) -> None:
        self._layers = _decoder_layers(target)
        self._eagle_layers = list(eagle_layers)
        self._captured: dict = {}
        self._handles: List[Any] = []

    def __enter__(self) -> "EagleActCapture":
        self._captured.clear()
        for li in self._eagle_layers:
            handle = self._layers[li].register_forward_pre_hook(
                self._make_hook(li), with_kwargs=True
            )
            self._handles.append(handle)
        return self

    def _make_hook(self, li: int):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            self._captured[li] = hs
            return None

        return hook

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def acts(self, *, last_token_only: bool = False) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for li in self._eagle_layers:
            if li not in self._captured:
                raise RuntimeError(f"eagle layer {li} not captured (forward not run under capture?)")
            h = self._captured[li]
            if last_token_only:
                h = h[:, -1, :]
            chunks.append(h.contiguous())
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
    前向 target，返回拼接后的多层 last-token 激活。

    Returns
    -------
    eagle_acts : Tensor [batch, len(eagle_layers) * hidden_dim]
    past_key_values
    outputs : model output object
    """
    from transformers import PreTrainedModel

    model = _unwrap_hf(target)
    if not isinstance(model, PreTrainedModel):
        raise TypeError("target must be a HuggingFace PreTrainedModel or wrap one in .model")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

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
    eagle_acts = select_eagle_acts_from_hidden_states(
        out.hidden_states, eagle_layers, last_token_only=last_token_only
    )
    return eagle_acts, out.past_key_values, out
