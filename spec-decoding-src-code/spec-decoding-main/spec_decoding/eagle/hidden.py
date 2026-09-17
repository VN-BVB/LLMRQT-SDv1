# SPDX-License-Identifier: Apache-2.0
"""
在 Target hidden 上建 Draft（条件信号从哪来）

相对只喂 token 的 draft：小模型每一步还应看见 大模型对当前前缀
的理解。工程上通常取 最后一层（或指定层）在 最后一个 token 位置
上的向量 h（即 lm_head 之前的 hidden），与 token embedding 组合后再过 draft
（见 runner.make_hidden_residual_draft_topk_fn 的 emb + proj(h) 范式）。

本文件职责：从 HF CausalLM 前向中取出该 h（output_hidden_states）。
如何把 h 接到你的 draft 结构里由调用方/工厂函数负责。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel


def _unwrap(m: Union[PreTrainedModel, nn.Module]) -> PreTrainedModel:
    if isinstance(m, PreTrainedModel):
        return m
    inner = getattr(m, "model", None)
    if inner is not None and isinstance(inner, PreTrainedModel):
        return inner
    raise TypeError("model must be HF PreTrainedModel or BaseModelForCausalLM")


# -----------------------------------------------------------------------------
# 完整 draft 侧：每个当前前缀对应一个 last-token hidden，供 topk 树
# 上每个节点扩展时作为条件（可配合 past_key_values 只做增量段，见调用处）。
# -----------------------------------------------------------------------------
@torch.inference_mode()
def forward_target_last_hidden(
    target: Union[PreTrainedModel, nn.Module],
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    *,
    hidden_layer_index: int = -1,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Any, Any]:
    """
    前向 target，返回 最后一个 token 位置 上指定层的 hidden、past_key_values、完整 outputs。

    Returns
    -------
    last_hidden : Tensor [batch, hidden_dim]
    past_key_values
    outputs : ModelOutput（含 hidden_states 时非 None）
    """
    model = _unwrap(target)
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
            "Model returned no hidden_states; ensure forward supports "
            "output_hidden_states=True (HF CausalLM)."
        )
    hs = out.hidden_states[hidden_layer_index]
    last_h = hs[:, -1, :].contiguous()
    return last_h, out.past_key_values, out
