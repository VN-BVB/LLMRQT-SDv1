# SPDX-License-Identifier: Apache-2.0
"""
EAGLE1 轻量 draft 网络（对齐论文 EAGLE1 的思想）：

  - 复用 target 的 embedding 与 lm_head（不另建词表/输出头）；
  - fc = Linear(2H, H)：把 token_embedding 与 target hidden 拼接后融合；
  - 本代码为 1 层 decoder（可配 num_layers多一些， 接受率会好点）；
  - 第 0 层跳过 input_layernorm（官方 EAGLE 设计：首层输入已是 fc 融合特征）。

draft 每次只对父 token + target hidden前向一步，输出下一 token 分布，配合
tree_draft.expand_draft_tree 静态展开 top-k 树。权重需由 EAGLE 训练得到；随机
初始化时接受率低，但 target verify 保证输出无损。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


def _unwrap(m: Union[PreTrainedModel, nn.Module]) -> PreTrainedModel:
    if isinstance(m, PreTrainedModel):
        return m
    inner = getattr(m, "model", None)
    if inner is not None and isinstance(inner, PreTrainedModel):
        return inner
    raise TypeError("target must be HF PreTrainedModel or BaseModelForCausalLM")


class _RMSNorm(nn.Module):
    """RMSNorm（Llama 风格）"""

    def __init__(self, dim: int, eps: float = 1e-6, **fk) -> None:
        # weight 初始化为 1；eps 防止除零。
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, **fk))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 fp32 计算 rms 以保证数值稳定，再转回原 dtype。
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # x / rms(x)
        return (x.to(dt)) * self.weight


def _sdpa_gqa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_heads: int, n_kv_heads: int
) -> torch.Tensor:
    """因果 SDPA，支持 GQA（k/v 头数 < q 头数时把 kv 头按组重复）。

    本项目里面有的 EAGLE-1 draft（如 LLaMA3-8B）是 GQA（32 q / 8 kv heads），不支持会导致权重 shape 不匹配。
    """
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class _DraftDecoderLayer(nn.Module):
    """Llama style的单层layer：(可选 input_norm) → self-attn → 残差 → post_norm → SwiGLU → 残差。"""

    def __init__(
        self,
        hidden: int,
        n_heads: int,
        intermediate: int,
        *,
        skip_input_layernorm: bool,
        n_kv_heads: int = None,
        eps: float = 1e-6,
        qkv_bias: bool = False,
        **fk,
    ) -> None:
        """构建单层layer的权重：attn 的 q/k/v/o 投影（GQA）+ SwiGLU MLP（gate/up/down）+ RMSNorm。

        qkv_bias：q/k/v_proj 是否带 bias（Qwen2/2.5 风格为 True；Llama 为 False）。o_proj 始终无 bias。
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = hidden // n_heads
        kv_dim = self.n_kv_heads * self.head_dim
        self.skip_input_layernorm = skip_input_layernorm
        # 第 0 层跳过 input_layernorm（官方 EAGLE：首层输入已是 fc 融合特征，无需再归一化）
        self.input_layernorm = (
            (lambda x: x) if skip_input_layernorm else _RMSNorm(hidden, eps=eps, **fk)
        )
        self.post_attention_layernorm = _RMSNorm(hidden, eps=eps, **fk)
        # 自注意力投影；q/o 输出 H，k/v 输出 kv_dim（GQA）。q/k/v bias 由 qkv_bias 控制（Qwen2 用）。
        self.q_proj = nn.Linear(hidden, hidden, bias=qkv_bias, **fk)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=qkv_bias, **fk)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=qkv_bias, **fk)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, **fk)
        # SwiGLU MLP：down(silu(gate(x)) * up(x))
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, **fk)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, **fk)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, **fk)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        """（GQA）多头因果自注意力。draft 通常只喂 1 个 token（decode阶段）。"""
        b, s, h = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # 因果 SDPA；无 rotary（单 token 位置无影响）
        o = _sdpa_gqa(q, k, v, self.n_heads, self.n_kv_heads)
        o = o.transpose(1, 2).contiguous().view(b, s, h)  # 合并头 -> [B,S,H]
        return self.o_proj(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """pre-norm 残差结构：(skip/RMSNorm)->attn->+res；RMSNorm->SwiGLU->+res。"""
        # 注意力层
        residual = x
        h = self.input_layernorm(x)
        x = residual + self._attn(h)
        # MLP层
        residual = x
        h = self.post_attention_layernorm(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + h


class Eagle1DraftModel(nn.Module):
    """
    轻量 EAGLE1 draft：复用 target embedding/lm_head + fc(2H→H) + num_layers 层 decoder。

    forward(token_ids[B,S], target_hidden[B,H]) -> logits[B, vocab]（取末位置）。
    """

    def __init__(
        self,
        target: Union[PreTrainedModel, nn.Module],
        *,
        num_layers: int = 1,
        use_final_norm: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """从 target 取维度/词表，复用其 embedding/lm_head，并新建 fc + decoder 层 + norm。

        use_final_norm=False：lm_head 前不加最终 RMSNorm（官方 EAGLE-1 checkpoint 无 draft norm，
        layer 输出直接进 lm_head；加载真实权重时设 False）。默认 True（随机初始化的轻量版保留）。
        """
        super().__init__()
        tm = _unwrap(target)
        cfg = tm.config
        hidden = int(cfg.hidden_size)
        n_heads = int(getattr(cfg, "num_attention_heads", max(1, hidden // 64)))
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
        intermediate = int(getattr(cfg, "intermediate_size", 4 * hidden))
        eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
        # 自动判定 q/k/v 是否带 bias（draft 沿用 target 的约定）：Qwen2/2.5 有、Llama 无。
        qkv_bias = False
        try:
            attn0 = tm.model.layers[0].self_attn
            qkv_bias = getattr(attn0, "q_proj", None) is not None and attn0.q_proj.bias is not None
        except Exception:
            qkv_bias = bool(getattr(cfg, "attention_bias", False))

        embed = tm.get_input_embeddings()
        lm_head = tm.get_output_embeddings()  # tie 时为 None
        # 复用 target 的 embedding / lm_head：用 list 持有，避免被注册为子模块而在 draft.to()
        # 时连带改写 target 自己的权重（参数共享但不归属本模块）。
        self._shared = [embed, lm_head]

        # 新建参数默认跟随 target 的 device/dtype
        if device is None:
            device = embed.weight.device
        if dtype is None:
            dtype = embed.weight.dtype
        fk = {"device": device, "dtype": dtype}

        # fc：把 [token_emb ; target_hidden]（2H）融合回 H —— EAGLE 的 feature 接入点。
        self.fc = nn.Linear(hidden * 2, hidden, bias=False, **fk)
        # decoder 层：第 0 层跳过 input_layernorm（支持 GQA）。
        self.layers = nn.ModuleList(
            [
                _DraftDecoderLayer(
                    hidden,
                    n_heads,
                    intermediate,
                    skip_input_layernorm=(i == 0),
                    n_kv_heads=n_kv_heads,
                    eps=eps,
                    qkv_bias=qkv_bias,
                    **fk,
                )
                for i in range(num_layers)
            ]
        )
        # 进 lm_head 前的最终归一化；官方 EAGLE-1 无此 norm（use_final_norm=False）。
        self.norm = _RMSNorm(hidden, eps=eps, **fk) if use_final_norm else (lambda x: x)
        self.hidden = hidden

    @property
    def embed(self) -> nn.Module:
        """复用的 target 输入 embedding（共享权重）。"""
        return self._shared[0]

    def _lm_logits(self, x: torch.Tensor) -> torch.Tensor:
        """用复用的 lm_head 投影到词表；若 target 与 embedding tie，则用 embedding 权重。"""
        lm_head = self._shared[1]
        if lm_head is not None:
            return lm_head(x)
        # tied embedding：logits = x @ embed.weight^T
        return F.linear(x, self.embed.weight)

    def forward(
        self,
        token_ids: torch.LongTensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """(token_ids[B,S], target_hidden[B,H] 或 [B,S,H]) -> logits[B, vocab]（取末位置）。"""
        emb = self.embed(token_ids)  # [B, S, H]：父 token 的词嵌入
        # 把 target hidden 对齐到 [B, S, H]（draft decode时 S=1）
        if target_hidden.dim() == 2:
            target_hidden = target_hidden.unsqueeze(1)  # [B, 1, H]
        if target_hidden.shape[1] != emb.shape[1]:
            target_hidden = target_hidden.expand(-1, emb.shape[1], -1)
        # feature 融合：x = fc([emb ; target_hidden]) —— EAGLE 的核心条件化
        x = self.fc(torch.cat([emb, target_hidden.to(emb.dtype)], dim=-1))
        for layer in self.layers:  # 通常 1 层，多层接受率会好些
            x = layer(x)
        x = self.norm(x)
        # 取最后一个位置过复用的 lm_head 得下一 token 分布
        logits = self._lm_logits(x[:, -1, :])
        return logits


def build_eagle1_draft(
    target: Union[PreTrainedModel, nn.Module],
    *,
    num_layers: int = 1,
    draft_weights_path: Optional[str] = None,
    use_final_norm: Optional[bool] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Eagle1DraftModel:
    """构造复用 target embedding/lm_head 的轻量 EAGLE1 draft。

    - draft_weights_path=None（默认）：随机初始化（可跑通、接受率低，正确性由 target verify 保证）。
    - draft_weights_path=<dir 或 .bin/.safetensors/.pt>：加载 官方 EAGLE-1 draft 权重
      （fc / layers.0.* / 复用 target embed），获得真实接受率。官方 checkpoint 无 draft 最终 norm，
      故加载真实权重时 use_final_norm 默认按 False（layer 输出直接进 lm_head）。
    """
    if use_final_norm is None:
        use_final_norm = draft_weights_path is None  # 真实权重默认无最终 norm
    m = Eagle1DraftModel(
        target, num_layers=num_layers, use_final_norm=use_final_norm, device=device, dtype=dtype
    )
    if draft_weights_path is not None:
        load_eagle1_draft_weights(m, draft_weights_path)
    return m


def _read_state_dict(path: str) -> dict:
    """从目录 / .safetensors / .bin / .pt 读出一个扁平的 name->tensor 状态字典。"""
    import glob
    import os

    if os.path.isdir(path):
        files = (
            sorted(glob.glob(os.path.join(path, "*.safetensors")))
            or sorted(glob.glob(os.path.join(path, "*.bin")))
            or sorted(glob.glob(os.path.join(path, "*.pt")))
        )
        if not files:
            raise FileNotFoundError(f"no weight file (*.safetensors/*.bin/*.pt) under {path}")
    else:
        files = [path]
    sd: dict = {}
    for f in files:
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd.update(load_file(f))
        else:
            obj = torch.load(f, map_location="cpu", weights_only=True)
            sd.update(obj.get("state_dict", obj) if isinstance(obj, dict) else obj)
    return sd


def _translate_eagle1_name(name: str) -> str:
    """官方 EAGLE-1 参数名 → 本模块命名：去 model. 前缀、self_attn./mlp. 中间名。

    例：layers.0.self_attn.q_proj -> layers.0.q_proj；layers.0.mlp.gate_proj -> layers.0.gate_proj。
    fc.weight / layers.0.post_attention_layernorm.weight 保持不变。对本模块自存的 sd 是幂等的。
    """
    if name.startswith("model."):
        name = name[len("model.") :]
    return name.replace("self_attn.", "").replace("mlp.", "")


def load_eagle1_draft_weights(
    draft_model: Eagle1DraftModel,
    path: str,
    *,
    strict: bool = False,
) -> dict:
    """把官方 EAGLE-1 draft 权重加载进 draft_model。

    名称翻译（去 self_attn./mlp.）、跳过 embed_tokens（复用 target）。
    官方 checkpoint 无 draft 最终 norm；建议 draft 以 use_final_norm=False 构造。

    Returns: {"loaded": [...], "missing": [...], "unexpected": [...]}。
    """
    src = _read_state_dict(path)
    local: dict = {}
    for k, v in src.items():
        if "embed_tokens" in k:
            continue  # 复用 target embedding
        local[_translate_eagle1_name(k)] = v
    res = draft_model.load_state_dict(local, strict=strict)
    unexpected = set(res.unexpected_keys)
    return {
        "loaded": [k for k in local if k not in unexpected],
        "missing": list(res.missing_keys),
        "unexpected": list(res.unexpected_keys),
    }


def make_eagle1_draft_topk_fn(draft_model: Eagle1DraftModel, topk: int):
    """工厂：把 Eagle1DraftModel 包成 tree_draft.DraftTopkFn。"""

    @torch.inference_mode()
    def fn(
        hidden: torch.Tensor, parent_token_id: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = draft_model(parent_token_id, hidden)  # [B, vocab]
        k = min(topk, logits.shape[-1])
        scores, idx = torch.topk(logits, k=k, dim=-1)
        return idx, scores

    return fn
