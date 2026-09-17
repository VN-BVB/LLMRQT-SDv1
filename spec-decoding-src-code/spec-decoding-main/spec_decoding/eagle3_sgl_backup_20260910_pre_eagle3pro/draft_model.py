# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3 轻量 draft 网络（对齐官方 EAGLE3 llama_eagle3 结构）。

相对 EAGLE1 的轻量 draft，这里实现了 EAGLE3 的2处关键点：

1. 多层 target hidden 特征：fc = Linear(n*H, H) 先把 target 低/中/高 n 层 hidden 拼接
   （n*H）降到 H（与 token embedding 分开），n = len(eagle_layers)。
2. 独立 draft 词表 + d2t 映射 + 不共享 lm_head（可选）：给定 draft_vocab_size 时，
   draft 用自己的 lm_head: Linear(H, draft_vocab_size)，输出在draft 词表上；
   再用 hot_token_id = d2t + arange 把 draft id 映射回 target id。未给定则回退复用
   target 的 lm_head（draft id 即 target id）。
   
注意: draft有独立此表的意义：target 的词表很大（如 128k）。但实际生成时，绝大多数概率质量集中在一小
撮高频 token 上。EAGLE3 的观察是：draft 只需要在这热门子集（hot tokens，比如几万个）上猜就够了。
2.1 draft 每步的 lm_head 矩阵乘和 softmax 维度从 V_target 降到 V_draft，draft 本来就追求轻量快速
2.2 draft 的输出头Lm_head专门为猜得准训练，与 target 的 lm_head 解耦
于是给 draft 一个更小的独立词表 + 独立 lm_head,代价是 draft 只能提议这子集内的 token，但因为它们覆
盖了主要概率质量，接受率影响很小；而且即使 draft 提议不在子集、或猜错，最终都由 target verify 兜底，不影响正确性。

3. midlayer：对 token embedding 与（降维后的）hidden 分别归一化（input_layernorm /
   hidden_norm），拼接成 2H 再进 qkv（q/k/v_proj 输入维度 = 2H）。

draft 权重随机初始化时接受率低，但 target verify 保证输出无损。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

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
    """RMSNorm"""

    def __init__(self, dim: int, eps: float = 1e-6, **fk) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, **fk))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out = t.float() * cos + _rotate_half(t.float()) * sin
    return out.to(t.dtype)


def _sdpa_gqa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_heads: int, n_kv_heads: int
) -> torch.Tensor:
    """因果 SDPA，支持 GQA（k/v 头数 < q 头数时把 kv 头按组重复）。

    真实 EAGLE3 draft（如 LLaMA3.1-8B）是 GQA（如 32 q / 8 kv heads），不支持会导致权重 shape 不匹配。
    """
    if n_kv_heads != n_heads:
        # SDPA 要求 q/k/v head 数一致：把每组 kv 复制到对应多个 q head
        rep = n_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class _Eagle3MidLayer(nn.Module):
    """EAGLE3 首层（midlayer）：

    对 embeds 与 hidden 分别归一化后拼成 2H 喂 self-attn（qkv 输入 = 2H，
    q/o 输出 H，k/v 输出 n_kv_heads*head_dim，即 GQA），再过 post-norm + SwiGLU。
    残差流取自传入的 hidden（降维后的多层 act）。
    """

    def __init__(
        self, hidden: int, n_heads: int, intermediate: int, n_kv_heads: int = None,
        rope_base: float = 10000.0, eps: float = 1e-6, **fk,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = hidden // n_heads
        kv_dim = self.n_kv_heads * self.head_dim
        inv = 1.0 / (
            rope_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)
        self.input_layernorm = _RMSNorm(hidden, eps=eps, **fk)   # 归一化 token embedding
        self.hidden_norm = _RMSNorm(hidden, eps=eps, **fk)        # 归一化（降维后的）多层 act —— EAGLE3 特有
        self.post_attention_layernorm = _RMSNorm(hidden, eps=eps, **fk)
        # qkv 输入维度 = 2H（embed 与 hidden 拼接）；q/o 输出 H，k/v 输出 kv_dim（GQA）
        self.q_proj = nn.Linear(2 * hidden, hidden, bias=False, **fk)
        self.k_proj = nn.Linear(2 * hidden, kv_dim, bias=False, **fk)
        self.v_proj = nn.Linear(2 * hidden, kv_dim, bias=False, **fk)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, **fk)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, **fk)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, **fk)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, **fk)

    def _attn(self, x2h: torch.Tensor) -> torch.Tensor:
        """对 [B,S,2H] 输入做（GQA）多头因果自注意力，输出 [B,S,H]。"""
        b, s, _ = x2h.shape
        q = self.q_proj(x2h).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x2h).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x2h).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        o = _sdpa_gqa(q, k, v, self.n_heads, self.n_kv_heads)
        return self.o_proj(o.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.head_dim))

    def forward(self, embeds: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # EAGLE3：两条输入流（token embed vs target 多层 act 经 fc变换的 H），拼成 2H 再进注意力
        residual = hidden                       # 残差流 = 降维后的多层 act
        e = self.input_layernorm(embeds)         # 归一化 embed
        hn = self.hidden_norm(hidden)            # 归一化 高中低hidden（EAGLE3论文idea，已经经过了fc）
        x = torch.cat([e, hn], dim=-1)           # [B,S,2H]
        hidden = residual + self._attn(x)        # attn 输出 H + 残差
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + mlp                    # [B,S,H]

    def _rope_cos_sin(self, positions: torch.Tensor, device, dtype) -> tuple:
        pos = positions.to(device=device, dtype=torch.float32).view(-1)
        freqs = pos[:, None] * self.inv_freq.to(device=device, dtype=torch.float32)[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def prefill(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ):
        b, S, _ = embeds.shape
        residual = hidden
        e = self.input_layernorm(embeds)
        hn = self.hidden_norm(hidden)
        x = torch.cat([e, hn], dim=-1)
        q = self.q_proj(x).view(b, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self._rope_cos_sin(positions, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        new_kv = (k, v)
        # GQA：把 kv repeat 到 q head 维再做SDPA
        rep = self.n_heads // self.n_kv_heads
        kk = k.repeat_interleave(rep, dim=1) if rep > 1 else k
        vv = v.repeat_interleave(rep, dim=1) if rep > 1 else v
        o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(b, S, self.n_heads * self.head_dim)
        hidden = residual + self.o_proj(o)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + mlp, new_kv

    def extend(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        past_kv,
        start_position: int,
    ):
        """在 past_kv 上批量 extend A 个新 token（一次 SDPA，比逐步 step 快）。"""
        b, A, _ = embeds.shape
        residual = hidden
        e = self.input_layernorm(embeds)
        hn = self.hidden_norm(hidden)
        x = torch.cat([e, hn], dim=-1)
        q = self.q_proj(x).view(b, A, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, A, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, A, self.n_kv_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(
            start_position, start_position + A, device=x.device, dtype=torch.long
        )
        cos, sin = self._rope_cos_sin(positions, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            # KV 沿 seq 维拼接；下面用显式 mask 保证新 Q 段对整段 K 的因果性可见性
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)
        rep = self.n_heads // self.n_kv_heads
        kk = k.repeat_interleave(rep, dim=1) if rep > 1 else k
        vv = v.repeat_interleave(rep, dim=1) if rep > 1 else v
        past_len = int(kk.shape[2]) - A
        # 显式 mask：第 i 个新 token 的 Q 只能 attend 前 past_len+i+1 个 K
        idx_q = torch.arange(A, device=x.device).view(A, 1)
        idx_k = torch.arange(kk.shape[2], device=x.device).view(1, -1)
        allowed = idx_k <= (past_len + idx_q)
        attn_mask = torch.zeros((A, kk.shape[2]), device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attn_mask)
        o = o.transpose(1, 2).contiguous().view(b, A, self.n_heads * self.head_dim)
        hidden = residual + self.o_proj(o)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + mlp, new_kv

    def step(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        past_kv,
        position: int,
    ):
        b = embeds.shape[0]
        residual = hidden
        e = self.input_layernorm(embeds)
        hn = self.hidden_norm(hidden)
        x = torch.cat([e, hn], dim=-1)
        q = self.q_proj(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self._rope_cos_sin(torch.tensor([position], device=x.device), x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            # 树 BFS 每分支每节点的1个step：在已有 KV 上追加当前 token 的 (k,v)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)
        rep = self.n_heads // self.n_kv_heads
        kk = k.repeat_interleave(rep, dim=1) if rep > 1 else k
        vv = v.repeat_interleave(rep, dim=1) if rep > 1 else v
        o = F.scaled_dot_product_attention(q, kk, vv)
        o = o.transpose(1, 2).contiguous().view(b, 1, self.n_heads * self.head_dim)
        hidden = residual + self.o_proj(o)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + mlp, new_kv


class _PlainLayer(nn.Module):
    """标准单输入 pre-norm 层（仅 num_layers>1 时，midlayer 之后再叠的层）。"""

    def __init__(
        self, hidden: int, n_heads: int, intermediate: int, n_kv_heads: int = None,
        eps: float = 1e-6, **fk,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = hidden // n_heads
        kv_dim = self.n_kv_heads * self.head_dim
        self.input_layernorm = _RMSNorm(hidden, eps=eps, **fk)
        self.post_attention_layernorm = _RMSNorm(hidden, eps=eps, **fk)
        self.q_proj = nn.Linear(hidden, hidden, bias=False, **fk)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False, **fk)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False, **fk)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, **fk)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, **fk)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, **fk)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, **fk)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        o = _sdpa_gqa(q, k, v, self.n_heads, self.n_kv_heads)
        return self.o_proj(o.transpose(1, 2).contiguous().view(b, s, h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准 Llama block：pre-norm attn + pre-norm SwiGLU
        x = x + self._attn(self.input_layernorm(x))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class Eagle3SglDraftModel(nn.Module):
    """
    EAGLE3 draft（官方 EAGLE3 midlayer 结构）：

    forward(token_ids[B,S], target_acts[B, n*H]) -> logits[B, V]（取末位置）。
    V = draft_vocab_size（给定时，draft 自己的词表）或 target vocab（回退复用 lm_head）。
    """

    def __init__(
        self,
        target: Union[PreTrainedModel, nn.Module],
        eagle_layers: Sequence[int],
        *,
        num_layers: int = 1,
        draft_vocab_size: Optional[int] = None,
        d2t: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """构造 EAGLE3 draft 网络。

        Parameters
        ----------
        target
            HF CausalLM；复用其 embedding（及可选 lm_head）。
        eagle_layers
            eagle3 act目标layer下标；n_acts = len(eagle_layers)，决定 fc 输入维 n*H。
        num_layers
            decoder 层数；1 时仅 midlayer；>1 时 midlayer 后叠 _PlainLayer。
        draft_vocab_size
            独立 draft 词表大小；给定则建 lm_head(H, draft_vocab_size) + hot_token_id。
        d2t
            draft id → target id 偏移；hot_token_id[i] = d2t[i] + i。
        device, dtype
            新建参数（fc/midlayer/lm_head）的设备与 dtype。
        """
        super().__init__()
        tm = _unwrap(target)
        cfg = tm.config
        hidden = int(cfg.hidden_size)
        n_heads = int(getattr(cfg, "num_attention_heads", max(1, hidden // 64)))
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
        intermediate = int(getattr(cfg, "intermediate_size", 4 * hidden))
        rope_base = float(getattr(cfg, "rope_theta", 10000.0))
        eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
        n_acts = len(list(eagle_layers))
        self.n_acts = n_acts
        self.hidden = hidden

        embed = tm.get_input_embeddings()
        # 复用 target embedding。
        self._shared_embed = [embed]

        if device is None:
            device = embed.weight.device
        if dtype is None:
            dtype = embed.weight.dtype
        fk = {"device": device, "dtype": dtype}

        # (1) 多层 aux 特征：fc 把 n 层拼接的 act（n*H）降到 H（与 embedding 分开）。
        self.fc = nn.Linear(n_acts * hidden, hidden, bias=False, **fk)
        # (2) midlayer：embed/hidden 分别归一化后拼 2H 进 qkv（支持 GQA）。
        self.midlayer = _Eagle3MidLayer(
            hidden, n_heads, intermediate, n_kv_heads=n_kv_heads,
            rope_base=rope_base, eps=eps, **fk
        )
        self.extra_layers = nn.ModuleList(
            [
                _PlainLayer(hidden, n_heads, intermediate, n_kv_heads=n_kv_heads, eps=eps, **fk)
                for _ in range(max(0, num_layers - 1))
            ]
        )
        self.norm = _RMSNorm(hidden, eps=eps, **fk)

        # (3) 独立 draft 词表 + d2t 映射 + 不共享 lm_head。
        self.draft_vocab_size = draft_vocab_size
        if draft_vocab_size is not None:
            self.lm_head = nn.Linear(hidden, draft_vocab_size, bias=False, **fk)  # draft 自己的输出头
            if d2t is None:
                d2t = torch.zeros(draft_vocab_size, dtype=torch.long)
            # hot_token_id[draft_id] = 该 draft id 对应的 target vocab id
            hot = d2t.to(torch.long) + torch.arange(draft_vocab_size, dtype=torch.long)
            self.register_buffer("hot_token_id", hot.to(device=device))
            self._shared_lmhead = None
        else:
            self.lm_head = None
            self.hot_token_id = None
            self._shared_lmhead = [tm.get_output_embeddings()]

    @property
    def embed(self) -> nn.Module:
        return self._shared_embed[0]

    def _lm_logits(self, x: torch.Tensor) -> torch.Tensor:
        """末 hidden → logits；基于独立词表或回退到 target lm_head / tied embed。"""
        if self.lm_head is not None:
            return self.lm_head(x)  # draft 词表 [B, V_draft]
        lmh = self._shared_lmhead[0]
        if lmh is not None:
            return lmh(x)
        return F.linear(x, self.embed.weight)  # tied embedding

    def forward(
        self,
        token_ids: torch.LongTensor,
        target_acts: torch.Tensor,
    ) -> torch.Tensor:
        """同step的核心区别主要是无状态无 KV extend update
        供非 autoregression模式下调用make_eagle3_sgl_draft_topk_fn建树）。

        Parameters
        ----------
        token_ids
            [B, S] draft token 序列（通常 S=1）。
        target_acts
            [B, n*H] 或 [B, S, n*H] target 多层 act；经 fc 降维后与 embed 进 midlayer。

        Returns
        -------
        logits
            [B, V] 末位置 logits（V = draft_vocab 或 target vocab）。
        """
        embeds = self.embed(token_ids)  # [B, S, H]
        if target_acts.dim() == 2:
            target_acts = target_acts.unsqueeze(1)  # [B, 1, n*H] 广播到 S
        if target_acts.shape[1] != embeds.shape[1]:
            target_acts = target_acts.expand(-1, embeds.shape[1], -1)
        # 主干：低中高3层 target act → fc → H，与 token embed 一起进 EAGLE3 midlayer
        reduced = self.fc(target_acts.to(embeds.dtype))  # n*H → H
        h = self.midlayer(embeds, reduced)               # embed/hidden 分别 norm 后拼 2H
        for layer in self.extra_layers:
            h = layer(h)
        h = self.norm(h)
        return self._lm_logits(h[:, -1, :])

    @torch.inference_mode()
    def step(
        self,
        token_id: torch.LongTensor,
        cond: torch.Tensor,
        past_kv,
        position: int,
    ):
        """自回归单 token 步（expand_draft_tree_ar 建树用）。

        Parameters
        ----------
        token_id
            当前步输入 token [B, 1]。
        cond
            draft 条件 hidden [B, H] 或 [B, 1, n*H]；若末维为 n*H 则过 fc。
        past_kv
            已有 draft KV (k, v) 或 None。
        position
            RoPE 绝对位置（树内 = past_len - 1 + depth）。

        Returns
        -------
        logits, hidden, new_kv
            末步 logits [B, V]、更新后的 cond hidden [B, H]、extend 后的 KV。
        """
        if self.extra_layers:
            raise NotImplementedError("autoregressive step supports num_layers=1 (midlayer only)")
        embeds = self.embed(token_id)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if cond.shape[-1] == self.n_acts * self.hidden:
            reduced = self.fc(cond.to(embeds.dtype))  # 首轮 cond 仍是 n*H act
        else:
            reduced = cond.to(embeds.dtype)           # 后续步 cond 已是 H（feature 递归）
        h, new_kv = self.midlayer.step(embeds, reduced, past_kv, position)
        # 末位置 norm + lm_head → 本步对下一 token的分布；h[:, -1] 作为下一步 cond（feature 递归）
        logits = self._lm_logits(self.norm(h)[:, -1, :])
        return logits, h[:, -1, :], new_kv

    @torch.inference_mode()
    def prefill(
        self,
        prefix_ids: torch.LongTensor,
        target_acts_all: torch.Tensor,
    ):
        """对 prompt做 draft prefill。

        Parameters
        ----------
        prefix_ids
            完整 prompt [1, P]。
        target_acts_all
            各位置多层 act [1, P, n*H]（与 prefix_ids 等长）。

        Returns
        -------
        root_logits, root_hidden, kv
            末步 logits [1, V]、末 hidden [1, H]、draft KV（长 P-1）。

        EAGLE shift：draft 消费 prefix_ids[:,1:] 与 acts_all[:,:-1]，
        即位置 t 的 draft 输入 act 来自 target 在 t-1 处的多层特征。
        """
        if self.extra_layers:
            raise NotImplementedError("autoregressive prefill supports num_layers=1 (midlayer only)")
        ids = prefix_ids[:, 1:]       # EAGLE shift：draft 不消费 prompt 首 token
        acts = target_acts_all[:, :-1, :]  # 位置 t 的 draft 条件 = target 在 t-1 的多层 act
        embeds = self.embed(ids)
        S = embeds.shape[1]
        if S == 0:
            raise ValueError("prefix too short for EAGLE3 draft prefill (need len >= 2)")
        reduced = self.fc(acts.to(embeds.dtype))
        positions = torch.arange(S, device=embeds.device)  # RoPE 0..S-1
        # 整段因果 self-attn，输出末 hidden + 长度为prefill tokens的 KV（供后续 extend；KV 条数 = prefill token 数）
        h, kv = self.midlayer.prefill(embeds, reduced, positions)
        last = self.norm(h)[:, -1, :]
        return self._lm_logits(last), h[:, -1, :], kv

    @torch.inference_mode()
    def extend_tokens(
        self,
        token_ids: torch.LongTensor,
        target_acts: torch.Tensor,
        past_kv,
        start_position: int,
    ):
        """	每轮 verify 后，在已有 draft KV 上增量 extend 本轮接受的 append token。

        Parameters
        ----------
        token_ids
            本轮 append [1, A]（target id）。
        target_acts
            与 append 对齐的 act 段 [1, A, n*H]。
        past_kv
            上一轮 draft KV。
        start_position
            append 首 token 的 RoPE 起始位置（= 旧 prefix_len - 1）。

        Returns
        -------
        root_logits, root_hidden, kv
            extend 末步 logits/hidden 与更新后的 KV（供下一轮 expand_draft_tree_ar）。
        """
        if self.extra_layers:
            raise NotImplementedError("autoregressive extend supports num_layers=1 (midlayer only)")
        A = int(token_ids.shape[1])
        if A == 0:
            raise ValueError("extend_tokens requires at least one token")
        if target_acts.shape[1] != A:
            raise ValueError(
                f"target_acts seq {target_acts.shape[1]} != token_ids seq {A}"
            )
        embeds = self.embed(token_ids)
        reduced = self.fc(target_acts.to(embeds.dtype))
        # 批量更新 KV等draft state
        h, kv = self.midlayer.extend(embeds, reduced, past_kv, int(start_position))
        last = self.norm(h)[:, -1, :]
        return self._lm_logits(last), h[:, -1, :], kv

    @torch.inference_mode()
    def topk_target_ids(self, logits: torch.Tensor, k: int):
        """logits top-k；独立词表时经 hot_token_id 映射回 target id。

        Parameters
        ----------
        logits
            [B, V_draft] 或 [B, V_target]。
        k
            保留候选数（不超过词表大小）。

        Returns
        -------
        idx, scores
            target 词表空间的 token id 与对应分数。
        """
        k = min(k, logits.shape[-1])
        scores, idx = torch.topk(logits, k=k, dim=-1)
        if self.hot_token_id is not None:
            idx = self.hot_token_id[idx]  # draft_id → target_id
        return idx, scores


def build_eagle3_sgl_draft(
    target: Union[PreTrainedModel, nn.Module],
    eagle_layers: Sequence[int],
    *,
    num_layers: int = 1,
    draft_vocab_size: Optional[int] = None,
    d2t: Optional[torch.Tensor] = None,
    draft_weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Eagle3SglDraftModel:
    """构造 EAGLE3 draft。

    权重来源由 draft_weights_path 控制：
    - draft_weights_path=None（默认）：随机初始化（结构对齐官方 EAGLE3，可跑通但接受率低，
      正确性仍由 target verify 保证）；
    - draft_weights_path=<dir 或 .safetensors/.bin/.pt>：加载 fine-tune 好的 EAGLE3 draft
      权重（fc / midlayer / lm_head / d2t），加载真实 checkpoint 可获得实际接受率。

    词表相关：
    - 不传 draft_vocab_size：复用 target lm_head（draft id 即 target id）。
    - 传 draft_vocab_size(+d2t) 或从 checkpoint 加载 d2t：独立 draft 词表 + lm_head，
      topk 后用 d2t 映射回 target id。加载真实 ckpt 时，draft_vocab_size 应与 ckpt 的
      draft 词表大小一致（= len(d2t)）。
    """
    m = Eagle3SglDraftModel(
        target, eagle_layers, num_layers=num_layers,
        draft_vocab_size=draft_vocab_size, d2t=d2t, device=device, dtype=dtype,
    )
    if draft_weights_path is not None:
        load_eagle3_draft_weights(m, draft_weights_path)
    return m


def _read_state_dict(path: str) -> dict:
    """从目录 / .safetensors / .bin / .pt 读出一个扁平的 name->tensor 状态字典。"""
    import glob
    import os

    files: List[str]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) or sorted(
            glob.glob(os.path.join(path, "*.bin"))
        ) or sorted(glob.glob(os.path.join(path, "*.pt")))
        if not files:
            raise FileNotFoundError(f"no weight file (*.safetensors/*.bin/*.pt) under {path}")
    else:
        files = [path]

    sd: dict = {}
    for f in files:
        # 目录下可能多个 shard：合并进同一 state_dict
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd.update(load_file(f))
        else:
            obj = torch.load(f, map_location="cpu", weights_only=True)
            sd.update(obj.get("state_dict", obj) if isinstance(obj, dict) else obj)
    return sd


def _translate_eagle3_name(name: str) -> str:
    """把官方 EAGLE3 checkpoint 的参数名翻译到本模块的命名。

    - 去掉可能的 model. 前缀；
    - 去掉 self_attn. 段（如 midlayer.self_attn.q_proj → midlayer.q_proj）；
    - 去掉 mlp. 段（如 midlayer.mlp.gate_proj → midlayer.gate_proj）。
    """
    if name.startswith("model."):
        name = name[len("model.") :]
    return name.replace("self_attn.", "").replace("mlp.", "")


def load_eagle3_draft_weights(
    draft_model: Eagle3SglDraftModel,
    path: str,
    *,
    strict: bool = False,
) -> dict:
    """把 fine-tune 好的 EAGLE3 draft 权重加载进 draft_model。

    处理：名称翻译（对齐官方 checkpoint 命名）、跳过 embed_tokens（复用 target）、
    由 d2t 计算 hot_token_id、按 my-layout 加载 fc/midlayer/norm/lm_head。

    Returns: {"loaded": [...], "missing": [...], "unexpected": [...]}。
    """
    src = _read_state_dict(path)
    local: dict = {}
    d2t = None
    for k, v in src.items():
        base = k.split(".")[-1]
        if base == "d2t" or k.endswith("d2t"):
            d2t = v
            continue
        if "t2d" in k or "embed_tokens" in k:
            continue  # t2d 不需要；embed 复用 target
        # 键名对齐本模块 Parameter 名（如去掉 self_attn. 前缀）
        local[_translate_eagle3_name(k)] = v

    # 由 d2t 设 hot_token_id（draft id -> target id）
    if d2t is not None and getattr(draft_model, "hot_token_id", None) is not None:
        hot = d2t.to(torch.long) + torch.arange(d2t.shape[0], dtype=torch.long)
        local["hot_token_id"] = hot.to(draft_model.hot_token_id.device)

    res = draft_model.load_state_dict(local, strict=strict)
    unexpected = set(res.unexpected_keys)
    return {
        "loaded": [k for k in local if k not in unexpected],
        "missing": list(res.missing_keys),
        "unexpected": list(res.unexpected_keys),
    }


def make_eagle3_sgl_draft_topk_fn(draft_model: Eagle3SglDraftModel, topk: int):
    """工厂：把 Eagle3SglDraftModel 包成 tree_draft.DraftTopkFn。

    若 draft 用独立词表（hot_token_id 非空），topk 得到的是 draft id，会经 d2t 映射成
    target vocab id 再返回（树与 verify 始终在 target 词表空间）。
    
    问题: 有了独立词表后，topk fn输出的draft topk ids为什么要映射回target id?
    答: 要统一draft 词表的 id 和 target 词表的 id 不是一回事。draft lm_head 输出的是
draft 词表空间里的下标（0..V_draft-1），它和 target 真实 token id 没有直接对应
    关系——d2t（draft-to-target）就是这个对应关系：target_id = hot_token_id[draft_id]
    （代码里 hot_token_id = d2t + arange）。

    回忆eagle1和eagle2，两步都在 target 词表空间工作：

    1.建 draft 树：树节点要存真实 token，下一步还要用它去 embed、拼前缀；
    2.target verify：target 用自己的完整词表逐位 argmax 比对，比较对象必须是 target id
    否则树里存的就是错的 id，verify 也无法对齐
    
    """

    @torch.inference_mode()
    def fn(
        acts: torch.Tensor, parent_token_id: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 与 expand_draft_tree 约定一致：parent_token 通常shape是 [B=batchsize,1]，acts 是根条件即低中高特征 [B,n*H]
        logits = draft_model(parent_token_id, acts)  # [B, V_draft 或 V_target]
        k = min(topk, logits.shape[-1])
        scores, idx = torch.topk(logits, k=k, dim=-1)
        if draft_model.hot_token_id is not None:
            idx = draft_model.hot_token_id[idx]  # draft id -> target id
        return idx, scores

    return fn
