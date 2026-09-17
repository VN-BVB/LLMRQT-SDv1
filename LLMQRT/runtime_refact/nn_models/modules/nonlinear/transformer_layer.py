import os
import torch.nn as nn
import torch
from runtime_refact.nn_models.modules.nonlinear.attention import QuantAttentionFused


class LlamaFamilyLayer(nn.Module):
    """
    LlamaFamilyLayer旨在被llama,mistral等一系列相似架构的模型复用.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        mlp,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        use_alibi=False,
        head_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        # To support gemma-7b, its head_dim is separate
        if head_dim:
            self.head_dim = head_dim

        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            max_seq_len=max_seq_len,
            use_alibi=use_alibi,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            head_dim=head_dim,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
    ):
        norm_out = self.norm_1(hidden_states) # 这个造成的缺失值 hiddenstates=[1,61,5120],原因在于iter0输出的hidden states有inf
        attn_output, _, _ = self.attn.forward(
            hidden_states=norm_out,
        )
        h = hidden_states.to(attn_output.device) + attn_output# fp32 = fp16 + bf16, 导致h为fp32了, sq_run_qwen2.py里面
        # 下一步的down proj完成后出现无穷大
        # 已在vi /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/modeling_qwen2.py#223行打断点
        # import pdb;pdb.set_trace()
        out = h + self.mlp.forward(self.norm_2(h))

        return out

class QwenFamilyLayer(nn.Module):
    """
    QwenFamilyLayer旨在被Qwen家族模型Qwen2/Qwen3等复用, 避免重复代码.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        mlp,
        norm_1,
        norm_2,        
        dev,
        max_seq_len,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        use_alibi=False,
        head_dim=None,
        q_norm=None,
        k_norm=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        # To support qwen3, its head_dim is separate
        if head_dim:
            self.head_dim = head_dim

        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            max_seq_len=max_seq_len,
            use_alibi=use_alibi,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            head_dim=head_dim,
            q_norm=q_norm,
            k_norm=k_norm,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, _ = self.attn.forward(
            hidden_states=norm_out,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.mlp.forward(self.norm_2(h))

        return out
