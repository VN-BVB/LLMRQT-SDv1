import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from runtime_refact.nn_models.modules.nonlinear.KVcache import SlidedCache # 注意，此windowcache不是SWA里的window的意思，只是在句子长度超出了maxseqlen的时候，自动lru
from runtime_refact.utils.fused_utils import get_attention_shapes


try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache # 原来是2.4.2，现在是2.7.4（不过这个报了不兼容某些包
    # from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
    FA_INSTALLED = True
except:
    FA_INSTALLED = False

HF_NEW_CACHE_FORMAT = False

import transformers

# https://github.com/huggingface/transformers/pull/26681 introduced a new cache format
HF_NEW_CACHE_FORMAT = hasattr(transformers, "cache_utils")
if HF_NEW_CACHE_FORMAT:
    from transformers.cache_utils import DynamicCache


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, device, rope_theta):
        super(RoPE, self).__init__()

        self.head_dim = head_dim
        self.freqs_cis = nn.Parameter(
            self.precompute_freqs_cis(head_dim, max_seq_len, rope_theta).to(device),
            requires_grad=False,
        )
    # ROPE公式
    # 输入输出都是headdim维度相邻两个元素[0,1],[2,3]...为一组的二维向量，m=0,1,...d/2-1
    # k= token index, seq_len = 10, k = 0 ~ 9
    # theta_m = 10000^(-2m/d),d = head_dim
    # y_2m = x_2m * cos(k * theta_m) - x_2m+1 * sin(k * theta_m)
    # y_2m+1 = x_2m * sin(k * theta_m) + x_2m+1 * cos(k * theta_m)
    # y_m =x_m cos(α_m)- x_m+D/2 sin(α_m)
    # y_m+D/2 =x_m sin(α_m)+ x_m+D/2 cos(α_m)
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta=10000.0):
        # 将一维的词向量按相邻两个维度划分为若干二维组，m 就是这些二维组的index
        # 比如将长度为 d 的词向量拆分成 d/2 个独立的二维向量组：
        # [x0,x1], [x2,x3], ..., [x_{d-2},x_{d-1}]，m=0,1,...,d/2-1。
        # 乘2后即 torch.arange(0, dim, 2) = [0,2,4,...,d-2]。
        # torch.arange(0, 7, 2)# [0, 2, 4, 6] 0 -- 6
        # torch.arange(0, 7, 2)[:3]# [0, 2, 4]
        # 因为 7 个维度只能组成 3 个完整二维组，第 7 个维度没有配对，所以被截掉。
        # 因此整段的含义是：为每一对维度生成一个频率下标，只保留能够组成完整二维组的部分。这里相当于2m

        # theta_m=10000^(-2m/d)
        # shape: [dim / 2]；例如 dim=64 时为 [32]。
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # 生成所有token的id: k
        # shape: [end]；例如 max_seq_len=2048 时为 [2048]。
        t = torch.arange(end)
        # k * theta_m
        # shape: [end, dim / 2]；例如 [2048, 32]。
        freqs = torch.outer(t, freqs).float()

        # 复数形式 cos(k * theta_m) + i sin(k * theta_m)
        # shape保持为 [end, dim / 2]，但每个元素是一个复数；例如 [2048, 32]。
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis
    
    # 例：实数 xq=[1,61,32,64] 经 view_as_complex 后为 [1,61,32,32]，
    # 所以 freqs_cis 必须是 [61,32]，广播形状为 [1,61,1,32]。
    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int,
        seqlen: int,
        partial: bool = False,
    ):
        if partial:
            xq, xq_pass = (
                xq[..., : self.head_dim],
                xq[..., self.head_dim :],
            )
            xk, xk_pass = (
                xk[..., : self.head_dim],
                xk[..., self.head_dim :],
            )
        # 复数形式一次性把 sin/cos 打包，后面直接做一次复乘就完成旋转，比“先拆实部虚部再分别乘”快且省显存
        # 把最后一维 head_dim 两两一组当成「实部、虚部」(x_2m, x_2m+1)→ 复数张量；
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        ) # shape: [1,61,32,64] -> [1,61,32,2,32] -> [1,61,32,32,2] -> [1,61,32,32](复数) 
        # 这里是分成了前后半区  x_m + i * x_ m+D/2 
        # 如果是相邻 xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous())  == > x_2m + i * x_ 2m+1 
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        )
        # shape: [seqlen, dim / 2]；本例 dim=64、seqlen=61，所以是 [61,32]。
        # 如果 dim=128，这里才会是 [61,64]。
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        # [61,32] -> [1,61,1,32]，随后与复数 xq_=[1,61,32,32] 广播相乘。
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_).to(xq_.device)
        # 复数乘法结果仍为 [1,61,32,32]；view_as_real 后为 [1,61,32,32,2]， -> [1,61,32,2,32] - flatten(3)-> [1,61,32,64]，恢复为实数。
        # transpose + flatten 后恢复为实数 [1,61,32,64]，再送给 Flash-Attention。
        # q rope
        # (xq_2m + xq_2m+1 * i) * (cos(k*theta_m) + sin(k*theta_m) * i) =
        #  (xq_2m*cos(k*theta_m)-xq_2m+1*sin(k*theta_m)
        #  + 
        # i * (xq_2m*sin(k*theta_m)+xq_2m+1*cos(k*theta_m)))
        xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-2, -1).flatten(3)
        # k rope
        xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-2, -1).flatten(3)

        if partial:
            xq = torch.cat((xq, xq_pass), dim=-1)
            xk = torch.cat((xk, xk_pass), dim=-1)

        return xq_out.type_as(xq), xk_out.type_as(xk)


class ALiBi(nn.Module):
    def __init__(self, n_heads, max_seq_len, device, alibi_bias_max=8):
        super(ALiBi, self).__init__()

        # Initialize ALiBi slopes and bias
        slopes, bias = self.build_alibi_bias(
            n_heads, max_seq_len, alibi_bias_max=alibi_bias_max
        )
        self.slopes = nn.Parameter(slopes.float().to(device), requires_grad=False)
        self.bias = nn.Parameter(bias.float().to(device), requires_grad=False)

    @staticmethod
    def gen_slopes(n_heads, alibi_bias_max=8):
        _n_heads = 2 ** math.ceil(math.log2(n_heads))
        m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
        m = m.mul(alibi_bias_max / _n_heads)
        slopes = 1.0 / torch.pow(2, m)

        if _n_heads != n_heads:
            slopes = torch.cat([slopes[1::2], slopes[::2]])[:n_heads]

        return slopes.view(1, n_heads, 1, 1)

    @staticmethod
    def build_alibi_bias(n_heads, seq_len, alibi_bias_max=8, dtype=torch.float32):
        alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32).view(
            1, 1, 1, seq_len
        )
        slopes = ALiBi.gen_slopes(n_heads, alibi_bias_max)
        alibi_bias = alibi_bias * slopes
        slopes = slopes.squeeze(0).squeeze(-1).squeeze(-1)
        return slopes.to(dtype=dtype), alibi_bias.to(dtype=dtype)

    def forward(self, scores, seqlen):
        scores += self.bias[..., :seqlen]
        return scores


class QuantAttentionFused(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        dev,
        q_proj=None,
        k_proj=None,
        v_proj=None,
        max_seq_len=2048,
        use_alibi=False,
        attention_shapes=None,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        head_dim=None,
        attn_logit_softcapping=0.0,
        q_norm=None,
        k_norm=None,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        # for mqa/gqa
        self.n_kv_groups = n_heads // n_kv_heads if n_kv_heads != 0 else 0
        self.head_dim = head_dim

        if head_dim is None:
            self.head_dim = hidden_size // n_heads

        self.qkv_proj = qkv_layer
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj        
        self.o_proj = o_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.start_pos = 0
        self.use_alibi = use_alibi
        self.cache_batch_size = int(os.getenv("BATCH_SIZE", "1"))

        if kwargs.get("max_length") is not None:
            max_seq_len = kwargs["max_length"]

        self.max_seq_len = max_seq_len
        self.is_hf_transformers = False
        self.rope_theta = rope_theta

        # attention shapes for self attention
        self.attention_shapes = get_attention_shapes(
            attention_shapes,
            n_heads,
            n_kv_heads,
            self.head_dim,
        )
        # cache store that rolls cache
        self.cache = SlidedCache(
            cache_batch_size=self.cache_batch_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            device=dev,
        )

        if use_alibi:
            self.alibi = ALiBi(n_heads, max_seq_len, dev)
            self.rotary_dim = 0
            self.is_neox = False
        else:
            self.alibi = None
            self.partial_rotary_factor = partial_rotary_factor
            self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
            self.rope = RoPE(self.rotary_dim, max_seq_len, dev, rope_theta)
            self.is_neox = True

        if kwargs.get("is_neox") is not None:
            self.is_neox = kwargs["is_neox"]

        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        bsz, seqlen, _ = hidden_states.shape

        # Reallocate cache if batch size changes
        if bsz != self.cache_batch_size:
            if bsz > self.cache_batch_size:
                self.cache.increase_batch_size(bsz)
                self.cache_batch_size = bsz
            elif bsz < self.cache_batch_size:
                self.cache.decrease_batch_size(bsz)
                self.cache_batch_size = bsz

            # Always reset to 0
            self.start_pos = 0
        # hf_xxxx全部都不会跑到，忽略即可
        hf_is_generating = False
        hf_is_first_forward = (
            "past_key_value" in kwargs and kwargs["past_key_value"] is None
        )
        hf_is_new_cache_first_forward = (
            "past_key_value" in kwargs
            and isinstance(kwargs["past_key_value"], DynamicCache)
            and kwargs["past_key_value"].get_seq_length() == 0
        )

        if self.is_hf_transformers and "use_cache" in kwargs:
            hf_is_generating = kwargs["use_cache"]

        # In case we re-generate, we need to refresh the starting position
        # to 0. We detect it by checking if `past_key_values` is set to None,
        # which indicates that we are on the first step of `generate()`.
        # This is only applicable for `transformers` integration
        if (
            self.is_hf_transformers
            and (hf_is_first_forward or hf_is_new_cache_first_forward)
        ) or (self.is_hf_transformers and not hf_is_generating):
            self.start_pos = 0
        # unfused qkv
        if self.q_proj is not None: # handle sq cant fuse qkv due to different scale for qkv act
            # 需要view成后面的shape，不然rope里面broadcast那里会assert报错
            xq = self.q_proj(hidden_states).view(bsz, seqlen, self.n_heads, self.head_dim)
            xk = self.k_proj(hidden_states).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            xv = self.v_proj(hidden_states).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # fused qkv
        else:
            xqkv = self.qkv_proj(hidden_states)
            xqkv = xqkv.view((bsz, seqlen) + self.attention_shapes["xqkv_view"])
            # import pdb;pdb.set_trace() # 检查nan是由哪里造成
            xq = self.attention_shapes["xq_slice"](xqkv)
            xk = self.attention_shapes["xk_slice"](xqkv)
            xv = self.attention_shapes["xv_slice"](xqkv)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        if not self.use_alibi:
            xq, xk = self.rope.forward(
                xq, xk, self.start_pos, seqlen, partial=self.partial_rotary_factor < 1
            )

        self.cache.to(xq)
        self.cache.update_kv(
            values_store=xv,
            keys_store=xk,
            batch_size=bsz,
            start_pos=self.start_pos,
            seqlen=seqlen,
        )
        #prefill阶段，无kv cache
        if seqlen > 1:
            # (Pdb) xq.shape
            # torch.Size([1, 61, 40/*n_head*/, 128])
            # (Pdb) xk.shape
            # torch.Size([1, 61, 8/*n_kv_head*/, 128])
            # (Pdb) xv.shape
            # torch.Size([1, 61, 8/*n_kv_head*/, 128])
            output = flash_attn_func( 
                q=xq,
                k=xk,
                v=xv,
                causal=True,
                softcap=self.attn_logit_softcapping,
            )
        # decode阶段，有kv cache
        else:
            # 创建(bsz,)形状，值全是self.start+seqlen的tensor
            cache_seqlens = torch.full(
                (bsz,), self.start_pos + seqlen, dtype=torch.int32, device=xq.device
            )

            output = flash_attn_with_kvcache(
                q=xq,
                k=None,#xk
                k_cache=self.cache.k,
                v=None,#xv
                v_cache=self.cache.v,
                cache_seqlens=cache_seqlens,
                causal=True,
                # alibi_slopes=self.alibi.slopes if self.alibi is not None else None,
                softcap=self.attn_logit_softcapping,
            )
        # (bsz, seqlen, n_heads, head_dim)=>(bsz, seqlen, hidden size)
        attention_weight = output.view(bsz, seqlen, -1) 
        # import pdb;pdb.set_trace()
        # bug: 经过一个o proj之后output为tensor(..., device='meta', size=(1, 61, 5120), dtype=torch.float16)
        # fixed: 由linear_awq.py里面引起，多余的x = x.to(self.device)，在o proj之前self.device变成了meta
        # self.oproj的所有参数都在cuda0
        attn_output = self.o_proj(attention_weight)
        # prefill阶段seqlen>1,decode阶段seqlen=1
        self.start_pos += seqlen

        if self.is_hf_transformers and not hf_is_generating:
            self.start_pos = 0

        # past_key_value is replaced with cache_v, cache_k, returning empty data
        # we pass a dummy past kv cache for transformers to be able to retrieve the correct info
        # about past key length
        past_key_value = [torch.zeros(1, 1, self.start_pos, 1)]

        if HF_NEW_CACHE_FORMAT and self.is_hf_transformers:
            new_cache = DynamicCache()
            new_cache.update(past_key_value[0], past_key_value[0], layer_idx=0)
            past_key_value = new_cache

        return attn_output, attention_weight, past_key_value
