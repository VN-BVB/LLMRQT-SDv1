import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from runtime_refact.utils.common_utils import get_best_device
from runtime_refact.utils.quantization_utils import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    quantize_per_token_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)

from runtime.sq_fp8_kernels import w8a8_int8_linear_bbf16_obf16_per_tensor

user_has_been_warned = False
from .linear_base import LinearBase

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    qt = t.div(scales).round().clamp(-128, 127)
    qt = qt.to(torch.int8)
    return qt, scales

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.values
    scales.clamp_(min=1e-5).div_(q_max)
    qt = t.div(scales).round().clamp(-128, 127)
    qt = qt.to(torch.int8)
    return qt, scales #[m,1]

# 输入的处理：需要把x乘以input scale得到int8 x
# 输出的处理：直接s32=>bf16/fp16=s32 * (input scale * weight scale)
class SqW8A8BBF16OBF16PerTensor(LinearBase):
    # For qkv_proj
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev="cuda:0", dtype=torch.float16, weight_scale=1.0, input_scale=1.0, alpha=1.0, beta=1.0, impl_mode="cutlass"):
        super().__init__()
        self.w_bit = w_bit
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # impl_mode: "cutlass" 使用 cutlass kernel, "naive" 使用 dequant + torch.matmul
        assert impl_mode in ["cutlass", "naive"], f"impl_mode must be 'cutlass' or 'naive', got {impl_mode}"
        self.impl_mode = impl_mode
        
        self.register_buffer('qweight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False,
                                                                device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False,device=dev)) 
        else:
            self.bias = None

        # 这里weight scale的shape都要和quant tool选择的PerTensor或者PerChannel对的上才行
        self.register_buffer('weight_scale', torch.tensor(weight_scale, device=dev)) 
        self.register_buffer('input_scale', torch.tensor(input_scale, device=dev))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.qweight = self.qweight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x): # bf16/fp16 in, static quant to s8
        assert x.device == self.qweight.device, "when sq linear fwd, input and qweight must be same device!"
        assert self.weight_scale.device == self.qweight.device, "when sq linear fwd, scale and qweight must be same device!"
        assert self.input_scale.device == self.qweight.device, "when sq linear fwd, scale and qweight must be same device!"

        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).to(self.qweight.device)
        
        # 因为qkv对各自act的scale不一样，所以sq里面不fuse qkv        
        # dyn per tensor quant act
        qx, input_scale = quantize_activation_per_tensor_absmax(x) # per tensor qwen2胡言乱语 搭配torch.matmul时opt正常说话6.3(在取消了quantize函数中的inplace操作，以及增加了quantized val的clamp)
        input_scale = input_scale.to(self.qweight.device)
        
        # static quant act
        # x.div_(self.input_scale.item()).round_().clamp_(-128, 127) 有下划线的函数为原地操作
        # qx = x.to(torch.int8) # static quant, 这不是一个inplace操作需要换行
        
        alpha = input_scale * self.weight_scale.item() # dyn act quant
        # alpha = self.input_scale.item() * self.weight_scale.item() # static act quant
        self.impl_mode = "cutlass"
        if self.impl_mode == "cutlass":
            # ========== Cutlass Kernel 实现 ==========
            qweight = self.qweight.t() #[20480,5120]:[5120,1] => [5120,20480]:[1,5120]
            if self.bias is not None:
                y = w8a8_int8_linear_bbf16_obf16_per_tensor(qx, qweight, # 确保qweight.shape=[K,N],stride=[1,K]
                                                            self.bias,
                                                            alpha, 1.0)
            else:
                a = torch.zeros((self.out_features), dtype=x.dtype, requires_grad=False, device=self.qweight.device)
                y = w8a8_int8_linear_bbf16_obf16_per_tensor(qx, qweight, 
                    a, alpha, 1.0)
        
        elif self.impl_mode == "naive":
            # ========== Naive 实现: Dequantize + torch.matmul ==========
            # Dequantize activation: int8 -> bf16/fp16
            x_dequant = qx.to(x.dtype).mul(input_scale)  # dynamic quant scale
            # x_dequant = qx.to(x.dtype) * self.input_scale.item()  # static quant (alternative)
            
            # Dequantize weight: int8 -> bf16/fp16
            weight_dequant = self.qweight.to(x.dtype) * self.weight_scale.item()
            weight_dequant = weight_dequant.t()  # [out_features, in_features] -> [in_features, out_features]
            
            # Pure BF16/FP16 matmul
            y = torch.matmul(x_dequant, weight_dequant)
            
            # Add bias
            if self.bias is not None:
                y += self.bias.unsqueeze(0)  # [M, out_features] + [1, out_features]
        
        else:
            raise ValueError(f"Unknown impl_mode: {self.impl_mode}")
        
        y = y.view(*x_shape[:-1], -1)
        return y

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, w_bit, group_size, init_only=True, dtype=torch.float16, per_tensor=True, impl_mode="cutlass"):#, input_scale):
        sq_linear = cls(
            w_bit=w_bit,
            group_size=group_size,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dev=module.weight.device,
            dtype=dtype,
            impl_mode=impl_mode,
        )

        if init_only:
            return sq_linear
    
class SqW8A8BBF16OBF16PerChannel(LinearBase):
    # For qkv_proj
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev="cuda:0", dtype=torch.float16, weight_scale=None, input_scale=1.0, alpha=1.0, beta=1.0, impl_mode="cutlass"):
        super().__init__()
        self.w_bit = w_bit
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # impl_mode: "cutlass" 使用 cutlass kernel, "naive" 使用 dequant + torch.matmul
        assert impl_mode in ["cutlass", "naive"], f"impl_mode must be 'cutlass' or 'naive', got {impl_mode}"
        self.impl_mode = impl_mode
        
        self.register_buffer('qweight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False,
                                                                device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False,device=dev))
        else:
            self.bias = None
            
        # Per-channel weight scale: [out_features]
        if weight_scale is None:
            weight_scale = torch.ones((self.out_features), dtype=dtype)
        self.register_buffer('weight_scale', torch.tensor(weight_scale, dtype=dtype, device=dev))
        self.register_buffer('input_scale', torch.tensor(input_scale, device=dev))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.qweight = self.qweight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        assert x.device == self.qweight.device, "when sq linear fwd, input and qweight must be same device!"
        assert self.weight_scale.device == self.qweight.device, "when sq linear fwd, scale and qweight must be same device!"

        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).to(self.qweight.device)
        
        # Per-token quantization for activation
        qx, input_scale = quantize_activation_per_token_absmax(x) 
        input_scale = input_scale.to(self.qweight.device)
        
        # alpha: [M, out_features] = [M, 1] * [out_features], only support dyn act per token quant
        alpha = input_scale * self.weight_scale  # broadcasting: [M,1] * [out_features] = [M, out_features]
        
        if self.impl_mode == "cutlass":
            # ========== Cutlass Kernel 实现 (Per-Channel) ==========
            qweight = self.qweight.t() 
            if self.bias is not None:
                y = w8a8_int8_linear_bbf16_obf16_per_channel(qx, qweight, # 确保qweight.shape=[K,N],stride=[1,K]
                                                            self.bias,
                                                            alpha, 1.0)
            else:
                a = torch.zeros((self.out_features), dtype=x.dtype, requires_grad=False, device=self.qweight.device)
                y = w8a8_int8_linear_bbf16_obf16_per_channel(qx, qweight, 
                    a, alpha, 1.0)
        
        elif self.impl_mode == "naive":
            # ========== Naive 实现: Dequantize + torch.matmul (Per-Channel) ==========
            # Dequantize activation: int8 -> bf16/fp16, per-token
            x_dequant = qx.to(x.dtype) * input_scale  # [M, in_features] * [M, 1] -> [M, in_features]
            
            # Dequantize weight: int8 -> bf16/fp16, per-channel
            # weight_scale: [out_features] -> need to broadcast properly
            weight_dequant = self.qweight.to(x.dtype) * self.weight_scale.unsqueeze(1)  # [out_features, in_features] * [out_features, 1]
            weight_dequant = weight_dequant.t()  # [in_features, out_features]
            
            # Pure BF16/FP16 matmul
            y = torch.matmul(x_dequant, weight_dequant)
            
            # Add bias
            if self.bias is not None:
                y += self.bias.unsqueeze(0)  # [M, out_features] + [1, out_features]
        
        else:
            raise ValueError(f"Unknown impl_mode: {self.impl_mode}")
        
        y = y.view(*x_shape[:-1], -1)
        return y

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, w_bit, group_size, init_only=True, dtype=torch.float16, per_tensor=False, impl_mode="cutlass"):
        sq_linear = cls(
            w_bit=w_bit,
            group_size=group_size,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dev=module.weight.device,
            dtype=dtype,
            impl_mode=impl_mode,
        )

        if init_only:
            return sq_linear