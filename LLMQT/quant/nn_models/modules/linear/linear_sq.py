import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from quant.utils.common_utils import get_best_device
from quant.utils.quantization_utils import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
)

user_has_been_warned = False
from .linear_base import LinearBase
class SqW8A8BBF16OBF16Linear(LinearBase):
    # For qkv_proj
    def __init__(self, in_features, out_features, bias, weight_scale=1.0, input_scale=1.0, alpha=1.0, beta=1.0, dev="cuda:0"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('qweight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False,
                                                                device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (self.out_features), dtype=torch.float16, requires_grad=False,device=dev)) # qwen2是bf16,opt是fp16
        else:
            self.bias = None
        self.register_buffer('weight_scale', torch.tensor(weight_scale,device=dev)) 
        self.register_buffer('input_scale', torch.tensor(input_scale,device=dev))

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # [batchsize, tokens, hiddendim] => [bs * tokens, hiddendim]
        x = x.view(-1, x_shape[-1]).to(self.qweight.device)

        # x_i8 = x.to(torch.bfloat16) / self.input_scale.item() # 这里输入本应该是int8，在后续runtime中会有int8 gemm，这里只有示意作用
        x_bf16 = x * self.input_scale.item()  # 反量化输入
        weight_bf16 = self.qweight.to(torch.bfloat16) * self.weight_scale.item()  # 反量化权重
        weight_bf16 = weight_bf16.t()
        y = torch.matmul(x_bf16, weight_bf16)  # FP16/BF16 计算
        if self.bias:
            y += self.bias #[xx, out feats] + [1, out feats]
        # [batchsize*tokens, out_feats] => [batchsize, tokens, out_feats]
        y = y.view(*x_shape[:-1], -1)
        return y

    # 正常 W8A8 的参考 forward（暂时注释，不改变当前运行逻辑）：
    # 1. 使用离线校准得到的 input_scale 将 BF16/FP16 激活量化成 INT8；
    # 2. INT8 激活与 INT8 权重相乘并使用 INT32 累加；
    # 3. 乘 input_scale * weight_scale 反量化，最终输出 BF16。
    #
    # 下面用 INT32 torch.matmul 表达数值正确的参考流程，但它不是高性能
    # INT8 Tensor Core Kernel。实际部署时应替换成 CUTLASS、cuBLASLt、
    # Triton 或其他支持 INT8 x INT8 -> INT32 的融合算子。
    #
    # @torch.no_grad()
    # def forward(self, x):
    #     x_shape = x.shape
    #     x = x.reshape(-1, x_shape[-1]).to(self.qweight.device)
    #
    #     # Quantize activation: BF16/FP16 -> INT8 (static per-tensor scale).
    #     input_scale = self.input_scale.to(device=x.device, dtype=torch.float32)
    #     input_scale = input_scale.clamp(min=1e-8)
    #     qinput = (
    #         torch.round(x.to(torch.float32) / input_scale)
    #         .clamp(-128, 127)
    #         .to(torch.int8)
    #     )
    #
    #     # Reference INT32 accumulation. qinput and qweight contain INT8 values. 后续会在cublass中使用
    #     acc_int32 = torch.matmul(
    #         qinput.to(torch.int32),
    #         self.qweight.t().to(torch.int32),
    #     )
    #
    #     # Dequantize: INT32 -> BF16.
    #     output_scale = input_scale * self.weight_scale.to(
    #         device=x.device,
    #         dtype=torch.float32,
    #     )
    #     y = (acc_int32.to(torch.float32) * output_scale).to(torch.bfloat16)
    #
    #     if self.bias is not None:
    #         y = y + self.bias.to(device=y.device, dtype=y.dtype)
    #
    #     return y.reshape(*x_shape[:-1], self.out_features)

    @staticmethod
    def from_linear(module: torch.nn.Linear, input_scale, dev="cuda:0"):
        int8_module = SqW8A8BBF16OBF16Linear(
            module.in_features, module.out_features, module.bias is not None)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        
        # 这里无论选择PerTensor或者PerChannel，weight scale的shape都要和register buffer的weight scale对的上才行
        int8_module.weight_scale = torch.tensor(weight_scale, device=dev) # scalar
        int8_module.input_scale = torch.tensor(input_scale, device=dev) # scalar
        int8_module.qweight = int8_weight # [out, in] row major
        if module.bias is not None:
            int8_module.bias = module.bias.clone()
        return int8_module
