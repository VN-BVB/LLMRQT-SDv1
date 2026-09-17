import transformers
import torch
import re
import gc
from typing import Tuple, Optional
from .linear_base import LinearBase

def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using dynamic per-tensor quant.
    Args:
        tensor: The input tensor.
    Return:
        qtensor: quantized act and their scales
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    # !!!!Note 因为torch.nn.Parameter不支持fp8的数据表示，所以这里不to fp8，即存储的时候还是以bf16/fp16存储，喂到kernel的时候再to fp8
    # qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale

def per_channel_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using dynamic per-tensor quant.
    Args:
        tensor: The input tensor.
    Return:
        qtensor: quantized act and their scales
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        print("[warning] You are experiencing empty MoE experts, tensor numbers = 0")
        qweight = torch.empty_like(tensor, dtype=torch.float8_e4m3fn)
        scales = torch.ones((*tensor.shape[:-1], 1), dtype=torch.float32)
        return qweight, scales
    amax = tensor.abs().amax(dim=-1, keepdim=True)
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    # 这里不to fp8，原因同上
    # qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale # [out, 1]

def static_per_tensor_quantize(tensor: torch.Tensor, static_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / static_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    native_fp8_support = False
    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        output, _ = torch._scaled_mm(
            A_input, # [m, in]
            B.t(), # [in, out]
            out_dtype=out_dtype,
            scale_a=A_scale,
            scale_b=B_scale,
            bias=bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
    else: # torch不支持fp
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale.to(out_dtype),
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )
    return output

def replace_module(model, name, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0] # model.layers.0.self_attn
        child_name = name[len(parent_name) + 1 :] # q_proj
        parent = model.get_submodule(parent_name)
        # Qwen2SdpaAttention(
        # (q_proj): Linear(in_features=5120, out_features=5120, bias=True)
        # (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
        # (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
        # (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
        # (rotary_emb): Qwen2RotaryEmbedding()
        # )
    else:
        parent_name = ""
        parent = model
        child_name = name
    # Qwen2SdpaAttention(
    # (q_proj): FP8DynamicLinear()
    # (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
    # (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
    # (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
    # (rotary_emb): Qwen2RotaryEmbedding()
    # )
    setattr(parent, child_name, new_module) # 把new module替换掉child name，成为parent的新child
    
# Class responsible for quantizing weights
class FP8DynamicLinear(LinearBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn,
        per_tensor=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qdtype = qdtype
        self.weight = torch.nn.Parameter(torch.randn((self.out_features, self.in_features) ,dtype=dtype, 
                                          device=dev, requires_grad=False)) # 只支持fp16/bf16/bf
        self.register_buffer("_weight_dequant_cache", None, persistent=False)
        self.per_tensor = per_tensor
        if self.per_tensor:
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (1) ,dtype=torch.float32, device=dev, requires_grad=False))
        else: #  per channel
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (self.out_features,1) ,dtype=torch.float32, device=dev, requires_grad=False))
        if bias:
            # 这里类型为bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
            self.bias = torch.nn.Parameter(torch.zeros( 
                (self.out_features), dtype=dtype, requires_grad=False, device=dev)) 
        else:
            self.bias = None
        
    @classmethod
    def from_linear(cls, module: torch.nn.Linear, group_size=0, zeros=None, per_tensor=True):
        assert group_size == 0, "not support group wise fp8 quant yet! pls set group_size = 0"  
        fp8_dynamic_linear = cls(
            module.in_features, module.out_features, module.bias is not None, per_tensor=True)

        # 这里无论选择PerTensor或者PerChannel，weight scale的shape都要和register buffer的weight scale对的上
        if module.bias is not None:
            fp8_dynamic_linear.bias.data = module.bias.clone()
        if per_tensor:
            fp8_weight, weight_scale = per_tensor_quantize(module.weight)
            fp8_dynamic_linear.weight_scale.data = torch.tensor(weight_scale, device="cuda:0").unsqueeze(0) # 加了unsqueeze，scalar => [1]
        else: # per channel
            fp8_weight, weight_scale = per_channel_quantize(module.weight)
            weight_scale = weight_scale.to("cuda:0")
            fp8_dynamic_linear.weight_scale.data = weight_scale.detach().clone() # To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()
        # fp8 weight在此处还是bf16 type fp8 val,fwd处to为fp8
        fp8_dynamic_linear.weight.data = fp8_weight # [out, in] row major

        return fp8_dynamic_linear
        
    def forward(self, x):
        # scale is computed in runtime, so naming dyn
        if self.per_tensor:
            qinput, x_scale = per_tensor_quantize(x)
        else:
            # per token
            qinput, x_scale = per_channel_quantize(x) # x.shape=[m,in] x_scale.shape = [m]==keep dim=>[m,1]
            self.weight_scale = self.weight_scale.t() # weight scale need to transpose from [out,1] to [1,out]
            # 此时torch._scaled_mm做rowwise gemm时, A=[m,in] B.t()=[in,out] A_scale=[m,1] B_scale=[1,out]，即act为per token量化，weight为per channel量化
            
        # The fallback kernel is BF16/FP16. Cache the FP8-rounded, dequantized
        # weight so every generated token does not convert the full model twice.
        if (
            self._weight_dequant_cache is None
            or self._weight_dequant_cache.device != self.weight.device
            or self._weight_dequant_cache.dtype != x.dtype
        ):
            self._weight_dequant_cache = (
                self.weight.to(self.qdtype).to(x.dtype)
                * self.weight_scale.to(x.dtype)
            )
        qinput = qinput.to(self.qdtype)
        output = torch.nn.functional.linear(
            qinput.to(x.dtype) * x_scale.to(x.dtype),
            self._weight_dequant_cache,
            bias=self.bias,
        )
        return output

# used in calib
# Module responsible for taking already quantized weights, and recording input scales (and possibly output scales) using an activation observer
class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        qdtype,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        quantize_output: bool = False,
    ):
        super().__init__()
        # 无需用Parameter
        self.qweight = weight #torch.nn.Parameter(weight, requires_grad=False)
        self.qdtype = qdtype
        self.weight_scale = weight_scale #torch.nn.Parameter(weight_scale, requires_grad=False)
        self.in_features = in_features
        self.out_features = out_features
        if bias is not None:
            self.bias = bias #torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None
        # 需要保存calibration得到的act input scale
        self.input_scale = None
        # 需要保存calibration得到的kv scale
        self.output_scale = None
        self.quantize_output = quantize_output

    def forward(self, x):
        qinput, x_input_scale = per_tensor_quantize(x) # observer
        # 后续修改：累计所有校准样本的最大 input scale，避免只保留最后一个样本的统计值。
        if self.input_scale is None:
            self.input_scale = x_input_scale
        else:
            self.input_scale = torch.maximum(self.input_scale, x_input_scale)
        qweight = self.qweight.to(self.qdtype)
        qinput = qinput.to(self.qdtype)
        output = fp8_gemm(
            A=qinput, # bf16
            A_scale=self.input_scale,
            B=qweight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype, # bf16
        )

        # Optionally, quantize output and record scale
        if self.quantize_output: # observer
            qoutput, output_scale = per_tensor_quantize(output)
            # 后续修改：量化 KV Cache 输出时同样累计最大 output scale。
            if self.output_scale is None:
                self.output_scale = output_scale
            else:
                self.output_scale = torch.maximum(self.output_scale, output_scale)
            output = qoutput.to(output.dtype) * output_scale # 这里我感觉应该是 / output scale啊？？160行的qoutput本就是output*outputscale得到的
        return output #fp16


# Module responsible for representing the final checkpoint representation
class FP8StaticLinear(LinearBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn, # TODO: 需要在quant tool中把e4m3或e5m2加入到quant config
        per_tensor=True, # only enable static quant when per tensor
        quantize_output=False
    ):
        super().__init__()        
        self.per_tensor = per_tensor
        self.qdtype = qdtype
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn((self.out_features, self.in_features) ,dtype=dtype, device=dev, requires_grad=False))
        self.register_buffer("_weight_dequant_cache", None, persistent=False)
        if self.per_tensor:
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (1) ,dtype=torch.float32, device=dev, requires_grad=False))
        else: # never go here
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (self.out_features, 1) ,dtype=torch.float32, device=dev, requires_grad=False))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False, device=dev)) 
        else:
            self.bias = None
        # static quant only support per tensor act 
        self.input_scale = torch.nn.Parameter(torch.randn((1) ,dtype=torch.float32, device=dev, requires_grad=False))
        self.quantize_output = quantize_output
        self.output_scale = torch.nn.Parameter(torch.randn((1) ,dtype=torch.float32, device=dev, requires_grad=False))
        
    @classmethod
    def from_linear(cls, in_features, out_features, fp8_weight, weight_scales, bias, input_scale=None, output_scale=None, group_size=0, zeros=None, quantize_output=False):
        assert group_size == 0, "not support group wise fp8 quant yet! pls set group_size = 0"  
        fp8_static_linear = cls(
            in_features, out_features, bias is not None, per_tensor=True, quantize_output=quantize_output)
        
        if bias is not None:
            fp8_static_linear.bias.data = bias.clone()
        if True:# fp8_static_linear.per_tensor always true
            fp8_static_linear.weight_scale.data = torch.tensor(weight_scales, device=fp8_weight.device) # scalar,这里无需unsqueeze，因为在dyn quant中unsqueeze过了
        fp8_static_linear.weight.data = fp8_weight # [out, in] row major
        fp8_static_linear.input_scale.data = input_scale.unsqueeze(0)
        if quantize_output:
            fp8_static_linear.output_scale.data = output_scale.unsqueeze(0)
        
        return fp8_static_linear
    
    def forward(self, x):
        # scale is known in advance, so naming static
        qinput = static_per_tensor_quantize(x, self.input_scale)
        if (
            self._weight_dequant_cache is None
            or self._weight_dequant_cache.device != self.weight.device
            or self._weight_dequant_cache.dtype != x.dtype
        ):
            self._weight_dequant_cache = (
                self.weight.to(self.qdtype).to(x.dtype)
                * self.weight_scale.to(x.dtype)
            )
        output = torch.nn.functional.linear(
            qinput.to(x.dtype) * self.input_scale.to(x.dtype),
            self._weight_dequant_cache,
            bias=self.bias,
        ) # fp16/fp32

        if self.quantize_output:
            qoutput = static_per_tensor_quantize(output, self.output_scale) # fp16/fp32 output / outputscale => fp8
            output = qoutput.to(output.dtype) * self.output_scale # fp8 * outputscale => fp16/32

        return output # fp16/fp32


class FP8NativeDynamicLinear(FP8DynamicLinear):
    """FP8 Linear whose checkpoint stores the normalized weight as real E4M3.

    The legacy FP8 modules keep normalized values in a BF16/FP16 Parameter and
    round/cast the full matrix during every Runtime forward.  This variant does
    that rounding once in LLMQT and keeps the same ``weight`` state-dict key so
    the rest of the model/checkpoint structure remains unchanged.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias: bool,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn,
        per_tensor=True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            dev=dev,
            dtype=dtype,
            qdtype=qdtype,
            per_tensor=per_tensor,
        )
        # A non-trainable Parameter is supported for FP8 by the PyTorch version
        # used by this project.  Keeping it as a Parameter also lets the existing
        # static-calibration observer register and move it without special cases.
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.out_features, self.in_features),
                dtype=self.qdtype,
                device=dev,
            ),
            requires_grad=False,
        )

    @classmethod
    def from_linear(
        cls,
        module: torch.nn.Linear,
        group_size=0,
        zeros=None,
        per_tensor=True,
    ):
        assert group_size == 0, "native FP8 does not support group-wise weight quantization"
        native_linear = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dev=module.weight.device,
            dtype=module.weight.dtype,
            per_tensor=per_tensor,
        )

        if module.bias is not None:
            native_linear.bias.data.copy_(module.bias.detach())

        if per_tensor:
            normalized_weight, weight_scale = per_tensor_quantize(module.weight)
        else:
            normalized_weight, weight_scale = per_channel_quantize(module.weight)

        # This is the defining difference from the legacy checkpoint: E4M3
        # rounding and storage happen once here instead of in every Runtime call.
        native_linear.weight.data.copy_(normalized_weight.to(native_linear.qdtype))
        native_linear.weight_scale.data.copy_(
            weight_scale.to(
                device=native_linear.weight_scale.device,
                dtype=native_linear.weight_scale.dtype,
            ).reshape_as(native_linear.weight_scale)
        )
        return native_linear


class FP8NativeStaticLinear(FP8StaticLinear):
    """Static-activation FP8 module with a real E4M3 checkpoint weight."""

    def __init__(
        self,
        in_features,
        out_features,
        bias,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn,
        per_tensor=True,
        quantize_output=False,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            dev=dev,
            dtype=dtype,
            qdtype=qdtype,
            per_tensor=per_tensor,
            quantize_output=quantize_output,
        )
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.out_features, self.in_features),
                dtype=self.qdtype,
                device=dev,
            ),
            requires_grad=False,
        )
