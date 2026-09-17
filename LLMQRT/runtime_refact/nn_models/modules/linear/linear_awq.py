import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from runtime_refact.utils.common_utils import get_best_device
from runtime_refact.utils.packing_utils import dequantize_gemm

# NOTE: We check if awq_ext or triton is available. awq_ext will be preferred if both are installed.

user_has_been_warned = False
# awq的两个kernel形式：triton dq + torch.matmul或者triton fused dq gemm
# 和quant里的linear awq一致，只是fwd部分换成了triton
try:
    from runtime.triton_kernels.awq_kernels import awq_gemm_triton, awq_dequantize_triton

    # covers CUDA, ROCm and XPU. If we can import triton, then we can use it.
    TRITON_AVAILABLE = True

except ImportError:
    TRITON_AVAILABLE = False

# Try to import CUDA AWQ kernels
try:
    from runtime.sq_fp8_kernels import awq_gemm, awq_gemv
    AWQ_CUDA_AVAILABLE = True
except ImportError:
    AWQ_CUDA_AVAILABLE = False

# Try to import optimized coalesced version
try:
    from runtime.sq_fp8_kernels import awq_gemv_coalesced
    AWQ_CUDA_COALESCED_AVAILABLE = True
except ImportError:
    AWQ_CUDA_COALESCED_AVAILABLE = False

# Try to import double buffering version
try:
    from runtime.sq_fp8_kernels import awq_gemm_db
    AWQ_CUDA_DB_AVAILABLE = True
except ImportError:
    AWQ_CUDA_DB_AVAILABLE = False
    
from .linear_base import LinearBase


def gemv_forward(x, qweight, scales, qzeros, in_features, group_size, use_coalesced=False, coalesced_version=2):
    """
    GEMV forward pass using CUDA kernel for small batch sizes.
    
    Args:
        x: input tensor of shape [batch_size, seq_len, in_features]
        qweight: quantized weight of shape [in_features, out_features // 8]
        scales: scaling factors of shape [in_features // group_size, out_features]
        qzeros: quantized zeros of shape [in_features // group_size, out_features // 8]
        out_features: number of output features
        in_features: number of input features
        group_size: group size for quantization
        use_coalesced: whether to use coalesced memory access version (default: True)
        coalesced_version: 1 for basic coalesced, 2 for cached scales/zeros (default: 2)
    
    Returns:
        output tensor of shape [batch_size, seq_len, out_features]
    """
    if not AWQ_CUDA_AVAILABLE:
        raise RuntimeError("AWQ CUDA kernels are not available. Please compile the extension.")
    
    # Reshape input to [batch_size * seq_len, in_features]
    orig_shape = x.shape
    x_2d = x.reshape(-1, in_features)
    
    # Choose kernel version
    if use_coalesced and AWQ_CUDA_COALESCED_AVAILABLE:
        # Use optimized coalesced version
        out = awq_gemv_coalesced(x_2d, qweight, scales, qzeros, group_size, coalesced_version)
    else:
        # Fall back to original version
        out = awq_gemv(x_2d, qweight, scales, qzeros, group_size)
    #import pdb;pdb.set_trace()
    out_features = out.shape[-1]
    # Reshape output back to original shape
    out = out.reshape(orig_shape[:-1] + (out_features,))
    return out


def gemm_forward(x, qweight, scales, qzeros, split_k_iters=8, use_double_buffer=True):
    """
    GEMM forward pass using CUDA kernel for larger batch sizes.
    
    Args:
        x: input tensor of shape [batch_size, seq_len, in_features]
        qweight: quantized weight of shape [in_features, out_features // 8]
        scales: scaling factors of shape [in_features // group_size, out_features]
        qzeros: quantized zeros of shape [in_features // group_size, out_features // 8]
        split_k_iters: number of split-K iterations for better parallelism
        use_double_buffer: whether to use double buffering optimization (default: True)
    
    Returns:
        output tensor of shape [batch_size, seq_len, out_features]
    """
    if not AWQ_CUDA_AVAILABLE:
        raise RuntimeError("AWQ CUDA kernels are not available. Please compile the extension.")
    
    # Reshape input to [batch_size * seq_len, in_features]
    orig_shape = x.shape
    in_features = x.shape[-1]
    x_2d = x.reshape(-1, in_features)
    
    # Choose kernel version
    if use_double_buffer and AWQ_CUDA_DB_AVAILABLE:
        # Use double-buffered version for better performance
        out = awq_gemm_db(x_2d, qweight, scales, qzeros, split_k_iters)
    else:
        # Fall back to original version
        out = awq_gemm(x_2d, qweight, scales, qzeros, split_k_iters)
    
    # Reshape output back to original shape
    # out.shape=[61,6144]
    # orig_shape=[1,61,4096]
    out_features = out.shape[-1]
    out = out.reshape(orig_shape[:-1] + (out_features,))
    # after reshape, out.shape=[1,61,6144]
    return out

# Adapted from https://github.com/compressa-ai/AutoAWQ/tree/dev
class AWQLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x, #fp16
        qweight,#(in_features, out_features // 8)
        qzeros, #(in_features // self.group_size, out_features// 8)
        scales,#(in_features // self.group_size, out_features)
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
        impl_mode="auto",
    ):
        # The forward pass can use ctx.
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features
        in_features = qweight.shape[0]
        # [1,1,6144]
        out_shape = x.shape[:-1] + (out_features,)
        # x = x.to(torch.float16)
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        
        if impl_mode not in {"auto", "dequant_matmul"}:
            raise ValueError(
                "AWQ impl_mode must be 'auto' or 'dequant_matmul', "
                f"got {impl_mode!r}"
            )

        if impl_mode == "dequant_matmul":
            # Explicit reference path: materialize the complete FP16 weight
            # matrix on every Linear invocation, then call torch.matmul.
            out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
            out = torch.matmul(x, out)
        # Small-token decode/verify should use one numerical path.  EAGLE3
        # verifies [pending, draft] with q_len=2 while ordinary greedy decode
        # uses q_len=1.  Dispatching those shapes to different GEMV/GEMM
        # kernels can change an argmax near a logit tie and violates strict
        # losslessness.  The GEMV kernel already flattens all leading token
        # dimensions, so use it for every small-token input (< 8 rows).
        elif x.numel() / x.shape[-1] < 8:
            # print("[info] invoke awq gemv")
            out = gemv_forward(
                x,
                qweight,
                scales,
                qzeros,
                in_features,
                group_size,
            )
            # out.shape[1,1,4096]
            
        else:
            # print("[info] invoke awq gemm")
            # awq_run_qwen2.py gemm测试通过，输出正确
            # 所有prefill阶段， 或者 decode 阶段且bs> 8 这个临界在每个gpu不一样
            out = gemm_forward(
                x, qweight, scales, qzeros
            )
        # if TRITON_AVAILABLE:
        #     FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

        #     if FP16_MATMUL_HEURISTIC_CONDITION:
        #         out = awq_dequantize_triton(qweight, scales, qzeros)
        #         out = torch.matmul(x, out.to(x.dtype))
        #     else:
        #         out = awq_gemm_triton(
        #             x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, split_k_iters=8,
        #         )

        # else:
        #     global user_has_been_warned
        #     if not user_has_been_warned:
        #         warnings.warn("Using naive (slow) implementation." + msg)
        #         user_has_been_warned = True
        #     out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        #     out = torch.matmul(x, out)
        # import pdb;pdb.set_trace()
        out = out + bias if bias is not None else out
        # out = out.reshape(out_shape) 已经在gemm_fwd和gemv_fwd里面reshape

        # always want 3D tensor if tensor is 2D
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out


# 每个量化方法都要像 AWQLinear 这样实现 init、from_linear 和 forward
class AWQLinear_GEMM(LinearBase):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev, training=False
    ):
        super(LinearBase, self).__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training
        # "auto": current GEMV/GEMM dispatch; "dequant_matmul": reference
        # implementation that fully dequantizes weights for every forward.
        self.impl_mode = "auto"
        # self.device = dev
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight", 
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros", # 既group size量化了，也pack了
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, 
        linear, 
        w_bit, 
        group_size, 
        init_only=False,
        dtype=torch.bfloat16, 
        scales=None, 
        zeros=None,
        per_tensor=False
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        # import pdb;pdb.set_trace()
        # awq_linear.qweight = linear.qweight
        # awq_linear.scales = linear.scales
        # awq_linear.qzeros = linear.qzeros
        # awq_linear.bias = linear.bias if linear.bias is not None else None
        
        # 是否还需要赋值一下scales和zeros，weight等
        if init_only:  # just prepare for loading sd
            return awq_linear
        # runtime里面，下面都没用，可以删了
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []# in feats维度上group wise quantize weight and save it by int32
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)

        # best_device = get_best_device()

        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device,
        )

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):#0-7
                # 把intweight上的8个int4按照order map来排进qweight的32bit
                # 然后在triton dq的时候再按照reverse order map取出对应的4bit，猜测可能和triton interleave机制有关
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32, device=intweight.device)#best_device)

        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        return awq_linear

    def forward(self, x):
        assert x.device == self.qweight.device, "when awq linear fwd, input and qweight must be same device!"
        assert self.qzeros.device == self.qweight.device, "when awq linear fwd, qzeros and qweight must be same device!"
        assert self.scales.device == self.qweight.device, "when awq linear fwd, scales and qweight must be same device!"
        # x.shape=[1,61,4096]
        # qweight.shape=[4096,6144/8]
        # out_shape=[1,61,6144]
        out_shape = x.shape[:-1] + (self.out_features,)
        # for none experts  https://github.com/casper-hansen/AutoAWQ/pull/751/files
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        input_dtype = x.dtype # torch.bfloat16
        # if input_dtype != torch.float16:
        #     x = x.half()
        if input_dtype != torch.float16:
            x = x.to(torch.float16) # triton only support fp16

        with torch.no_grad():
            out = AWQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
                self.impl_mode,
            )
        # when finish triton, cast to original dtype
        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        return out#.reshape(out_shape), 已经在gemm_fwd和gemv_fwd里面reshape

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )


# 兼容旧名称：旧代码仍可导入，但实际类名和模型打印使用 AWQLinear_GEMM。
WQLinear_GEMM = AWQLinear_GEMM
WQLinearMMFunction = AWQLinearMMFunction
