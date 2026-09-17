from .linear_base import LinearBase
from .linear_awq import AWQLinear_GEMM, WQLinear_GEMM
from .linear_fp8 import (
    FP8DynamicLinear,
    FP8NativeDynamicLinear,
    FP8NativeStaticLinear,
    FP8StaticLinear,
)
from .linear_sq import SqW8A8BBF16OBF16PerTensor
method_to_linear: dict[str, type[LinearBase]] = {
    "awq": AWQLinear_GEMM,
    "sq": SqW8A8BBF16OBF16PerTensor, # 待改进，把per channel引入
    "fp8_static_quant": FP8StaticLinear, # per tensor only
    "fp8_dynamic_quant": FP8DynamicLinear, # per tensor default, per token available too
    "fp8_native_static_quant": FP8NativeStaticLinear,
    "fp8_native_dynamic_quant": FP8NativeDynamicLinear,
}

def get_concrete_linear_module(quant_method):
    return method_to_linear[quant_method]


    
