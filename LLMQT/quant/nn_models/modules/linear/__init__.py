from .linear_base import LinearBase
from .linear_awq import AWQLinear_GEMM
from .linear_sq import SqW8A8BBF16OBF16Linear
from .linear_fp8 import (
    FP8DynamicLinear,
    FP8NativeDynamicLinear,
    FP8NativeStaticLinear,
    FP8StaticLinear,
    FP8StaticLinearQuantizer,
)
method_to_linear: dict[str, type[LinearBase]] = {
    "awq": AWQLinear_GEMM,
    "sq": SqW8A8BBF16OBF16Linear,
    "fp8_static_quant": FP8StaticLinear, # per tensor
    "fp8_dynamic_quant": FP8DynamicLinear, # per tensor
    "fp8_native_static_quant": FP8NativeStaticLinear,
    "fp8_native_dynamic_quant": FP8NativeDynamicLinear,
    #TODO: support more grained quantization
}
def get_concrete_linear_module(quant_method):
    return method_to_linear[quant_method]
