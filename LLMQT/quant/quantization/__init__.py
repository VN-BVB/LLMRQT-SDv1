from .base.quantizer import BaseQuantizer
from .awq.quantizer import AwqQuantizer
from .fp8.quantizer import Fp8Quantizer
from .sq.quantizer import SqQuantizer
method_to_quantizer: dict[str, type[BaseQuantizer]] = {
    "awq": AwqQuantizer,
    "sq": SqQuantizer,
    "fp8_dynamic_quant": Fp8Quantizer,
    "fp8_static_quant": Fp8Quantizer,
    "fp8_native_dynamic_quant": Fp8Quantizer,
    "fp8_native_static_quant": Fp8Quantizer,
    
}
def get_concrete_quantizer_cls(quant_method):
    return method_to_quantizer[quant_method]
