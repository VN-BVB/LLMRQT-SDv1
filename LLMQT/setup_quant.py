from setuptools import setup, find_packages
from torch.utils import cpp_extension
        
setup(
    name="quant",
    version="0.1",
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False) # 必须禁用ninja，不然老是报import quant.nn_models找不到该module
    },
    packages=["quant", "quant.nn_models", "quant.core", "quant.utils", "quant.quantization",],
)
