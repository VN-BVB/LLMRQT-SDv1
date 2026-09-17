import os
from setuptools import setup, find_packages
import torch
from torch.utils import cpp_extension

# Use the real host/CUDA compilers. Recent PyTorch versions validate the
# compiler executable and reject a ccache wrapper as the CUDA compiler.
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['CUDA_NVCC_EXECUTABLE'] = '/usr/local/cuda/bin/nvcc'

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10 #900
project_root = os.path.abspath(os.path.dirname(__file__))
cutlass_path1 = os.path.join(project_root, "runtime_refact/3rdparty/cutlass/include")
cutlass_path2 = os.path.join(project_root, "runtime_refact/3rdparty/cutlass/tools/util/include")
cuda_path = "/usr/local/cuda/include"

setup(
    name="runtime",
    version="0.1",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='runtime.sq_fp8_kernels',
            sources=[
                'runtime_refact/csrc/awq/gemm.cu',
                'runtime_refact/csrc/awq/gemm_cutlass.cu',
                'runtime_refact/csrc/awq/gemm_db.cu',
                'runtime_refact/csrc/awq/gemv.cu',
                'runtime_refact/csrc/awq/gemv_coalesced.cu',
                'runtime_refact/csrc/smoothquant/sq_gemm.cu',
                'runtime_refact/csrc/fp8/sm89_fp8_rowwise_fbgemm.cu',
                'runtime_refact/csrc/fp8/sm89_fp8_tensorwise_cutlassgemm.cu',
                'runtime_refact/csrc/bindings.cpp',
            ],
            #  'runtime/csrc/smoothquant', cutlass_path1, cutlass_path2, cuda_path,
            include_dirs=[ 'runtime_refact/csrc/awq', 'runtime_refact/csrc/fp8', 'runtime_refact/csrc/smoothquant', cutlass_path1, cutlass_path2, cuda_path, torch.utils.cpp_extension.include_paths()],
            extra_link_args=['-lculibos',
                             '-lcudart', '-lcudart_static',
                             '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/',],

            extra_compile_args={'cxx': ['-std=c++17', '-O3', f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}'],
                                'nvcc': ['-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', \
                                         '-U__CUDA_NO_HALF2_OPERATORS__', f'-DCUDA_ARCH={cuda_arch}',f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}','-DCUTLASS_SM90_ENABLED=1','-DCUTLASS_DEBUG=0', \
                                         '-gencode=arch=compute_80,code=sm_80', \
                                         '-gencode=arch=compute_89,code=sm_89',]},
                                        #  '-gencode=arch=compute_90,code=sm_90',]},
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False) # 必须禁用ninja，不然老是报import runtime.autoAWQ_models找不到该module
    },
    packages=[#"runtime_refact", "runtime_refact.nn_models", "runtime_refact.triton_kernels", "runtime_refact.utils", "runtime_refact.core"
        ],
)
