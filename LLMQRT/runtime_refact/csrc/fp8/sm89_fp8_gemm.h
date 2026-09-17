#pragma once

#include <torch/types.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <optional>
#include <torch/torch.h>
#include <torch/extension.h>

// 这里不能放cutlass头文件，否则会报找不到blockIdx.x threadIdx.x syncthreads等错误，因为cpp文件不能包含cuda

// #include <ATen/native/cuda/cutlass_common.cuh>

// #include <ATen/Dispatch.h>
// #include <ATen/core/Tensor.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/macros/Macros.h>

torch::Tensor f8f8bf16_tensorwise(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    float alpha,
    float beta,
    const std::string& dtype_str);

torch::Tensor f8f8bf16_rowwise(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    bool use_fast_accum);


    // torch::Tensor XQ, // FP8
    // torch::Tensor WQ, // FP8
    // torch::Tensor x_scale, // FP32
    // torch::Tensor w_scale, // FP32
    // torch::Tensor bias, // BF16
    // bool use_fast_accum,
    // torch::Tensor& out);

