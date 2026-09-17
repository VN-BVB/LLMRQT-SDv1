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

torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

torch::Tensor awq_gemm_db(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

torch::Tensor awq_gemm_cutlass(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);
