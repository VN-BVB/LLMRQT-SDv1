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

torch::Tensor awq_gemv(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int group_size);

torch::Tensor awq_gemv_coalesced(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int group_size,
  int version);

