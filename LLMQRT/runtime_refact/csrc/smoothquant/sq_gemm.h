#pragma once

#include <torch/types.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor w8a8_int8_linear_bbf16_obf16_per_tensor(torch::Tensor input,  // INT8
                                                    torch::Tensor weight, // INT8
                                                    torch::Tensor bias,   // BF16
                                                    float alpha,          // BF16
                                                    float beta            // BF16
);

torch::Tensor w8a8_int8_linear_bbf16_obf16_per_channel(torch::Tensor input,  // INT8
                                                        torch::Tensor weight, // INT8
                                                        torch::Tensor bias,   // BF16
                                                        torch::Tensor alpha,  // BF16 ,暂不确定是否是torch::Tensor类型
                                                        torch::Tensor beta    // BF16
);