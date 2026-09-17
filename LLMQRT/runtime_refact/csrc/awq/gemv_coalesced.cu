#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "gemv_cuda.h"

#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define WARP_SIZE 32


// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}

// Reverse order map for AWQ weight packing
// Original packing order: [0, 2, 4, 6, 1, 3, 5, 7]
// To get original position i, look at packed position reverse_order_map[i]
__device__ __forceinline__ int get_reverse_order_map(int i){
  // reverse_order_map = [0, 4, 1, 5, 2, 6, 3, 7]
  const int reverse_map[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  return reverse_map[i];
}


/*
Computes GEMV (group_size = 128) with COALESCED memory access.
Modified to match GEMM layout where weight/zeros/scales are transposed.

Key Optimization:
  - Adjacent threads access consecutive memory locations for better coalescing
  - Thread i processes IC[i, i+32, i+64, i+96, ...] instead of IC[i*32:(i+1)*32]

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [IC, OC/8];  (transposed layout)
  zeros: matrix of shape [IC/G, OC/8];  (transposed layout)
  scaling_factors: matrix of shape [IC/G, OC];  (transposed layout)
  output: vector of shape [batch_size, OC];

Implementation:
  - Each warp processes 4 OC in parallel
  - IC dimension is processed in tiles of 1024 (requires multiple iterations if IC > 1024)
  - Within each tile, threads use strided access pattern for coalesced loads
  - Each thread computes one output element by reducing over all IC

Notes:
  This layout is compatible with awq_gemm layout used in linear_awq.py:
  - qweight: [in_features, out_features // 8]
  - qzeros: [in_features // group_size, out_features // 8]
  - scales: [in_features // group_size, out_features]
*/ 
__global__ void gemv_kernel_g128_coalesced(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    
    // Tile size calculation: Each warp iteration processes a tile of IC
    // With coalesced access: threads access strided positions
    const int IC_per_warp_iter = WARP_SIZE * 4 * PACK_FACTOR;  // 1024 IC per iter
    const int groups_per_iter = IC_per_warp_iter / group_size;  // 8 groups
    const int num_iterations = make_divisible(IC / group_size, groups_per_iter);
    
    // New layout dimensions
    const int weight_stride = OC / PACK_FACTOR;  // weight is [IC, OC/8]
    const int zeros_stride = OC / PACK_FACTOR;   // zeros is [IC/G, OC/8]
    const int sf_stride = OC;                     // scales is [IC/G, OC]
    
    // Packed OC index for accessing weight and zeros
    const int packed_oc_idx = oc_idx / PACK_FACTOR;
    const int oc_lane = oc_idx % PACK_FACTOR;
    
    // Main loop: process IC in tiles of 1024
    for(int iter_idx = 0; iter_idx < num_iterations; iter_idx++){
      int iter_base = iter_idx * WARP_SIZE * 4;  // Base offset for this iteration
      
      // Process 4 rounds of coalesced loads (each round loads 32 float4 across the warp)
      #pragma unroll
      for (int round = 0; round < 4; round++){
        // Coalesced load: adjacent threads load consecutive float4
        // Thread 0-31 load float4[base + round*32 + 0:31]
        int load_offset = iter_base + round * WARP_SIZE + threadIdx.x;
        
        half packed_inputs[PACK_FACTOR];
        bool valid_load = (load_offset < IC / PACK_FACTOR);
        
        if (valid_load) {
          *((float4*)packed_inputs) = *(inputs + load_offset);
        }
        
        // Now process each of the 8 IC positions in this float4
        #pragma unroll
        for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
          // Calculate the actual IC index
          int ic_actual = load_offset * PACK_FACTOR + ic_1;
          
          if (ic_actual < IC) {
            // Determine which group this IC belongs to
            int group_idx = ic_actual / group_size;
            
            // Load scale and zero for this group (these are per-group, not per-IC)
            // Use warp shuffle to broadcast from the thread that has the right value
            // or load independently (scales/zeros are likely cached)
            float scaling_factor = __half2float(scaling_factors[group_idx * sf_stride + oc_idx]);
            
            uint32_t packed_zeros = *(zeros + group_idx * zeros_stride + packed_oc_idx);
            // Use reverse order map because weights are packed with order [0,2,4,6,1,3,5,7]
            int zero_bit_offset = get_reverse_order_map(oc_lane) * 4;
            float current_zeros = (float)((packed_zeros >> zero_bit_offset) & 0xF);
            
            // Load weight for weight[ic_actual, oc_packed] and extract the oc_lane-th 4-bit
            uint32_t packed_weight = *(weight + ic_actual * weight_stride + packed_oc_idx);
            // Use reverse order map because weights are packed with order [0,2,4,6,1,3,5,7]
            int weight_bit_offset = get_reverse_order_map(oc_lane) * 4;
            float current_single_weight_fp = (float)((packed_weight >> weight_bit_offset) & 0xF);
            
            // Dequantize weight and multiply with input
            float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
          }
        }
      }
    }
    
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Optimized version with cached scales and zeros to reduce redundant loads.
*/
__global__ void gemv_kernel_g128_coalesced_v2(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    
    const int IC_per_warp_iter = WARP_SIZE * 4 * PACK_FACTOR;  // 1024 IC per iter
    const int groups_per_iter = IC_per_warp_iter / group_size;  // 8 groups
    const int num_iterations = make_divisible(IC / group_size, groups_per_iter);
    
    const int weight_stride = OC / PACK_FACTOR;
    const int zeros_stride = OC / PACK_FACTOR;
    const int sf_stride = OC;
    
    const int packed_oc_idx = oc_idx / PACK_FACTOR;
    const int oc_lane = oc_idx % PACK_FACTOR;
    
    // Cache scales and zeros for this iteration's groups
    float cached_scales[8];
    float cached_zeros[8];
    
    for(int iter_idx = 0; iter_idx < num_iterations; iter_idx++){
      int iter_base = iter_idx * WARP_SIZE * 4;
      
      // Preload scales and zeros for all 8 groups in this iteration
      #pragma unroll
      for (int g = 0; g < groups_per_iter; g++) {
        int group_idx = iter_idx * groups_per_iter + g;
        if (group_idx < IC / group_size) {
          cached_scales[g] = __half2float(scaling_factors[group_idx * sf_stride + oc_idx]);
          uint32_t packed_zeros_val = *(zeros + group_idx * zeros_stride + packed_oc_idx);
          // Use reverse order map because weights are packed with order [0,2,4,6,1,3,5,7]
          int zero_bit_offset = get_reverse_order_map(oc_lane) * 4;
          cached_zeros[g] = (float)((packed_zeros_val >> zero_bit_offset) & 0xF);
        }
      }
      
      // Process 4 rounds of coalesced loads
      #pragma unroll
      for (int round = 0; round < 4; round++){
        int load_offset = iter_base + round * WARP_SIZE + threadIdx.x;
        
        half packed_inputs[PACK_FACTOR];
        bool valid_load = (load_offset < IC / PACK_FACTOR);
        
        if (valid_load) {
          *((float4*)packed_inputs) = *(inputs + load_offset);
        }
        
        // Determine which group this round belongs to
        // Each round processes 32 threads * 8 IC = 256 IC
        // So round 0,1 -> group 0+offset, round 2,3 -> group 1+offset (for group_size=128)
        int local_group = round / (group_size / (WARP_SIZE * PACK_FACTOR));  // 128/(32*8)=0, so need different calc
        // Actually: round*32*8 = round*256, so round 0 -> IC[0:255] (groups 0,1)
        
        #pragma unroll
        for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
          int ic_actual = load_offset * PACK_FACTOR + ic_1;
          
          if (ic_actual < IC) {
            // Calculate which cached group to use
            int ic_offset_in_iter = ic_actual - iter_idx * IC_per_warp_iter;
            int local_group_idx = ic_offset_in_iter / group_size;
            
            if (local_group_idx < groups_per_iter) {
              float scaling_factor = cached_scales[local_group_idx];
              float current_zeros = cached_zeros[local_group_idx];
              
              uint32_t packed_weight = *(weight + ic_actual * weight_stride + packed_oc_idx);
              // Use reverse order map because weights are packed with order [0,2,4,6,1,3,5,7]
              int weight_bit_offset = get_reverse_order_map(oc_lane) * 4;
              float current_single_weight_fp = (float)((packed_weight >> weight_bit_offset) & 0xF);
              
              float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
              psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            }
          }
        }
      }
    }
    
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (PyTorch interface) - Coalesced version.

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [IC, OC // 8];  (transposed layout, compatible with GEMM)
  _scaling_factors: tensor of shape [IC // G, OC];  (transposed layout)
  _zeros: int tensor of shape [IC // G, OC // 8];  (transposed layout)
  group_size: quantization group size (64 or 128)
  version: 1 for basic coalesced, 2 for cached scales/zeros

Returns:
  out_feats: tensor of shape [B, OC];
  
Note:
  This layout matches linear_awq.py:
  - qweight: [in_features, out_features // 8]
  - scales: [in_features // group_size, out_features]
  - qzeros: [in_features // group_size, out_features // 8]
*/
torch::Tensor awq_gemv_coalesced(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size,
    int version = 2)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    
    auto in_feats = reinterpret_cast<float4*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto zeros = reinterpret_cast<uint32_t*>(_zeros.data_ptr<int>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [IC, OC/8], so OC = _kernel.size(1) * 8
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(1) * 8}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    
    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
    dim3 num_threads(32, 4);
    
    if (group_size == 128) {
        if (version == 1) {
            gemv_kernel_g128_coalesced<<<num_blocks, num_threads>>>(
                in_feats, kernel, zeros, scaling_factors, out_feats,
                num_in_channels, num_out_channels
            );
        } else {
            gemv_kernel_g128_coalesced_v2<<<num_blocks, num_threads>>>(
                in_feats, kernel, zeros, scaling_factors, out_feats,
                num_in_channels, num_out_channels
            );
        }
    }
    return _out_feats;
}

