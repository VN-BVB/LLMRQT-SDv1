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
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
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
Computes GEMV (group_size = 128).
Modified to match GEMM layout where weight/zeros/scales are transposed.

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [IC, OC/8];  (transposed layout)
  zeros: matrix of shape [IC/G, OC/8];  (transposed layout)
  scaling_factors: matrix of shape [IC/G, OC];  (transposed layout)
  output: vector of shape [batch_size, OC];

Implementation:
  - Each warp processes 4 OC in parallel
  - IC dimension is processed in tiles of 1024 (requires multiple iterations if IC > 1024)
  - Each thread computes one output element by reducing over all IC

Notes:
  This layout is compatible with awq_gemm layout used in linear_awq.py:
  - qweight: [in_features, out_features // 8]
  - qzeros: [in_features // group_size, out_features // 8]
  - scales: [in_features // group_size, out_features]
*/ 
// weight一共out/8列，int32，列上分配了out/4个block，也就是说一个block需要处理16位，分4个warp，1个warp在weight尺度上处理4位，刚刚好，在weight的行尺度上可以用4个float4_perthread等等
__global__ void gemv_kernel_g128(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    // dim3 num_blocks(1, num_out_channels / 4, num_out_feats); //每个block负责4个OC，一个warp负责一个OC
    // dim3 num_threads(32, 4);
    const int batch_idx = blockIdx.z; // B或者M
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; //表示未pack情况下的oc idx， 这个用来表述weight上具体的列
    //地址
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR; // 8HALF=FLAOT4,故除以pack_factor
    half* outputs = _outputs + batch_idx * OC;
    
    // Tile size calculation: Each warp iteration processes a tile of IC
    // Each thread: 4 float4 * 8 half/float4 = 32 IC
    // Each warp: 32 threads * 32 IC/thread = 1024 IC per iteration
    const int IC_per_warp_iter = WARP_SIZE * 4 * PACK_FACTOR;  // fp16情况下每个block一次处理的IC数量 32 * 4 * 8 = 1024
    const int groups_per_iter = IC_per_warp_iter / group_size;  // 1024/128=8 # 每个block一次处理的group数量
    const int num_iterations = make_divisible(IC / group_size, groups_per_iter);  // total iterations needed
    // 总共需要的groupsize数量和一次处理的groupsize数量，决定了需要多少次迭代才能把所有IC处理完

    // New layout dimensions (transposed compared to original gemv)
    const int weight_stride = OC / PACK_FACTOR;  // weight is [IC, OC/8]
    const int zeros_stride = OC / PACK_FACTOR;   // zeros is [IC/G, OC/8]
    const int sf_stride = OC;                     // scales is [IC/G, OC]
    
    // Packed OC index for accessing weight and zeros
    const int packed_oc_idx = oc_idx / PACK_FACTOR;
    const int oc_lane = oc_idx % PACK_FACTOR;
    
    // Main loop: process IC in tiles of 1024 (may need multiple iterations if IC > 1024)
    for(int iter_idx = 0; iter_idx < num_iterations; iter_idx++){
      // 关键：拿到group id即可取出scale和zeros
      // zeros is now [IC/G, OC/8], we need zeros[iter_idx * groups_per_iter + warp_group, oc_idx/8]
      // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
      int group_idx = iter_idx * groups_per_iter + (threadIdx.x / 4); // 4个thread处理一个group
      //！！scale factor和zeros只和group id有关
      //zeros shape=[IC/G, OC/8]
      uint32_t packed_zeros = *(zeros + group_idx * zeros_stride + packed_oc_idx);// 找到了zeros哪一列（packed_oc_idx）上位于第group_idx个（IC/G索引）上了

      // Extract the 4-bit zero value for this OC lane
      // Use reverse order map because weights are packed with order [0,2,4,6,1,3,5,7]
      int zero_bit_offset = get_reverse_order_map(oc_lane) * 4;
      float current_zeros = (float)((packed_zeros >> zero_bit_offset) & 0xF);
      // Load scaling factor for this group and output channel
      // scales is now [IC/G, OC], we need scales[group_idx, oc_idx]
      float scaling_factor = __half2float(scaling_factors[group_idx * sf_stride + oc_idx]);
      // 每个thread处理4float4，但这4个float4是挨着的，而不是跨stride的
      int inputs_ptr_delta = iter_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // thread local reduce
      // Process weights: for each of the 4(4f4)*8(8half)=32 IC positions this thread handles
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){ //4float4
        // Load 8 inputs for this batch，8half=float4
        half packed_inputs[PACK_FACTOR];
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          
          // Now process each of the 8 IC positions in this float4
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){//每个f4中的8half
            // Calculate the actual IC index (unpacked)
            int ic_actual = (inputs_ptr_delta + ic_0) * PACK_FACTOR + ic_1;
            // 每个具体的IC，拿出对应的weight出来unpack
            if (ic_actual < IC) {
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
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [IC, OC // 8];  (transposed layout, compatible with GEMM)
  _scaling_factors: tensor of shape [IC // G, OC];  (transposed layout)
  _zeros: int tensor of shape [IC // G, OC // 8];  (transposed layout)
  group_size: quantization group size (64 or 128)

Returns:
  out_feats: tensor of shape [B, OC];
  
Note:
  This layout matches linear_awq.py:
  - qweight: [in_features, out_features // 8]
  - scales: [in_features // group_size, out_features]
  - qzeros: [in_features // group_size, out_features // 8]
*/
torch::Tensor awq_gemv(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size)
{
    int num_in_feats = _in_feats.size(0); //B
    int num_in_channels = _in_feats.size(1); //IC
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<float4*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto zeros = reinterpret_cast<uint32_t*>(_zeros.data_ptr<int>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [IC, OC//8]
    // _out_feats is [B, OC]
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(1) * 8}, options);
    int num_out_feats = _out_feats.size(-2);//B
    int num_out_channels = _out_feats.size(-1); //OC
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    // 把所有输入和输出的指针拿到了
    int blockDim_z = num_out_feats;
    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);//（1,oc ///4 , b）
    dim3 num_threads(32, 4);
    if (group_size == 128)
    {
      gemv_kernel_g128<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels
      );
    }
    return _out_feats;
;}

