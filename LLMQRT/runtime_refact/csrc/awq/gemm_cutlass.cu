#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "dequantize.cuh"

#include <cuda_fp16.h>

#include <cutlass/arch/memory_sm75.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/array.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

// This kernel intentionally keeps gemm.cu's AWQ packing, dequantization,
// M16N128K32 CTA shape, two-warps-per-CTA mapping, split-K, and epilogue.
//
// Only the hand-written Tensor Core PTX is replaced:
//
//   gemm.cu                                  this file
//   --------------------------------------   ---------------------------------
//   cvta.to.shared + ldmatrix.x4             cutlass::arch::ldsm<RowMajor, 4>
//   cvta.to.shared + ldmatrix.x4.trans       cutlass::arch::ldsm<ColumnMajor,4>
//   mma.sync.m16n8k16.row.col                cutlass::arch::Mma<M16N8K16>
//
// CUTLASS is a C++ wrapper at this level. It still emits the same ldmatrix and
// mma.sync instructions, but gives their register operands explicit fragment
// types instead of manually indexing half[]/float[] storage.

namespace {

using CutlassMma = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 16>,
    32,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAdd>;

using FragmentA = CutlassMma::FragmentA;  // per lane: 8 half  = 4 b32
using FragmentB = CutlassMma::FragmentB;  // per lane: 4 half  = 2 b32
using FragmentC = CutlassMma::FragmentC;  // per lane: 4 float = 4 f32

static_assert(sizeof(cutlass::half_t) == sizeof(half),
              "CUTLASS half and CUDA half must have the same representation");
static_assert(sizeof(FragmentA) == 4 * sizeof(unsigned),
              "m16n8k16 A fragment must contain four b32 registers");
static_assert(sizeof(FragmentB) == 2 * sizeof(unsigned),
              "m16n8k16 B fragment must contain two b32 registers");
static_assert(sizeof(FragmentC) == 4 * sizeof(float),
              "m16n8k16 C fragment must contain four f32 registers");

// ldmatrix writes packed b32 registers, whereas CUTLASS Mma names the same bits
// as Array<half_t, N>. These helpers make that bit-for-bit bridge visible.
CUTLASS_DEVICE
void copy_a_registers(FragmentA &dst, cutlass::Array<unsigned, 4> const &src) {
  unsigned *dst_u32 = reinterpret_cast<unsigned *>(&dst);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    dst_u32[i] = src[i];
  }
}

CUTLASS_DEVICE
void split_b_registers(
    FragmentB &left_n8,
    FragmentB &right_n8,
    cutlass::Array<unsigned, 4> const &src) {
  unsigned *left_u32 = reinterpret_cast<unsigned *>(&left_n8);
  unsigned *right_u32 = reinterpret_cast<unsigned *>(&right_n8);
  left_u32[0] = src[0];
  left_u32[1] = src[1];
  right_u32[0] = src[2];
  right_u32[1] = src[3];
}

__global__ void __launch_bounds__(64)
gemm_forward_4bit_cuda_m16n128k32_cutlass(
    int G,
    int split_k_iters,
    half *__restrict__ A,
    int *__restrict__ B,
    half *__restrict__ scaling_factors,
    int *__restrict__ zeros,
    int M,
    int IC,
    int OC,
    half *__restrict__ C) {
  static constexpr uint32_t ZERO = 0x0;
  static constexpr int kTileM = 16;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 32;
  static constexpr int kMmaN = 8;
  static constexpr int kMmaK = 16;
  static constexpr int kAPaddedStride = 40;
  static constexpr int kBPaddedStride = 136;

  // Same shared-memory tiles as gemm.cu.
  __shared__ half A_shared[kTileM * kAPaddedStride];
  __shared__ half B_shared[kTileK * kBPaddedStride];

  int j_factors1 = (OC + kTileN - 1) / kTileN;
  int blockIdx_y = blockIdx.x % ((M + kTileM - 1) / kTileM * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + kTileM - 1) / kTileM * j_factors1);

  // One warp computes C[16,64]. Eight m16n8k16 instructions therefore own
  // eight independent C fragments. Every fragment contains four floats/lane.
  FragmentC accumulators[8];
#pragma unroll
  for (int mma_n = 0; mma_n < 8; ++mma_n) {
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      accumulators[mma_n][reg] = 0.0f;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / kTileK;  // 8 A rows/warp
  static constexpr int row_stride = 2 * 32 * 8 / kTileN;   // 4 B rows/CTA

  bool ld_A_flag =
      (blockIdx_y / j_factors1 * kTileM +
       threadIdx.y * row_stride_warp + threadIdx.x * 8 / kTileK) < M;

  // The global/shared pointer mapping below is identical to gemm.cu.
  half *A_ptr =
      A + (blockIdx_y / j_factors1 * kTileM +
           threadIdx.y * row_stride_warp + threadIdx.x / (kTileK / 8)) * IC +
      (threadIdx.x % (kTileK / 8)) * 8;

  int *B_ptr =
      B + (blockIdx_y % j_factors1) * (kTileN / 8) +
      threadIdx.y * (OC / 8) * 2 +
      (threadIdx.x / (kTileN / 8)) * (OC / 8) +
      threadIdx.x % (kTileN / 8);

  half *A_shared_ptr =
      A_shared + threadIdx.y * row_stride_warp * kAPaddedStride +
      (threadIdx.x / (kTileK / 8)) * kAPaddedStride +
      (threadIdx.x % (kTileK / 8)) * 8;

  half *B_shared_ptr =
      B_shared + threadIdx.y * (row_stride / 2) * kBPaddedStride +
      (threadIdx.x / (kTileN / 8)) * kBPaddedStride +
      (threadIdx.x % (kTileN / 8)) * 8;

  int *zeros_ptr =
      zeros + (blockIdx_y % j_factors1) * (kTileN / 8) +
      threadIdx.x % (kTileN / 8);

  half *scaling_factors_ptr =
      scaling_factors + (blockIdx_y % j_factors1) * kTileN +
      (threadIdx.x % (kTileN / 8)) * 8;

  half *C_ptr =
      C + blockIdx_z * M * OC +
      (blockIdx_y % j_factors1) * kTileN +
      threadIdx.y * 64 + (threadIdx.x % 4) * 2;

  int k_bound = (IC / kTileK + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * kTileK + blockIdx_z * kTileK >= IC) {
    --k_bound;
  }

  for (int k_tile_in_slice = 0; k_tile_in_slice < k_bound;
       ++k_tile_in_slice) {
    int k_tile = k_tile_in_slice * split_k_iters + blockIdx_z;
    __syncthreads();

    // A global -> shared: unchanged, 8 consecutive half values per thread.
    if (ld_A_flag) {
      *reinterpret_cast<uint4 *>(A_shared_ptr) =
          *reinterpret_cast<uint4 *>(A_ptr + k_tile * kTileK);
    } else {
      *reinterpret_cast<uint4 *>(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    // AWQ zero/scale loading and int4 dequantization: unchanged.
    uint32_t zeros_loaded = *reinterpret_cast<uint32_t *>(
        zeros_ptr + k_tile * kTileK / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale = *reinterpret_cast<uint4 *>(
        scaling_factors_ptr + k_tile * kTileK / G * OC);
    int *B_ptr_local = B_ptr + k_tile * kTileK * (OC / 8);

#pragma unroll
    for (int load_round = 0; load_round < 8; ++load_round) {
      uint32_t B_loaded = *reinterpret_cast<uint32_t *>(
          B_ptr_local + load_round * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

      *reinterpret_cast<uint4 *>(
          B_shared_ptr + load_round * row_stride * kBPaddedStride) =
          B_loaded_fp16;
    }
    __syncthreads();

    // A K32 tile is still consumed as two K16 Tensor Core steps.
#pragma unroll
    for (int k_mma = 0; k_mma < kTileK / kMmaK; ++k_mma) {
      FragmentA fragment_A;
      FragmentB fragments_B[8];

      // CUTLASS replacement 1:
      //
      // Original:
      //   cvta.to.shared(addr, A_ldsm_ptr)
      //   ldmatrix.sync.aligned.m8n8.x4.shared.b16 {a0..a3}, [addr]
      //
      // CUTLASS receives the ordinary shared pointer, converts its address
      // space internally, and returns the same four b32 registers.
      half const *A_ldsm_ptr =
          A_shared + k_mma * kMmaK +
          (threadIdx.x & 15) * kAPaddedStride +
          (threadIdx.x >> 4) * 8;
      cutlass::Array<unsigned, 4> A_ldsm_registers;
      cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
          A_ldsm_registers, A_ldsm_ptr);
      copy_a_registers(fragment_A, A_ldsm_registers);

      // CUTLASS replacement 2:
      // Each x4.trans load covers B[K16,N16]. Its first two b32 registers
      // form the left N8 FragmentB; its last two form the right N8 FragmentB.
#pragma unroll
      for (int n16 = 0; n16 < 4; ++n16) {
        half const *B_ldsm_ptr =
            B_shared + k_mma * kMmaK * kBPaddedStride +
            threadIdx.y * 64 + n16 * 16 +
            (threadIdx.x & 15) * kBPaddedStride +
            (threadIdx.x >> 4) * 8;

        cutlass::Array<unsigned, 4> B_ldsm_registers;
        cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(
            B_ldsm_registers, B_ldsm_ptr);
        split_b_registers(
            fragments_B[2 * n16],
            fragments_B[2 * n16 + 1],
            B_ldsm_registers);
      }

      // CUTLASS replacement 3:
      // 8 calls per warp cover N=64:
      //
      //   mma_n:       0       1       2       3    ...       7
      //   output N:  [0..7] [8..15] [16..23] [24..31] ... [56..63]
      //   logical op: C[16,8] += A[16,16] * B[16,8]
      //
      // All 32 lanes execute each call together. FragmentA/FragmentB are not
      // private mini-matrices; the 32 lanes' fragments collectively encode
      // the full A[16,16] and B[16,8] operands.
      CutlassMma mma;
#pragma unroll
      for (int mma_n = 0; mma_n < 8; ++mma_n) {
        mma(
            accumulators[mma_n],
            fragment_A,
            fragments_B[mma_n],
            accumulators[mma_n]);
      }
    }
  }

  // Same epilogue mapping as gemm.cu. The original flat C_warp index
  // [n16 * 8 + local_id] maps to:
  //   mma fragment = 2*n16 + local_id/4
  //   register     = local_id%4
#pragma unroll
  for (int n16 = 0; n16 < 4; ++n16) {
#pragma unroll
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset =
          blockIdx_y / j_factors1 * kTileM + threadIdx.x / 4 +
          (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        int mma_n = 2 * n16 + local_id / 4;
        int fragment_register = local_id % 4;
        *(C_ptr + n16 * 16 + row_offset * OC +
          (local_id / 4) * kMmaN + local_id % 2) =
            __float2half(accumulators[mma_n][fragment_register]);
      }
    }
  }
}

}  // namespace

torch::Tensor awq_gemm_cutlass(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters) {
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

  auto options =
      torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty(
      {split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
  int num_out_feats = _out_feats.size(-2);
  int num_out_channels = _out_feats.size(-1);

  auto in_feats = reinterpret_cast<half *>(_in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<int *>(_kernel.data_ptr<int>());
  auto out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());
  auto scaling_factors =
      reinterpret_cast<half *>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int *>(_zeros.data_ptr<int>());
  int group_size = num_in_channels / _scaling_factors.size(0);

  if (num_out_channels % 64 != 0) {
    throw std::invalid_argument("OC is not multiple of cta_N = 64");
  }
  if (num_out_channels % 8 != 0) {
    throw std::invalid_argument("OC is not multiple of pack_num = 8");
  }
  if (group_size % 32 != 0) {
    throw std::invalid_argument("Group size should be a multiple of 32");
  }
  if (num_out_channels % group_size != 0) {
    throw std::invalid_argument("OC is not multiple of Group size");
  }
  if (num_out_channels % 128 != 0) {
    throw std::invalid_argument("OC is not multiple of 128");
  }

  int j_factors1 = num_out_channels / 128;
  dim3 num_blocks(
      (num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);
  dim3 threads_per_block(32, 2);

  gemm_forward_4bit_cuda_m16n128k32_cutlass<<<
      num_blocks, threads_per_block>>>(
      group_size,
      split_k_iters,
      in_feats,
      kernel,
      scaling_factors,
      zeros,
      num_in_feats,
      num_in_channels,
      num_out_channels,
      out_feats);

  return _out_feats.sum(0);
}
