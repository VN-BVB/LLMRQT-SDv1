#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "dequantize.cuh"

#include <cuda_fp16.h>

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// ============================================================================
// Tile configuration: M32N128K32 with Double Buffering
// ============================================================================
// - Each block computes a 32×128 output tile (doubled M dimension)
// - K dimension processed 32 elements at a time
// - 128 threads = 4 warps (32×4)
// - Double buffering for both shared memory (A, B) and registers (optional)
//
// Warp assignment:
//   Warp0: C[0:16, 0:64]   - loads A[0:16, k]
//   Warp1: C[0:16, 64:128] - loads A[0:16, k] (shared with Warp0)
//   Warp2: C[16:32, 0:64]  - loads A[16:32, k]
//   Warp3: C[16:32, 64:128]- loads A[16:32, k] (shared with Warp2)
//
// Benefits:
//   - 50% reduction in redundant A loads (was 100% in M16 version)
//   - Fewer blocks (better SM utilization)
//   - Higher arithmetic intensity
// ============================================================================

__global__ void __launch_bounds__(128) gemm_forward_4bit_cuda_m32n128k32_db(
    int G, int split_k_iters, 
    half* __restrict__ A, 
    int* __restrict__ B, 
    half* __restrict__ scaling_factors, 
    int* __restrict__ zeros, 
    int M, int IC, int OC, 
    half* __restrict__ C) 
{
  static constexpr uint32_t ZERO = 0x0;
  
  // ========== Register Storage ==========
  // Each thread computes 4×8 = 32 output elements (fp32 accumulation)
  float C_warp[32];  // 4 (n chunks) × 8 (elements per m16n8k16)
  
  // Warp-level register storage for mma inputs
  half A_shared_warp[8];   // One m16n16 tile fragment for A
  half B_shared_warp[32];  // Four m16n16 tile fragments for B (4×8)
  
  // ========== Shared Memory: Double Buffered ==========
  // Buffer 0 and Buffer 1 for ping-pong
  
  // A tiles: 2 × [32, 32+8] = 2 × 1280 halves = 2.5 KB per buffer = 5 KB total
  // - 32: tile height (M dimension, doubled from 16)
  // - 32: tile width (K dimension per iteration)
  // - +8: padding to avoid bank conflicts
  __shared__ half A_shared[2][32 * (32 + 8)];  // Double buffered
  
  // B tiles: 2 × [32, 128+8] = 2 × 4352 halves = 8.5 KB per buffer = 17 KB total
  // - 32: tile height (K dimension)
  // - 128: tile width (N dimension)
  // - +8: padding to avoid bank conflicts
  __shared__ half B_shared[2][32 * (128 + 8)];  // Double buffered
  
  // Total shared memory: 5 KB + 17 KB = 22 KB (well within 48 KB limit)
  
  // ========== Grid/Block Index Decoding ==========
  // Number of 128-wide tiles in N dimension
  int j_factors1 = ((OC + 128 - 1) / 128);
  
  // Decode 1D block index into 2D tile coordinates + split-K index
  int blockIdx_y = blockIdx.x % ((M + 32 - 1) / 32 * j_factors1);  // 32 = M tile height
  int blockIdx_z = blockIdx.x / ((M + 32 - 1) / 32 * j_factors1);
  
  // Which M tile does this block handle? (0, 1, 2, ...)
  int m_tile_idx = blockIdx_y / j_factors1;
  // Which N tile does this block handle? (0, 1, 2, ...)
  int n_tile_idx = blockIdx_y % j_factors1;
  
  // Which warp pair does this warp belong to? (0 or 1)
  // Warp 0-1: pair 0 (handles M[0:16])
  // Warp 2-3: pair 1 (handles M[16:32])
  int warp_id = threadIdx.y;
  int warp_pair = warp_id / 2;  // 0 or 1
  
  // ========== Initialize Accumulator ==========
  for (int j_0_4_init = 0; j_0_4_init < 4; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }
  
  // ========== Stride Calculations ==========
  // A loading: each warp loads 8 rows (32 threads × 8 halves / 32 K = 8 rows)
  static constexpr int row_stride_warp = 32 * 8 / 32;  // = 8
  
  // B loading: 4 warps × 32 threads × 8 halves / 128 N = 8 rows
  static constexpr int row_stride = 4 * 32 * 8 / 128;  // = 8
  
  // ========== Global Memory Pointers ==========
  // A pointer: [M, IC]
  // Each warp pair loads 16 rows (warp_pair * 16)
  bool ld_A_flag = (m_tile_idx * 32 + warp_pair * 16 + threadIdx.x * 8 / 32) < M;
  
  half* A_ptr = A 
                + m_tile_idx * 32 * IC                    // M tile base
                + warp_pair * 16 * IC                     // Warp pair offset (0 or 16 rows)
                + (threadIdx.x / (32 / 8)) * IC           // Thread row offset
                + (threadIdx.x % (32 / 8)) * 8;           // Thread column offset
  
  // B pointer: [IC, OC/8]
  int* B_ptr = B
            + n_tile_idx * (128 / 8)                   // N tile offset
            + threadIdx.y * (OC / 8) * 2               // Warp row offset
            + (threadIdx.x / (128 / 8)) * (OC / 8)     // Thread row offset
            + (threadIdx.x % (128 / 8)) * 1;           // Thread column offset
  
  // Zeros pointer: [IC/G, OC/8]
  int* zeros_ptr = zeros
                + n_tile_idx * (128 / 8)
                + (threadIdx.x % (128 / 8));
  
  // Scales pointer: [IC/G, OC]
  half* scaling_factors_ptr = scaling_factors
                            + n_tile_idx * 128
                            + (threadIdx.x % (128 / 8)) * 8;
  
  // Output pointer: [split_k_iters, M, OC]
  half* C_ptr = C 
              + blockIdx_z * M * OC
              + n_tile_idx * 128
              + warp_id * 32                         // Warp offset in N (each warp: 32 cols for output)
              + (threadIdx.x % 4) * 2;
  
  // ========== K-loop Bound ==========
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  
  // ========== Shared Memory Pointers (Double Buffered) ==========
  // Will be updated each iteration to point to current buffer
  half* A_shared_ptr[2];
  half* B_shared_ptr[2];
  
  for (int buf = 0; buf < 2; ++buf) {
    A_shared_ptr[buf] = A_shared[buf] 
                      + warp_pair * 16 * (32 + 8)          // Warp pair base (0 or 16 rows)
                      + (threadIdx.x / (32 / 8)) * (32 + 8)
                      + (threadIdx.x % (32 / 8)) * 8;
    
    B_shared_ptr[buf] = B_shared[buf]
                      + (threadIdx.y % 2) * 4 * (128 + 8)  // Warp offset within pair
                      + (threadIdx.x / (128 / 8)) * (128 + 8)
                      + (threadIdx.x % (128 / 8)) * 8;
  }
  
  // ========== PROLOGUE: Load first tile into buffer 0 ==========
  if (k_bound > 0) {
    int k_0_0 = blockIdx_z;  // First K tile for this split-K slice
    
    // Load A tile
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr[0]) = *(uint4*)(A_ptr + k_0_0 * 32);
    } else {
      *(uint4*)(A_shared_ptr[0]) = make_uint4(0, 0, 0, 0);
    }
    
    // Load scales and zeros
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * OC);
    
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);
    
    // Load and dequantize B tile
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {  // 4 rounds for 32 rows (128 threads)
      uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      
      // Dequantize: (w - zero) * scale
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      
      *(uint4*)(B_shared_ptr[0] + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
    }
    
    __syncthreads();  // Wait for prologue load to complete
  }
  
  // ========== MAINLOOP: Double-Buffered Pipeline ==========
  for (int _k_0_0 = 0; _k_0_0 < k_bound - 1; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    int k_0_0_next = k_0_0 + split_k_iters;
    
    int compute_buf = _k_0_0 % 2;      // Which buffer to compute from
    int load_buf = (_k_0_0 + 1) % 2;   // Which buffer to load into
    
    // ===== PHASE 1: COMPUTE current tile (from compute_buf) =====
    // Process K=32 in 2 chunks of K=16
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      // Load A fragment from shared memory
      {
        unsigned int addr;
        __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(addr)
          : "l"((void *)((&(A_shared[compute_buf][k_0_1 * 16])) + 
                         (((threadIdx.x & 15) * (32 + 8)) + ((threadIdx.x >> 4) * 8))))
        );
        
        __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
          "{%0, %1, %2, %3}, [%4];\n"
          : "=r"(((unsigned *)(A_shared_warp + 0))[0]), 
            "=r"(((unsigned *)(A_shared_warp + 0))[1]), 
            "=r"(((unsigned *)(A_shared_warp + 0))[2]), 
            "=r"(((unsigned *)(A_shared_warp + 0))[3])
          : "r"(addr)
        );
      }
      
      // Load B fragments from shared memory (4 chunks)
      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[compute_buf][k_0_1 * 16 * (128 + 8) + (threadIdx.y % 2) * 64 + ax1_0 * 16])) + 
                           (((threadIdx.x & 15) * (128 + 8)) + ((threadIdx.x >> 4) * 8))))
          );
          
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), 
              "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), 
              "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), 
              "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
            : "r"(addr)
          );
        }
      }
      
      // Execute mma instructions
      for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
        // First m16n8k16
        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), 
              "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]),
              "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }
        
        // Second m16n8k16
        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), 
              "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
      }
    }  // End k_0_1 loop (compute phase)
    
    __syncthreads();  // Wait for compute to finish before reusing shared memory
    
    // ===== PHASE 2: LOAD next tile (into load_buf) =====
    // Load A tile
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr[load_buf]) = *(uint4*)(A_ptr + k_0_0_next * 32);
    } else {
      *(uint4*)(A_shared_ptr[load_buf]) = make_uint4(0, 0, 0, 0);
    }
    
    // Load scales and zeros for next tile
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0_next * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0_next * 32 / G * OC);
    
    int* B_ptr_local = B_ptr + k_0_0_next * 32 * (OC / 8);
    
    // Load and dequantize B tile
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
      uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      
      // Dequantize
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      
      *(uint4*)(B_shared_ptr[load_buf] + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
    }
    
    __syncthreads();  // Wait for load to complete
  }  // End mainloop
  
  // ========== EPILOGUE: Compute last tile ==========
  if (k_bound > 0) {
    int final_buf = (k_bound - 1) % 2;
    
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      // Load A fragment
      {
        unsigned int addr;
        __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(addr)
          : "l"((void *)((&(A_shared[final_buf][k_0_1 * 16])) + 
                         (((threadIdx.x & 15) * (32 + 8)) + ((threadIdx.x >> 4) * 8))))
        );
        
        __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
          "{%0, %1, %2, %3}, [%4];\n"
          : "=r"(((unsigned *)(A_shared_warp + 0))[0]), 
            "=r"(((unsigned *)(A_shared_warp + 0))[1]), 
            "=r"(((unsigned *)(A_shared_warp + 0))[2]), 
            "=r"(((unsigned *)(A_shared_warp + 0))[3])
          : "r"(addr)
        );
      }
      
      // Load B fragments
      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[final_buf][k_0_1 * 16 * (128 + 8) + (threadIdx.y % 2) * 64 + ax1_0 * 16])) + 
                           (((threadIdx.x & 15) * (128 + 8)) + ((threadIdx.x >> 4) * 8))))
          );
          
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), 
              "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), 
              "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), 
              "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
            : "r"(addr)
          );
        }
      }
      
      // Execute mma
      for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), 
              "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]),
              "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }
        
        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), 
              "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
      }
    }
  }
  
  // ========== Write Results Back ==========
  // Note: M dimension handling updated for M32 tile
  for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      // Row offset includes warp_pair offset (0 or 16)
      int row_offset = m_tile_idx * 32                // M tile base
                     + warp_pair * 16                 // Warp pair offset (0 or 16)
                     + (threadIdx.x / 4)              // Thread row within warp
                     + (local_id % 4) / 2 * 8;        // mma layout row offset
      
      if (row_offset < M) {
        // Column offset: each warp handles 32 cols (warp_id % 2 distinguishes left/right within pair)
        int col_base = (warp_id % 2) * 32;  // 0 or 32
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + col_base + (local_id / 4) * 8 + local_id % 2) = 
          __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
}


// ============================================================================
// PyTorch Interface
// ============================================================================
torch::Tensor awq_gemm_db_m32(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
    int group_size = num_in_channels / _scaling_factors.size(0);

    // Input validation
    if (num_out_channels % 128 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 128");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    if (group_size % 32 != 0)
	      throw std::invalid_argument("Group size should be a multiple of 32");
    if (num_out_channels % group_size != 0)
        throw std::invalid_argument("OC is not multiple of Group size");

    // Launch kernel
    int j_factors1 = num_out_channels / 128;
    
    // Grid: (M tiles) × (N tiles) × (split_k slices)
    // M tile size changed from 16 to 32
    dim3 num_blocks((num_out_feats + 32 - 1) / 32 * j_factors1 * split_k_iters);
    
    // Block: 128 threads = 4 warps (32×4)
    dim3 threads_per_block(32, 4);
    
    gemm_forward_4bit_cuda_m32n128k32_db<<<num_blocks, threads_per_block>>>(
        group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, 
        num_in_feats, num_in_channels, num_out_channels, out_feats);
    
    return _out_feats.sum(0);
}

