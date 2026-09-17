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

// Tile configuration: M16N128K32
// - Each block computes a 16×128 output tile
// - K dimension processed 32 elements at a time
// - 64 threads = 2 warps (32×2)
// 建模一个长N ，宽k 高M的立方体，
__global__ void __launch_bounds__(64) gemm_forward_4bit_cuda_m16n128k32(int G, int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C) 
{
  static constexpr uint32_t ZERO = 0x0;
  
  // Each thread computes 4×8 = 32 output elements (stored in fp32 for accumulation)
  float C_warp[32];  // 32 = 4 (n chunks) × 8 (C reg数量 per chunk, chunk指每个m16n16 layout)
  
  // Shared memory for A tile: [16, 32+8]
  // - 16: tile height (M dimension)
  // - 32: tile width (K dimension per iteration)
  // - +8: padding to avoid bank conflicts (16-way banks, accessing stride=32 would conflict)
  // t0-t3访问第一行，t4-t7访问第二行，t0-t7位于同一phase
  // half8读取，也就说第一行的起始地址和第二行的起始地址要错开8half，即4个bank才行，否则会出现bank conflict
  __shared__ half A_shared[16 * (32 + 8)];  // 16×40 = 640 halves = 1.25 KB
  
  // Shared memory for B tile: [32, 128+8]
  // - 32: tile height (K dimension)
  // - 128: tile width (N dimension)
  // - +8: padding to avoid bank conflicts
  __shared__ half B_shared[32 * (128 + 8)];  // 32×136 = 4352 halves = 8.5 KB
  
  // Unused shared memory (for future optimization)
  __shared__ half scaling_factors_shared[128];
  __shared__ half zeros_shared[128];
  
  // Number of 128-wide tiles in N dimension
  int j_factors1 = ((OC + 128 - 1) / 128);  // 128 = tile width (N)
  
  int blockIdx_x = 0;  // Unused (could be used for batch dimension)
  
  // Decode 1D block index into 2D tile coordinates + split-K index
  // blockIdx_y: which M×N tile (flattened from 2D grid)
  // blockIdx_z: which split-K slice
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);  //  MN中的第几个  原注释：16 = tile height (M)感觉有问题
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  // Warp level Register storage for mma instruction inputs
  half A_shared_warp[8];   // 8 halves = one m16n16 tile fragment for A
  half B_shared_warp[32];  // 32 halves = four m16n16 tile fragments for B (4×8)
  
  // Initialize accumulator to zero
  // 64/16=4 chunks in N dimension, 8 elements per chunk(m16n16)
  for (int j_0_4_init = 0; j_0_4_init < 4; ++j_0_4_init) {  // 4 = 128 / 32 (N tile / output chunk)
    for (int i = 0; i < 8; ++i) {  // 8 = elements per mma output
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  // Stride calculations for shared memory layout
  // row_stride_warp: rows per warp when loading A from gmem
  // = (32 threads × 8 halves) / 32 (K tile width) = 8 rows per warp 32个线程一个 8个half
  static constexpr int row_stride_warp = 32 * 8 / 32;  // = 8
  
  // row_stride: rows per block when loading B from gmem
  // = (2 warps × 32 threads × 8 halves) / 128 (N tile width) = 4 rows
  static constexpr int row_stride = 2 * 32 * 8 / 128;  // = 4
    
  // Flag: whether this thread should load A (check if row is within M)
  // TODO: blockIdx_y / j_factors1 in A loading to support bsz > 16
  bool ld_A_flag = (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;  // 16 = M tile, 8 halves, 32 = K tile
  // bool wb_C_flag = (threadIdx.x / 4) < M;

  // Global memory pointer for A (input activations): [M, IC]
  // Each thread loads 8 consecutive halves
  half* A_ptr = A 
                + (((int)blockIdx_y) / j_factors1 * 16          // M tile offset: 16 rows per tile
                   + (((int)threadIdx.y) * row_stride_warp)     // row offset within tile (8 rows per warp)
                   + ((int)threadIdx.x) / (32 / 8)) * IC        // row offset within warp (32/8=4 threads per row)
                + (((int)threadIdx.x) % (32 / 8)) * 8;          // column offset (8 halves per thread, 32/8=4 groups)
  
  // Global memory pointer for B (quantized weights): [IC, OC/8]
  // Layout is packed: each int32 contains 8×4-bit weights
  //注意这里不和A配合，这里是a与B分别写入共享内存 
  int* B_ptr = B
            +  (((int)blockIdx_y) % j_factors1) * (128 / 8) // N tile offset: 128/8=16 packed columns per tile
            + ((int)threadIdx.y) * (OC / 8) * 2            // row offset within tile (2 rows per warp (32/16=2))
            + (((int)threadIdx.x) / (128 / 8)) * (OC / 8)  // row offset within warp (128/8=16 threads per row)
            + (((int)threadIdx.x) % (128 / 8)) * 1;        // column offset: each thread loads 1 int32 (8×4-bit)
                        
  // Shared memory pointer for A tile: [16, 32+8]
  half* A_shared_ptr = A_shared 
                    + ((int)threadIdx.y) * row_stride_warp * (32 + 8)  // warp row offset: 8 rows, stride=40
                    + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)        // thread row offset: 4 threads per row, stride=40
                    + (((int)threadIdx.x) % (32 / 8) ) * 8;             // thread column offset: 8 halves, 32/8=4 groups

  // Shared memory pointer for B tile: [32, 128+8]
  half* B_shared_ptr = B_shared
                    + ((int)threadIdx.y) * (row_stride / 2) * (128 + 8)  // warp offset: row_stride/2=2 rows per warp
                    + (((int)threadIdx.x) / (128 / 8)) * (128 + 8)        // thread row offset: 128/8=16 threads cover one row
                    + (((int)threadIdx.x) % (128 / 8)) * 8;               // thread column offset: 8 halves per thread
  
  // Global memory pointer for zeros: [IC/G, OC/8]
  // G = group_size (e.g., 128), each int32 packs 8×4-bit zeros
  int* zeros_ptr = zeros
                + (((int)blockIdx_y) % j_factors1) * (128 / 8)  // N tile offset: 128/8=16 packed columns
                + ((int)threadIdx.x) % (128 / 8);                // thread offset: 1 int32 per thread (128/8)
  
  // Global memory pointer for scales: [IC/G, OC]
  half* scaling_factors_ptr = scaling_factors
                            + (((int)blockIdx_y) % j_factors1) * (128)      // N tile offset: 128 columns
                            + (((int)threadIdx.x) % (128 / 8)) * 8;         // thread offset: 8 halves per thread (128/8=16 groups)

  // Output pointer for C: [split_k_iters, M, OC]
  // Each thread writes 2 halves at a time (for coalescing)
  half* C_ptr = C 
              + blockIdx_z * M * OC                       // split-K slice offset (reduce at host)
              + (((int)blockIdx_y) % j_factors1) * 128    // N tile offset: 128 columns per tile
              + ((int)threadIdx.y) * 64                   // warp offset: 64 columns per warp (128/2)
              + (((int)threadIdx.x) % 4) * 2;             // thread offset: 2 halves per thread (for mma layout)每个线程指向layout所示的每个tid的起始地址

  // Calculate K-loop bound for this split-K slice
  // IC/32: number of K=32 tiles
  // divide by split_k_iters: each slice processes this many tiles
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;  // 32 = K tile size
  // Handle edge case: last slice may have fewer iterations
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;  // 32 = K tile size
  
  // Main K-loop: iterate over K tiles of 32
  //split_k_iters 代表目前有多少个并行处理k维，最后把这split_k_iters个mn tile的结果做reduce
  // blockIdx_z是split_k_iters中的第几个
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;  // Actual K tile index (with split-K offset)
    __syncthreads();  // Ensure previous iteration's shared memory writes are complete
    // ========== Step 1: Load A tile from global to shared memory ==========
    // TODO: blockIdx_y / j_factors1 in A loading to support bsz > 16
    if (ld_A_flag)  // Check if row is within M
    {
      // Load 8 halves (uint4 = 4×uint32 = 8×half) from A[m, k_0_0*32:k_0_0*32+32]
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));  // 32 = K tile size
    }
    else  // Padding for out-of-bounds rows
    {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    // ========== Step 2: Load scales and zeros for dequantization ==========
    // Load zeros for this K tile and N tile
    // zeros: [IC/G, OC/8], each int32 packs 8×4-bit zeros
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));  // 32=K tile, G=group_size
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);  // Unpack to 8 fp16 values
    
    // Load scales for this K tile and N tile
    // scales: [IC/G, OC], direct fp16 values
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));  // uint4 = 8 halves
    /*
    if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 && threadIdx.y == 0){
      printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x, B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x, B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
    }
    */
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);  // Offset to current K tile (32 rows)

    // ========== Step 3: Load and dequantize B tile ==========
    // Load B in 8 rounds to cover all 32 rows of the K tile
    // Each round: 64 threads load (32×128) / 8 = 512 int32 values
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {  // 8 rounds to cover 32 rows
      // Target B_shared layout: [32, 128+8] in fp16
      // Each thread: load 1 int32 (8×4-bit) → dequantize → store 8 fp16
      // row_stride = 4: each round advances 4 rows (2 warps × 32 threads × 8 halves / 128 cols = 4 rows)
      
      // Load packed 4-bit weights from global memory
      uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));  // row_stride=4
      
      // Unpack 8×4-bit integers to 8×fp16 (still quantized values)
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);  // uint4 = 4×uint32 = 8×half
      
      // ========== Dequantization: w_fp16 = (w_int4 - zero) * scale ==========
      // Process 4×2 = 8 halves (stored in .x, .y, .z, .w, each is 2 packed halves)
      // Using PTX inline assembly for optimal performance
      // TODO: can save 4 instructions if formulated as: deq = w * scale - zero * scale
      
      // Process first 2 halves (.x contains 2 packed fp16)
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));  // w - zero
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));  // (w-zero)*scale
      
      // Process next 2 halves (.y)
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      
      // Process next 2 halves (.z)
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      
      // Process last 2 halves (.w)
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      /*
      if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 17 && threadIdx.y == 0){
        printf("[x] %X %X %X %X\n", B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
      }
      */

      // Write dequantized B tile to shared memory
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;  // row_stride=4, 128+8=136 (padded)
    }
    __syncthreads();  // Wait for all A and B tiles to be loaded

    // ========== Step 4: Tensor Core computation ==========
    // Split K=32 into 2 iterations of K=16 (mma instruction processes k=16)
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {  //32拆成两个16  2 = 32 / 16 (K tile / mma K size)
      // ===== Load A fragment from shared memory using ldmatrix =====
      {
        unsigned int addr;
        // Compute shared memory address for A tile
        // k_0_1 * 16: offset for K=16 chunk (0 or 16)
        // 以下两个式子非常重要，为了符合ldsm对A_shared buffer上各个thread指向各个起始地址的要求
        // ((threadIdx.x & 15) * 40): thread's row offset (16 threads per warp for A, stride=40 includes padding)
        // ((threadIdx.x >> 4) * 8): upper 16 threads have different column offset
        //                      col 0～7       col 8～15
        //            +-------------+-------------+
        // row 0～7   |  matrix 0   |  matrix 2   |
        //            +-------------+-------------+
        // row 8～15  |  matrix 1   |  matrix 3   |
        //            +-------------+-------------+
        // lane  0～7  → matrix 0 的8个行地址
        // lane  8～15 → matrix 1 的8个行地址
        // lane 16～23 → matrix 2 的8个行地址
        // lane 24～31 → matrix 3 的8个行地址
        __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(addr)
          : "l"((void *)((&(A_shared[(k_0_1 * 16)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))  // 16=K chunk, 40=padded stride, 8=col offset
        );

        // ldmatrix: hardware-accelerated warp-level load optimized for mma
        // .m8n8.x4 一次加载 4 个 8x8。按上面的 addr 公式，四块在 A_shared 中是：
        //
        //                       col 0~7           col 8~15
        //                    +----------------+----------------+
        //   row 0~7          | matrix 0 -> %0 | matrix 2 -> %2 |
        //                    +----------------+----------------+
        //   row 8~15         | matrix 1 -> %1 | matrix 3 -> %3 |
        //                    +----------------+----------------+
        //
        // 每个 lane 最终得到 4 个 b32 寄存器，每个 b32 装 2 个 half，合计 8 half。
        // 对任意 lane，令：
        //
        //   group = lane >> 2;  // lane/4，取值 0~7
        //   pair  = lane & 3;   // lane%4，取值 0~3
        //   kbase = k_0_1*16;
        //
        // 则该 lane 的四个输出寄存器按 %0、%1、%2、%3 的顺序为：
        //
        //   %0 = { A_shared[group    ][kbase + 2*pair    ],
        //          A_shared[group    ][kbase + 2*pair + 1] }  // matrix 0，左上
        //   %1 = { A_shared[group + 8][kbase + 2*pair    ],
        //          A_shared[group + 8][kbase + 2*pair + 1] }  // matrix 1，左下
        //   %2 = { A_shared[group    ][kbase + 8 + 2*pair    ],
        //          A_shared[group    ][kbase + 8 + 2*pair + 1] } // matrix 2，右上
        //   %3 = { A_shared[group + 8][kbase + 8 + 2*pair    ],
        //          A_shared[group + 8][kbase + 8 + 2*pair + 1] } // matrix 3，右下
        //
        // 具体例子：lane=5，group=1，pair=1；先令 k_0_1=0（kbase=0）：
        //
        //                       col 0~7                 col 8~15
        //                    +----------------------+----------------------+
        //   row 1            | col 2,3 -> %0       | col 10,11 -> %2     |
        //                    | A[1][2], A[1][3]    | A[1][10], A[1][11] |
        //                    +----------------------+----------------------+
        //   row 9            | col 2,3 -> %1       | col 10,11 -> %3     |
        //                    | A[9][2], A[9][3]    | A[9][10], A[9][11] |
        //                    +----------------------+----------------------+
        //
        // 因此 lane 5 的 A_shared_warp[0:8]（按 half 看）依次是：
        //
        //   { A[1][2], A[1][3], A[9][2], A[9][3],
        //     A[1][10],A[1][11],A[9][10],A[9][11] }
        //       \---- %0 ----/  \---- %1 ----/  \----- %2 -----/  \----- %3 -----/
        //
        // k_0_1=1 时所有列再加 16。注意 lane 5 在上一段提供的 addr 是
        // A_shared[5][kbase]；“提供哪一行地址”和“最终收到哪几个元素”是
        // ldmatrix 的两套映射，不能把 addr 当成 lane 5 自己连续读取 8 half。
        __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
          "{%0, %1, %2, %3}, [%4];\n"
          : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])  // 4 output regs
          : "r"(addr)
        );
      }

      // ===== Load B fragments from shared memory =====
      // Load 4 chunks(each chunk is K16N16) per warp (each ldsm.x4 load k16n16 B_shared to B_Reg)
      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        {
          unsigned int addr;
          // Compute shared memory address for B tile
          // k_0_1 * 2176: B_shared (k16n128) offset (2176 = 16 rows × 136 padded width)
          // threadIdx.y * 64: warp offset in N (each warp handles 64 columns, 128/2=64)
          // ax1_0 * 16: mma offset in N (16 columns per mma n)
          // 以下两个式子非常重要，为了符合ldsm对B_shared buffer上各个thread指向各个起始地址的要求
          // (threadIdx.x & 15) * 136: thread row offset (16 threads, stride=136)
          // (threadIdx.x >> 4) * 8: upper 16 threads have column offset
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(((k_0_1 * 2176) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 136) + ((((int)threadIdx.x) >> 4) * 8))))  // 2176=16×136, 64=N/warp, 16=N/chunk, 136=padded, 8=col offset
          );
          // ldmatrix with transpose (.trans): B needs to be transposed for mma
          // ax1_0 * 8 [0~3]: 每个k16n16里面每个thread own的那8个reg
          // 把8*8矩阵转置为col major 然后再拿
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])  // ax1_0*8: output offset
            : "r"(addr)
          );
        }
      }
      
      // ===== Execute Tensor Core mma instructions =====
      // Compute C[16, 64] += A[16, 16] × B[16, 64] using 64/8=8mma， m16n8k16
      // Split N_tile/2=64 into 4 chunks of N=16 (2 mma processes n16)
      //       32个线程同时来到这里
      //          ↓
      // 收集整个warp的A寄存器
      // 收集整个warp的B寄存器
      //          ↓
      // 组成逻辑：

      // A[16][16]
      // B[16][8]

      //          ↓
      // Tensor Core

      // C[16][8]

      //          ↓
      // 再把C fragment分回32个lane
      for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {  // 4 = 64 / 16 (N per warp / N handled by two mma calls)
        // two mmas compute C[16,16] += A[16,16] × B[16,16]
        // Each call computes C[16,8] += A[16,16] × B[16,8]
        // first mma for half of N dimension (process left 8 cols of N=16)
        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])  // 4 output float regs
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]),  // A: 4 2xfp16 regs 
              "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]),  // 2 fp16x2 regs for n8k16 B
              "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"      (((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));  // C accumulator
        }

        // Second mma for the other half of N dimension (process right 8 cols of N=16)
        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])  // +4 selects the accumulator fragment for the right-side N8
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]),
              "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),  // +4 selects matrix 2/3: the right-side N8
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
      }
    }  // End k_0_1 loop (K=16 chunks)
  }  // End K-loop (_k_0_0)

  // ========== Step 5: Write results back to global memory ==========
  // Write C_warp (fp32 accumulator) back to global memory C (fp16)
  for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {  //  (for each warp, 64 / 16 = 4)
    for (int local_id = 0; local_id < 8; ++local_id) {  // 8 elements for C per m16n16
      // Compute output C global row index
      // blockIdx_y / j_factors1 * 16: M tile base (m16 per tile)
      // threadIdx.x / 4: thread row offfset, 4 threads in one row of one m16n16 tile
      // (local_id % 4) / 2 * 8: additional row offset (0 or 8) based on mma output layout
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 + ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;  // 16=M tile, 4=threads per row, 8=row stride
      
      if (row_offset < M)  // Boundary check
      {
        // Write to C[row_offset, col]
        // ax1_0_1 * 16: N chunk offset (16 columns per chunk for 2 mma calls)
        // row_offset * OC: row stride in C
        // (local_id / 4) * 8: column offset within chunk (0 or 8)
        // local_id % 2: fine-grained column offset (0 or 1) for coalescing
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);  // ax1_0_1 * 8: 每个n16有8个C reg，local_id： 这8个C reg中的每个reg
      }
    }
  }
}  // End kernel


// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor awq_gemm(
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

    // ========== Input validation ==========
    if (num_out_channels % 64 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 64");  // 64: minimum CTA output width
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");  // 8: int32 packs 8×4-bit
    if (group_size % 32 != 0)
	      throw std::invalid_argument("Group size should be a multiple of 32");  // 32: K tile size
    if (num_out_channels % group_size != 0)
        throw std::invalid_argument("OC is not multiple of Group size");

    // ========== Launch kernel ==========
    if (num_out_channels % 128 == 0)  // 128: N tile size
    {
        // Calculate grid dimensions
        int j_factors1 = num_out_channels / 128 / 1;  // Number of N tiles (128 columns per tile)
        
        // Total blocks = (M tiles) × (N tiles) × (split_k slices)
        // Each block processes a 16×128 tile (or a split-K slice of it)
        dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);  // 16: M tile height
        
        // Block shape: 64 threads = 2 warps
        // threadIdx.x: 32 (one warp width)
        // threadIdx.y: 2 (two warps)
        dim3 threads_per_block(32, 2);  // 32×2 = 64 threads
        
        gemm_forward_4bit_cuda_m16n128k32<<<num_blocks, threads_per_block>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, out_feats);
    }
    else
    {
      throw std::invalid_argument("OC is not multiple of 128");  // 128: N tile size
    }
    
    // ========== Split-K reduction ==========
    // If split_k_iters > 1, output is [split_k_iters, M, OC]
    // Sum along split_k dimension to get final result [M, OC]
    return _out_feats.sum(0);  // sum over dim=0 (split_k dimension)
}


/*
================================================================================
只详解 A 的以下两段 PTX：

  1. 把每个 lane 算出的 A_shared C++ 指针转换成 shared-space 地址 addr
  2. 用 ldmatrix.x4 把四个 8x8 FP16 矩阵装入 warp 的寄存器
================================================================================

原始代码 1：生成 shared-memory 地址
--------------------------------------------------------------------------------

  unsigned int addr;
  asm volatile(
    "{ .reg .u64 addr;"
    "  cvta.to.shared.u64 addr, %1;"
    "  cvt.u32.u64 %0, addr; }\n"
    : "=r"(addr)
    : "l"((void *)((&A_shared[k_0_1 * 16])
        + (threadIdx.x & 15) * 40
        + (threadIdx.x >> 4) * 8))
  );

这段 asm 不读取矩阵数据，只计算并转换地址。先把传给 asm 的 C++ 指针
写成容易理解的形式：

  int lane = threadIdx.x;                  // 0..31
  int row  = lane & 15;                    // 0..15 lane  % 16
  int col  = k_0_1*16 + (lane>>4)*8;        // lane>> 4 = lane / 16 
  half* generic_ptr = &A_shared[row*40 + col];

A_shared 的有效数据是 [16,32]，物理行跨度是 40 half：

  每行 = 32 个有效 half + 8 个 padding half


1.1  threadIdx.x & 15 是什么
--------------------------------------------------------------------------------

这里的 & 是按位与，不是 C/C++ 的取地址符。15 的二进制低四位全为 1：

  15 = 0b01111

所以 lane & 15 会清除第 4 位及更高位，只保留最低四位。对 lane=0..31，
它等价于 lane%16：

  lane  0..15 & 15 -> row 0..15
  lane 16..31 & 15 -> row 0..15

例如：

  lane 4  = 0b00100，4  & 15 = 4
  lane 20 = 0b10100，20 & 15 = 4

因此 lane 4 和 lane 20 都提供 A_shared 第 4 行的地址。它们的列不同，
由后面的 lane>>4 决定。


1.2  threadIdx.x >> 4 是什么
--------------------------------------------------------------------------------

右移 4 位相当于对非负整数除以 16：

  lane  0..15 >> 4 = 0 -> col_offset=0
  lane 16..31 >> 4 = 1 -> col_offset=8

所以 lane 4 指向第 4 行的前 8 个 half，lane 20 指向同一行的后 8 个
half：

  lane 4  -> &A_shared[4*40 + k_0_1*16 + 0]
  lane 20 -> &A_shared[4*40 + k_0_1*16 + 8]


1.3  k_0_1*16 是什么
--------------------------------------------------------------------------------

外层一次把 A 的 [16,32] K32 tile 放入 A_shared，但 mma 每次只计算
K=16，所以分成两次 ldmatrix：

  k_0_1=0 -> 当前读取 A_shared[:,  0:16]
  k_0_1=1 -> 当前读取 A_shared[:, 16:32]

例如 lane=20：

  k_0_1=0 -> row=4, col= 8 -> 指向 A_shared[4][ 8]
  k_0_1=1 -> row=4, col=24 -> 指向 A_shared[4][24]

每个地址都指向对应 8-half 行片段的起点。


1.4  32 个 lane 分别提供什么地址
--------------------------------------------------------------------------------

以 k_0_1=0 为例：

  lane  0.. 7 -> A_shared rows 0..7,  每行 col 0 的地址
  lane  8..15 -> A_shared rows 8..15, 每行 col 0 的地址
  lane 16..23 -> A_shared rows 0..7,  每行 col 8 的地址
  lane 24..31 -> A_shared rows 8..15, 每行 col 8 的地址

这些地址描述了四个 8x8 矩阵：

  matrix 0 = A_shared[ 0: 8][0:8]   // 地址由 lane  0.. 7 提供
  matrix 1 = A_shared[ 8:16][0:8]   // 地址由 lane  8..15 提供
  matrix 2 = A_shared[ 0: 8][8:16]  // 地址由 lane 16..23 提供
  matrix 3 = A_shared[ 8:16][8:16]  // 地址由 lane 24..31 提供

四块合起来就是一个 A_shared[0:16][0:16]。当 k_0_1=1 时，所有列
整体加 16，四块合起来变成 A_shared[0:16][16:32]。


1.5  三条 inline PTX 分别做什么
--------------------------------------------------------------------------------

  { .reg .u64 addr;

在花括号限定的 PTX 局部作用域中声明一个临时 64-bit 寄存器 addr。
它与外面的 C++ 变量 unsigned int addr 不是同一个变量，只是名字相同。

  cvta.to.shared.u64 addr, %1;

C++ 中的 &A_shared[...] 是 generic address-space 的 64-bit 指针。
cvta.to.shared 把它转换成 shared address-space 中的地址。

  cvt.u32.u64 %0, addr;

再把 64-bit shared 地址缩成 32 bit，写入 C++ 变量 addr，供下一条
ldmatrix 的 [addr] 使用。一个 CTA 的 shared memory 很小，32-bit offset
足以表示。

inline asm 的操作数编号先数输出、再数输入：

  %0 -> "=r"(addr)        // 32-bit 输出寄存器
  %1 -> "l"(generic_ptr)  // 64-bit 输入寄存器/指针

"=r" 中的 = 表示只写输出，r 表示 32-bit 通用寄存器；l 表示 64-bit
寄存器。volatile 表示编译器不能把这段 asm 当成无用代码删除或随意移动。


原始代码 2：ldmatrix.x4
--------------------------------------------------------------------------------

  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
    "{%0, %1, %2, %3}, [%4];\n"
    : "=r"(((unsigned*)A_shared_warp)[0]),
      "=r"(((unsigned*)A_shared_warp)[1]),
      "=r"(((unsigned*)A_shared_warp)[2]),
      "=r"(((unsigned*)A_shared_warp)[3])
    : "r"(addr)
  );

这是一条 warp collective 指令：整个 warp 的 32 个 lane 必须共同执行，
共同从 shared memory 加载矩阵 fragment，而不是 32 次互不相关的普通 load。


2.1  指令后缀逐项解释
--------------------------------------------------------------------------------

  ldmatrix  : 从 shared memory 加载 Tensor Core 使用的矩阵 fragment
  .sync     : warp 中参与的 lane 在这条指令上同步执行
  .aligned  : 要求 warp 中的线程一致执行该指令；行地址也必须满足自然对齐
  .m8n8     : 每一个被加载的矩阵大小是 8x8
  .x4       : 一次加载四个 8x8 矩阵
  .shared   : 源地址属于 shared address space
  .b16      : 每个矩阵元素是 16 bit，本代码中就是 FP16 的位表示

一个 8x8 FP16 矩阵大小为：

  8 * 8 * 2 bytes = 128 bytes

.x4 总共加载：

  4 * 128 = 512 bytes

整个 warp 每个 lane 得到四个 32-bit 输出寄存器：

  32 lanes * 4 registers * 4 bytes = 512 bytes

数据量正好一致。


2.2  addr 是“当前 lane 提供的行地址”，不是唯一总地址
--------------------------------------------------------------------------------

每个 lane 在第一段 asm 中都算出自己的 addr。对于 .x4：

  lane  0.. 7 的 addr -> matrix 0 的 8 个行起始地址
  lane  8..15 的 addr -> matrix 1 的 8 个行起始地址
  lane 16..23 的 addr -> matrix 2 的 8 个行起始地址
  lane 24..31 的 addr -> matrix 3 的 8 个行起始地址

硬件收集整个 warp 的 32 个 addr 后，完成四个 8x8 的协作加载。不要把
[addr] 理解成“每个 lane 独自从自己的 addr 连续加载 8 个 half 到自己的
四个寄存器”。lane 提供行地址和数据最终分发到 lane 寄存器，是两个不同
的映射过程。


2.3  %0、%1、%2、%3 为什么是四个 32-bit 输出
--------------------------------------------------------------------------------

A_shared_warp 声明为：

  half A_shared_warp[8];

它总共是 8*2=16 bytes。代码把它看作四个 unsigned：

  ((unsigned*)A_shared_warp)[0] -> 32 bit -> 2 个 half
  ((unsigned*)A_shared_warp)[1] -> 32 bit -> 2 个 half
  ((unsigned*)A_shared_warp)[2] -> 32 bit -> 2 个 half
  ((unsigned*)A_shared_warp)[3] -> 32 bit -> 2 个 half

因此每个 lane 的四个输出寄存器总共保存：

  4 * 32 bit = 128 bit = 8 个 FP16

对于 .x4，每一个输出寄存器保存对应矩阵分发给该 lane 的一个 fragment：

  %0 -> matrix 0 的 fragment（两个 FP16）
  %1 -> matrix 1 的 fragment（两个 FP16）
  %2 -> matrix 2 的 fragment（两个 FP16）
  %3 -> matrix 3 的 fragment（两个 FP16）

所有 lane 的 %0 合起来是完整 matrix 0；所有 lane 的 %1 合起来是完整
matrix 1；%2、%3 同理。


2.4  一个具体 lane 的输入地址和输出寄存器不是同一件事
--------------------------------------------------------------------------------

仍以 k_0_1=0、lane=20 为例。第一段 asm 算出的输入地址是：

  addr(lane20) = &A_shared[4][8]

它是 matrix 2 第 4 行的起始地址，lane 20 的职责之一是把这个行地址提供
给 ldmatrix。但执行 .x4 后，lane 20 自己得到四个寄存器，分别含有
matrix 0、1、2、3 按 MMA fragment 布局分给 lane 20 的两个 FP16。

以不带 .trans 的 m8n8.b16 行主序加载来看，每连续四个 lane 合作接收
一个矩阵行的 16 bytes，每个 lane 得到其中一个 32-bit（两个 FP16）。
例如 lane 4..7 合作接收每个 matrix 的第 1 行：

  lane 4 -> 该行 col 0:2
  lane 5 -> 该行 col 2:4
  lane 6 -> 该行 col 4:6
  lane 7 -> 该行 col 6:8

这套分发对四个 matrix 同时进行，结果分别进入每个 lane 的四个目标
寄存器。随后 mma.sync 直接按规定的 fragment 布局消费这些寄存器。


两段代码连起来的最简伪代码
--------------------------------------------------------------------------------

  // 每个 lane：只负责提供一个 8-half 行片段的 shared 起始地址
  generic_ptr = &A_shared[row_for_this_lane][col_for_this_lane];
  addr = convert_generic_pointer_to_shared_address(generic_ptr);

  // 整个 warp：收集 32 个地址，加载四个 8x8
  {reg0, reg1, reg2, reg3} = warp_ldmatrix_x4(all_lanes_addr);

  // 当前 lane 最终得到 4 个 b32 = 8 个 half，供 mma 使用

核心区别：

  第一段 asm：只生成地址，不搬数据。
  第二段 asm：整个 warp 根据 32 个地址协作搬数据，并按 MMA 布局分发
              到每个 lane 的四个 32-bit 寄存器。

================================================================================
*/
/*用一个简化的 2×2 数值例子最容易理解。真实指令处理 8×8/K16，原理完全一样。
假设：
A = [2  3]

B = [ 5   7
     11  13]
要计算：
C = A × B
正确结果：
C[0] = 2×5  + 3×11 = 43
C[1] = 2×7  + 3×13 = 53

C = [43  53]
1. B 在 shared memory 中横着存
B 的布局是 [K,N]：
              N
             n0   n1
          +----------+
K   k0    |  5    7 |  ← shared中连续的一行
    k1    | 11   13 |  ← shared中连续的一行
          +----------+
shared memory 里的线性存储：
5, 7, 11, 13
因此 ldmatrix 提供的地址是每一行的起点：
k0行地址 → [5, 7]
k1行地址 → [11, 13]
这就是所谓的“横着读取”。
2. .trans 在寄存器中重新排列
执行：
ldmatrix...trans.shared.b16
以后，逻辑上的寄存器视图变成：
              K
             k0   k1
          +----------+
N   n0    |  5   11 |  ← B的第0列
    n1    |  7   13 |  ← B的第1列
          +----------+
也就是：
B的第0列：[5, 11]
B的第1列：[7, 13]
注意，shared memory 本身并没有被修改。.trans 只是把读取结果按 column-major fragment 规则分发到各 lane 的寄存器。NVIDIA PTX ISA将 .trans 定义为以 column-major 格式加载矩阵。
3. mma.row.col 做点积
指令：
mma.sync.aligned.m16n8k16.row.col...
                               ^^^ ^^^
                                A   B
要求：
A：row fragment
B：column fragment
因此 Tensor Core 看到的是：
A的行：[2, 3]

B的第0列：[5, 11]
B的第1列：[7, 13]
然后计算：
              B的第0列
C[0] = [2,3] · [5,11]
     = 2×5 + 3×11
     = 43

              B的第1列
C[1] = [2,3] · [7,13]
     = 2×7 + 3×13
     = 53
所以完整过程：
B在shared：
[ 5   7 ]   横着读
[11  13 ]
     │
     │ ldmatrix.trans
     ▼
B在register的逻辑布局：
[ 5  11 ]   第0列
[ 7  13 ]   第1列
     │
     │ mma.row.col
     ▼
A的行 × B的列
[2,3] · [5,11] = 43
[2,3] · [7,13] = 53
如果没有 .trans
假设依然使用：
mma...row.col
但 B 用不带 .trans 的方式加载，寄存器可能按照行 fragment 排列：
[5, 7]
[11, 13]
Tensor Core却把它当成列 fragment，于是概念上会得到：
错误 C[0] = [2,3] · [5,7]
          = 2×5 + 3×7
          = 31

错误 C[1] = [2,3] · [11,13]
          = 2×11 + 3×13
          = 61
即类似误算成：
A × Bᵀ
而不是 A × B。
对应到当前真正的 K16×N16
当前代码一次准备：
A fragment：[M16,K16]
B fragment：[K16,N16]
对某个输出 C[m,n]：
A寄存器提供：
A[m][k0], A[m][k1], ..., A[m][k15]

B经过ldmatrix.trans后提供：
B[k0][n], B[k1][n], ..., B[k15][n]
MMA 计算：
C[m][n] +=
    A[m][k0]  × B[k0][n]
  + A[m][k1]  × B[k1][n]
  + ...
  + A[m][k15] × B[k15][n]
所以关键不是“B 从 shared 横着还是竖着读”，而是：
物理读取：沿shared中的连续行横着读取
寄存器排列：ldmatrix.trans转成column fragment
数学计算：A的一行与B的一列做点积*/

/*
================================================================================
B 的写入、ldmatrix.trans 和 mma：按矩阵形状看索引
================================================================================

下面全部使用当前 CTA 内的局部坐标：

  B_shared[k][n]，k = 0~31，n = 0~127

真实地址每行有 136 个 half，其中 n=128~135 是 padding。为了让图容易看，
下面只画真正存放数据的 n=0~127。


1. B_loaded_fp16 怎样写进 B_shared
--------------------------------------------------------------------------------

写入代码：

  *(uint4*)(B_shared_ptr
      + ax0_ax1_fused_0 * 4 * 136) = B_loaded_fp16;

一个 uint4 = 16 bytes = 8 个 half，所以每个线程横向写连续 8 列。
令：

  r    = ax0_ax1_fused_0;  // 写入轮次，0~7
  warp = threadIdx.y;      // 0 或 1
  lane = threadIdx.x;      // 0~31
  q    = lane >> 4;        // lane/16，0 或 1
  p    = lane & 15;        // lane%16，0~15

该线程最终写入：

  shared_row = 4*r + 2*warp + q;
  shared_col = 8*p ... 8*p+7;

一轮 r 内，64 个线程的分工是：

                 lane 范围       B_shared 行       每个 lane 写入的列
              +--------------+----------------+--------------------------+
  warp 0      |    0~15      |     4*r        | 8*p ... 8*p+7           |
              +--------------+----------------+--------------------------+
  warp 0      |   16~31      |     4*r+1      | 8*p ... 8*p+7           |
              +--------------+----------------+--------------------------+
  warp 1      |    0~15      |     4*r+2      | 8*p ... 8*p+7           |
              +--------------+----------------+--------------------------+
  warp 1      |   16~31      |     4*r+3      | 8*p ... 8*p+7           |
              +--------------+----------------+--------------------------+

把一轮写入直接画成 B_shared 的 4x128 条带：

                         N / 每格表示连续 8 列
             0~7    8~15   16~23              112~119 120~127
           +-------+-------+-------+----- ... -----+-------+-------+
  K=4*r    | W0 L0 | W0 L1 | W0 L2 |             | W0L14 | W0L15 |
           +-------+-------+-------+----- ... -----+-------+-------+
  K=4*r+1  | W0L16 | W0L17 | W0L18 |             | W0L30 | W0L31 |
           +-------+-------+-------+----- ... -----+-------+-------+
  K=4*r+2  | W1 L0 | W1 L1 | W1 L2 |             | W1L14 | W1L15 |
           +-------+-------+-------+----- ... -----+-------+-------+
  K=4*r+3  | W1L16 | W1L17 | W1L18 |             | W1L30 | W1L31 |
           +-------+-------+-------+----- ... -----+-------+-------+

8 轮恰好覆盖 K=0~31：

  r=0 -> K 行  0~3
  r=1 -> K 行  4~7
  r=2 -> K 行  8~11
  ...
  r=7 -> K 行 28~31

具体例子：r=2、warp=0、lane=18。

  q = 18>>4 = 1
  p = 18&15 = 2
  row = 4*2 + 2*0 + 1 = 9
  col = 8*2 ... 8*2+7 = 16...23

因此这个线程把反量化后的 8 个 half 横着写到：

  B_shared[9][16...23]


2. 一个 warp 怎样从 B_shared 选择 K16xN16
--------------------------------------------------------------------------------

计算阶段固定一次 k_0_1、threadIdx.y 和 ax1_0：

  kbase = k_0_1 * 16;                  // 0 或 16
  nbase = threadIdx.y * 64 + ax1_0*16; // 当前 warp 的一个 N16

选择出的逻辑矩阵是：

  B_shared[kbase : kbase+15][nbase : nbase+15]

                         N
                    nbase       nbase+7 nbase+8      nbase+15
                         +-------------+-------------+
  K  kbase ... kbase+7  |  matrix 0   |  matrix 2   |
                         +-------------+-------------+
     kbase+8...kbase+15 |  matrix 1   |  matrix 3   |
                         +-------------+-------------+

这四块是 ldmatrix.m8n8.x4 眼中的四张独立 8x8：

  lane  0~7  提供 matrix 0 的 8 个行地址
  lane  8~15 提供 matrix 1 的 8 个行地址
  lane 16~23 提供 matrix 2 的 8 个行地址
  lane 24~31 提供 matrix 3 的 8 个行地址

注意：这里说的是“提供 shared 行地址”。数据最后进入哪个 lane 的寄存器，
由下一条 ldmatrix.trans 的 fragment 规则决定。


3. ldmatrix.x4.trans 后，一个 lane 在 B 寄存器里拿到什么
--------------------------------------------------------------------------------

对于当前 warp 中任意 lane，令：

  n_local = lane >> 2;  // 0~7，在每张 8x8 中选择一列
  k_pair  = lane & 3;   // 0~3，在每张 8x8 中选择一对 K 行

因为使用了 .trans，每个 lane 在每张独立 8x8 中得到的是竖直的两个 half：

                              N 的局部列
                 n0       n1       n2             n6       n7
              +--------+--------+--------+-- ... --+--------+--------+
  K row 0,1   | lane 0 | lane 4 | lane 8 |         |lane 24 |lane 28 |
              +--------+--------+--------+-- ... --+--------+--------+
  K row 2,3   | lane 1 | lane 5 | lane 9 |         |lane 25 |lane 29 |
              +--------+--------+--------+-- ... --+--------+--------+
  K row 4,5   | lane 2 | lane 6 |lane 10 |         |lane 26 |lane 30 |
              +--------+--------+--------+-- ... --+--------+--------+
  K row 6,7   | lane 3 | lane 7 |lane 11 |         |lane 27 |lane 31 |
              +--------+--------+--------+-- ... --+--------+--------+

同一个 lane 在四张 8x8 中的局部位置完全相同。四个 b32 输出为：

  %0 = matrix 0 的竖直 2 half -> 左上，低 K、左 8 列
  %1 = matrix 1 的竖直 2 half -> 左下，高 K、左 8 列
  %2 = matrix 2 的竖直 2 half -> 右上，低 K、右 8 列
  %3 = matrix 3 的竖直 2 half -> 右下，高 K、右 8 列

写成当前 K16xN16 的真实坐标。令 g=n_local、p=k_pair：

  %0 = { B[kbase + 2*p    ][nbase + g],
         B[kbase + 2*p + 1][nbase + g] }       // 左上

  %1 = { B[kbase + 8 + 2*p    ][nbase + g],
         B[kbase + 8 + 2*p + 1][nbase + g] }   // 左下

  %2 = { B[kbase + 2*p    ][nbase + 8 + g],
         B[kbase + 2*p + 1][nbase + 8 + g] }   // 右上

  %3 = { B[kbase + 8 + 2*p    ][nbase + 8 + g],
         B[kbase + 8 + 2*p + 1][nbase + 8 + g] }// 右下

因此每个 ax1_0 的 8 个 half 在 B_shared_warp 中是：

  half offset       0,1          2,3          4,5          6,7
                 +------------+------------+------------+------------+
  来源矩阵       | matrix 0   | matrix 1   | matrix 2   | matrix 3   |
                 +------------+------------+------------+------------+
  PTX 输出       |    %0      |    %1      |    %2      |    %3      |
                 +------------+------------+------------+------------+
  N 范围         | 左边 8 列  | 左边 8 列  | 右边 8 列  | 右边 8 列  |
                 +------------+------------+------------+------------+
  K 范围         | 低 8 行    | 高 8 行    | 低 8 行    | 高 8 行    |
                 +------------+------------+------------+------------+

具体例子：lane=5，因此 g=1、p=1；先假设 kbase=0、nbase=32：

  %0 = { B[2][33],  B[3][33]  }
  %1 = { B[10][33], B[11][33] }
  %2 = { B[2][41],  B[3][41]  }
  %3 = { B[10][41], B[11][41] }

可见 .trans 后，单个寄存器中的两个 half 沿 K 方向竖着排列；整个 warp
合起来才拥有供 MMA 使用的完整 B[K16,N16] fragment。


4. 四次 ax1_0 怎样填满 B_shared_warp[32]
--------------------------------------------------------------------------------

每次 ax1_0 加载一个 K16xN16，每个 lane 收到 8 half。4 次正好是 32 half：

  B_shared_warp（当前 lane 私有）

       half 0             8              16             24            32
          +---------------+---------------+---------------+---------------+
          | ax1_0=0       | ax1_0=1       | ax1_0=2       | ax1_0=3       |
          | warp N  0~15  | warp N 16~31  | warp N 32~47  | warp N 48~63  |
          +---------------+---------------+---------------+---------------+

这里的 N 是相对当前 warp 的局部 N。warp 0 对应 CTA 的 N=0~63，warp 1
对应 CTA 的 N=64~127。


5. j_0_4 四轮 MMA 到底分别算哪一块
--------------------------------------------------------------------------------

代码：

  for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4)

从矩阵形状看，4 的来源应理解为：

  4 = 每个 warp 的 64 列 / 每轮处理的 16 列

从寄存器容量看也等价于：

  4 = B_shared_warp 的 32 half / 每轮使用的 8 half

每一轮 j_0_4 处理一个 M16xN16 输出块，但一条 PTX 指令只能计算 N8，
所以代码在一轮中连续发出两条 mma：

                               当前 warp 的 N=0~63
                 0       8      16      24      32      40      48      56     64
                 +-------+-------+-------+-------+-------+-------+-------+-------+
  j_0_4=0        | mma 0 | mma 1 |
                 +-------+-------+
  j_0_4=1                        | mma 0 | mma 1 |
                                 +-------+-------+
  j_0_4=2                                        | mma 0 | mma 1 |
                                                 +-------+-------+
  j_0_4=3                                                        | mma 0 | mma 1 |
                                                                 +-------+-------+

对应关系表：

  j_0_4   ax1_0装入的N16   第一条 mma 使用       第二条 mma 使用
  ------  --------------  --------------------  --------------------
    0          0~15       B_warp half  0~3      B_warp half  4~7
    1         16~31       B_warp half  8~11     B_warp half 12~15
    2         32~47       B_warp half 16~19     B_warp half 20~23
    3         48~63       B_warp half 24~27     B_warp half 28~31

第一条 mma 的两个 b32 正是 matrix 0、matrix 1，也就是完整 K16 对应的
左 8 列；第二条 mma 的两个 b32 是 matrix 2、matrix 3，也就是完整 K16
对应的右 8 列：

                         B 的当前 K16xN16
                    左 8 列              右 8 列
               +-------------------+-------------------+
  K 低 8 行    | %0 / first mma    | %2 / second mma   |
               +-------------------+-------------------+
  K 高 8 行    | %1 / first mma    | %3 / second mma   |
               +-------------------+-------------------+

所以代码中的 +4 是“跳过 4 个 half，来到右边 N8 的 %2”，不是跳到下一
个 K 区块，也不是跳到输出矩阵的下 8 行。


6. 一条 mma 如何形成 C
--------------------------------------------------------------------------------

指令形状：

  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                           |  |  |
                           M  N  K

整个 warp 共同执行一次指令，数学上完成：

  C[16,8] += A[16,16] * B[16,8]

也就是对于当前输出块内的任意 m、n：

  C[m][n] += sum(k=0..15) A[m][kbase+k] * B[kbase+k][n]

两条 mma 拼成 N16，四轮 j_0_4 拼成当前 warp 的 N64：

       A[M16,K16]              B[K16,N64]                 C[M16,N64]
      +-----------+      +----+----+----+----+      +----+----+----+----+
      |           |      | B0 | B1 | B2 | B3 |      | C0 | C1 | C2 | C3 |
      |    A      |  x   |N16 |N16 |N16 |N16 |  =   |N16 |N16 |N16 |N16 |
      |           |      |    |    |    |    |      |    |    |    |    |
      +-----------+      +----+----+----+----+      +----+----+----+----+
                              j=0  j=1  j=2  j=3

两个 warp 再沿 N 方向拼成 CTA 的 M16xN128：

                         CTA 输出 C[M16,N128]
               +--------------------------+--------------------------+
               | warp 0: N=0~63           | warp 1: N=64~127         |
               | j=0  j=1  j=2  j=3       | j=0  j=1  j=2  j=3       |
               +--------------------------+--------------------------+

最后还要注意：单个 lane 的 A fragment、B fragment 和 C fragment 不能当成
一次普通的“线程内向量点积”。mma 是 warp 集体指令，Tensor Core 会使用
32 个 lane 的全部 A/B 寄存器共同产生 32 个 lane 的 C 寄存器。

以 lane=5 为例，它提供的某些 B 值可能属于 B 的第 nbase+1 列，而它最终
接收的 C 寄存器却位于第 nbase+2、nbase+3 列；中间的跨 lane 配对由 MMA
硬件完成。这正是为什么不能只盯着同一个线程的 A 和 B 寄存器去手算结果。


7. 三层循环的整体关系
--------------------------------------------------------------------------------

  k_0_0：选择一个 K32 大块
      |
      +-- k_0_1=0：用 K=0~15 进行累加
      |       |
      |       +-- ax1_0=0~3：预先装好当前 warp 的四个 B[K16,N16]
      |       |
      |       +-- j_0_4=0~3：每轮用两条 MMA 算一个 C[M16,N16]
      |
      +-- k_0_1=1：用 K=16~31 继续累加到同一批 C_warp

外层下一个 k_0_0 会再取下一个 K32，并继续累加。最终 K 方向全部完成后，
C_warp 才被转换成 half 并写回全局内存。

================================================================================
*/
