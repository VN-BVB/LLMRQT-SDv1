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

/*
 * Double Buffered GEMM Kernel with Prologue/Mainloop/Epilogue
 * 
 * Key optimizations:
 * 1. Shared memory double buffering: overlap data loading with computation
 * 2. Register double buffering: overlap ldmatrix with mma
 * 3. Software pipelining: Prologue -> Mainloop -> Epilogue structure
 * 
 * Pipeline stages:
 *   Prologue:   Load tile 0
 *   Mainloop:   For each tile k:
 *                 - Compute with tile k (using buffer A)
 *                 - Load tile k+1 (into buffer B)
 *                 - Swap buffers
 *   Epilogue:   Compute last tile
 */
__global__ void __launch_bounds__(64) gemm_forward_4bit_cuda_m16n128k32_db(
    int G, int split_k_iters, 
    half* __restrict__ A, int* __restrict__ B, 
    half* __restrict__ scaling_factors, int* __restrict__ zeros, 
    int M, int IC, int OC, 
    half* __restrict__ C) 
{
  static constexpr uint32_t ZERO = 0x0;
  
  // Accumulator registers
  float C_warp[32];
  
  // ========== Double-buffered shared memory ==========
  // Buffer 0 and Buffer 1 for ping-pong buffering
  __shared__ half A_shared[2][16 * (32 + 8)];  // 2 buffers for A
  __shared__ half B_shared[2][32 * (128 + 8)]; // 2 buffers for B
  
  // Decode block indices
  int j_factors1 = ((OC + 128 - 1) / 128);
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  // ========== Double-buffered register storage ==========
  // Two sets of registers for A and B fragments
  half A_shared_warp[2][8];   // Ping-pong A fragments
  half B_shared_warp[2][32];  // Ping-pong B fragments
  
  // Initialize accumulator
  for (int j_0_4_init = 0; j_0_4_init < 4; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  // Stride calculations
  static constexpr int row_stride_warp = 32 * 8 / 32;  // = 8
  static constexpr int row_stride = 2 * 32 * 8 / 128;  // = 4
  
  bool ld_A_flag = (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;

  // ========== Global memory pointers ==========
  half* A_ptr = A 
                + (((int)blockIdx_y) / j_factors1 * 16
                   + (((int)threadIdx.y) * row_stride_warp)
                   + ((int)threadIdx.x) / (32 / 8)) * IC
                + (((int)threadIdx.x) % (32 / 8)) * 8;
  
  int* B_ptr = B
            + (((int)blockIdx_y) % j_factors1) * (128 / 8)
            + ((int)threadIdx.y) * (OC / 8) * 2
            + (((int)threadIdx.x) / (128 / 8)) * (OC / 8)
            + (((int)threadIdx.x) % (128 / 8)) * 1;
  
  int* zeros_ptr = zeros
                + (((int)blockIdx_y) % j_factors1) * (128 / 8)
                + ((int)threadIdx.x) % (128 / 8);
  
  half* scaling_factors_ptr = scaling_factors
                            + (((int)blockIdx_y) % j_factors1) * (128)
                            + (((int)threadIdx.x) % (128 / 8)) * 8;

  half* C_ptr = C 
              + blockIdx_z * M * OC
              + (((int)blockIdx_y) % j_factors1) * 128
              + ((int)threadIdx.y) * 64
              + (((int)threadIdx.x) % 4) * 2;

  // ========== Shared memory pointers (double buffered) ==========
  half* A_shared_ptr[2];
  half* B_shared_ptr[2];
  
  for (int buf = 0; buf < 2; buf++) {
    A_shared_ptr[buf] = A_shared[buf]
                      + ((int)threadIdx.y) * row_stride_warp * (32 + 8)
                      + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                      + (((int)threadIdx.x) % (32 / 8)) * 8;
    
    B_shared_ptr[buf] = B_shared[buf]
                      + ((int)threadIdx.y) * (row_stride / 2) * (128 + 8)
                      + (((int)threadIdx.x) / (128 / 8)) * (128 + 8)
                      + (((int)threadIdx.x) % (128 / 8)) * 8;
  }

  // Calculate K-loop bounds
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  
  if (k_bound == 0) return;  // Early exit if no work

  // ========================================================================
  // PROLOGUE: Load first tile into buffer 0
  // ========================================================================
  {
    int k_0_0 = 0 * split_k_iters + blockIdx_z;
    int write_buf = 0;  // Write to buffer 0
    
    // Load A tile
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr[write_buf]) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr[write_buf]) = make_uint4(0, 0, 0, 0);
    }

    // 加载当前 K32 tile 使用的量化参数。
    //
    // qweight: [IC,   OC/8]，一个 uint32 打包 8 个 UINT4 weight。
    // qzeros:  [IC/G, OC/8]，一个 uint32 打包 8 个 UINT4 zero point。
    // scales:  [IC/G, OC]，每个 (K-group, output channel) 一个 FP16 scale。
    //
    // 当前 tile 从 K = k_0_0 * 32 开始，所以：
    //   group_id = (k_0_0 * 32) / G
    // 当 G=128 时，连续 4 个 K32 tile 共用同一组 zero 和 scale。
    //
    // 下面只从一个线程的视角看地址。令：
    //   lane       = threadIdx.x                         // 0..31
    //   warp       = threadIdx.y                         // 0..1
    //   n_tile     = blockIdx_y % j_factors1             // CTA 的 N128 编号
    //   packed_n   = lane % 16                           // 本线程的 N8 编号
    //   n0         = n_tile * 128 + packed_n * 8         // 全局首个 N 通道
    //   k_local0   = warp * 2 + lane / 16                // tile 内首个 K row
    //   k_global0  = k_0_0 * 32 + k_local0
    //   group_id   = (k_0_0 * 32) / G
    //
    // 当前线程固定负责全局列 N[n0:n0+8]，并在下面 8 次循环中负责 K row：
    //   k_global0 + {0,4,8,12,16,20,24,28}
    // zeros_ptr/scaling_factors_ptr 已包含 n_tile 和 packed_n 对应的 N 偏移。
    // 因为 G 是 32 的倍数，一个 K32 tile 不跨 group，所以这 8 个 K row
    // 可复用同一个 zero 向量和 scale 向量。
    //
    // qzeros 的物理矩阵（每格是一个 uint32，打包 8 个 zero）：
    //
    //                         packed-N (= N/8)
    //                   ...  p-1       p=packed_n       p+1  ...
    //                 +-----+---------+================+-----+
    //   group_id - 1  | ... |         |                | ... |
    //                 +-----+---------+================+-----+
    //   group_id      | ... |         | z[n0:n0+8]     | ... | <- 本线程读取
    //                 +-----+---------+================+-----+
    //   group_id + 1  | ... |         |                | ... |
    //                 +-----+---------+================+-----+
    //
    // 具体例子：n_tile=0、threadIdx=(x=3,y=0) 时，packed_n=3、n0=24、
    // k_local0=0；该线程读取 zero[group_id, N=24:32]，并把它用于
    // B 的 K={0,4,8,12,16,20,24,28}, N=24:32（K 再加当前 tile 的起点）。
    //
    // 对应的物理地址正是：qzeros[group_id][n_tile*16 + packed_n]。
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));

    // 将 1 个 uint32 中的 8 个 UINT4 zero 展开成 8 个 FP16。
    // uint4 只是 128-bit 容器，x/y/z/w 各保存一个 half2。
    // AWQ pack 的 nibble 顺序为 {0,2,4,6,1,3,5,7}，快速转换函数利用该顺序
    // 恢复成四个逻辑相邻的 half2：
    //
    //   zeros_loaded 的 nibble slot：
    //     +----+----+----+----+----+----+----+----+
    //     | s7 | s6 | s5 | s4 | s3 | s2 | s1 | s0 |  (高 bit -> 低 bit)
    //     +----+----+----+----+----+----+----+----+
    //     | z7 | z5 | z3 | z1 | z6 | z4 | z2 | z0 |
    //     +----+----+----+----+----+----+----+----+
    //
    //   B_loaded_zero.x = half2(z0,z1) = zero[group_id, n0+0:n0+2]
    //   B_loaded_zero.y = half2(z2,z3) = zero[group_id, n0+2:n0+4]
    //   B_loaded_zero.z = half2(z4,z5) = zero[group_id, n0+4:n0+6]
    //   B_loaded_zero.w = half2(z6,z7) = zero[group_id, n0+6:n0+8]
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);

    // 一次 uint4 load 是 16 bytes，正好加载 8 个连续 FP16 scale，
    // 与一个 packed weight word 内的 8 个输出通道逐一对应。
    // 即 B_loaded_scale = scale[group_id, n0:n0+8]，其 x/y/z/w 与上述
    // B_loaded_zero 的四个 half2 完全对齐。
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));

    // qweight 的逻辑 shape 是 [IC, OC/8]。跳过 k_0_0 * 32 个 K row，
    // 到达当前 K32 tile 的第一行 packed weight。
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    // 加载并反量化当前线程负责的 8(K row) x 8(N channel) 子集。
    // 注意：K32 x N128 是整个 64-thread CTA 的 tile，不是一个线程的 tile。
    // row_stride=4：一个线程访问 k_local0、k_local0+4、...、k_local0+28；
    // B_ptr 中已有 k_local0 和 packed_n 偏移，所有线程合起来覆盖整块。
    //
    // 单线程所取的 weight（行在完整 K32 中是跨 4 采样，列是连续 N8）：
    //
    //                         本线程固定的 N[n0:n0+8]
    //                  n0   n0+1 n0+2 n0+3 n0+4 n0+5 n0+6 n0+7
    //                +----+----+----+----+----+----+----+----+
    // k_local0 +  0  | w  | w  | w  | w  | w  | w  | w  | w  | loop 0
    //                +----+----+----+----+----+----+----+----+
    // k_local0 +  4  | w  | w  | w  | w  | w  | w  | w  | w  | loop 1
    //                +----+----+----+----+----+----+----+----+
    // k_local0 +  8  | w  | w  | w  | w  | w  | w  | w  | w  | loop 2
    //                +----+----+----+----+----+----+----+----+
    //       ...      |                ...                 |
    //                +----+----+----+----+----+----+----+----+
    // k_local0 + 28  | w  | w  | w  | w  | w  | w  | w  | w  | loop 7
    //                +----+----+----+----+----+----+----+----+
    //
    // 上面每一行都广播使用同一组：
    //   zero  = [z0,z1,z2,z3,z4,z5,z6,z7]
    //   scale = [s0,s1,s2,s3,s4,s5,s6,s7]
    // 对格子 (i,j) 执行：
    //   k = k_0_0*32 + k_local0 + 4*i
    //   n = n0 + j
    //   B_deq[k,n] = (B_q4[k,n] - zero[group_id,n]) * scale[group_id,n]
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {
      // 本轮从 K row = k_global0 + 4*ax0_ax1_fused_0 读取 N[n0:n0+8]；
      // 这 8 个相邻 UINT4 weight 按 AWQ 顺序打包在一个 uint32 中。
      uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));

      // 展开为 8 个 FP16；与 zero/scale 相同，x/y/z/w 分别保存
      // 逻辑相邻通道 {w0,w1}、{w2,w3}、{w4,w5}、{w6,w7} 四个 half2。
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      
      // 在寄存器中完成 AWQ groupwise-per-output-channel 反量化：
      //   W_fp16[k,n] = (W4[k,n] - zero[k/G,n]) * scale[k/G,n]
      //
      // sub.f16x2 与 fma.rn.f16x2 每条指令同时处理两个 FP16；FMA 计算
      // value * scale + 0，其中 ZERO 是 packed half2 {0,0}。因为当前 K32
      // tile 位于同一个量化 group 中，循环中的 K row 复用这 8 个 zero/scale。

      // 本线程局部通道 0、1（全局通道 n0+0、n0+1）。
      // B_loaded_fp16.x = half2(q0, q1)
      // B_loaded_zero.x = half2(z0, z1)
      // sub.f16x2 会把每个 32-bit 寄存器解释成两个 FP16，然后逐元素相减：
      // B_loaded_fp16.x = half2(q0-z0, q1-z1)
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      // B_loaded_fp16.x = half2(q0-z0, q1-z1)
      // B_loaded_scale.x = half2(s0, s1)
      // ZERO = half2(0.0, 0.0)
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));

      // 本线程局部通道 2、3（全局通道 n0+2、n0+3）。
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));

      // 本线程局部通道 4、5（全局通道 n0+4、n0+5）。
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));

      // 本线程局部通道 6、7（全局通道 n0+6、n0+7）。
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      
      // 用一次 uint4 store 将 8 个反量化后的 FP16 weight 写入 shared memory。
      // 逻辑 tile 为 [32,128]，物理行跨度为 136；多出的 8 个 FP16 是 padding，
      // 用来改善后续 ldmatrix.trans 的 shared-memory bank conflict。
      // 对本线程的 (i,j)，写入 shared 的逻辑坐标为：
      //   shared_k = k_local0 + 4*i
      //   shared_n = packed_n*8 + j
      // 即 global 的 N128 tile 起点在 shared 中被归零，只保留 CTA 内部坐标。
      // 这里不会在 global memory 中物化完整的 FP16 权重矩阵。
      *(uint4*)(B_shared_ptr[write_buf] + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
    }
    
    __syncthreads();  // Wait for prologue to complete
  }

  // ========================================================================
  // MAINLOOP: Software pipelined computation (Load next + Compute current)
  // ========================================================================
  // Prologue 已经把逻辑 tile 0 放进 shared buffer 0。Mainloop 的每轮处理：
  //
  //   1. 从 compute_buf 读取当前 K32 tile，完成两组 K16 Tensor Core MMA；
  //   2. 把下一 K32 tile 从 global memory 搬到另一个 load_buf，并现场反量化 B；
  //   3. 下一轮交换两个 buffer 的角色。
  //
  // 迭代 t 的 buffer 状态为：
  //   compute_buf = t % 2       ：只读，保存 tile t；
  //   load_buf    = (t + 1) % 2 ：只写，准备 tile t+1。
  //
  // 循环只运行到 k_bound-2，因为每轮还要预取 tile t+1；最后一个已经装好的
  // tile 留给后面的 EPILOGUE 计算，避免 mainloop 内额外的末轮边界分支。
  // 注意：当前代码在 compute 和 load 之间有 __syncthreads()，因此两阶段在
  // 指令时间线上是先算后搬，并非真正并发；双 buffer 在这里主要保证 ping-pong
  // 生命周期正确。若要实质隐藏 global-memory latency，需要进一步使用 cp.async
  // 或调整 warp/stage 调度，不能只依赖两个 shared buffer。
  // 同理，下面所有 ldmatrix/MMA 都只使用 A_shared_warp[0] 和
  // B_shared_warp[0]；数组声明出的 [1] 当前没有参与寄存器级 ping-pong，因而
  // 这版代码也尚未真正实现文件头所说的 register double buffering。
  for (int _k_0_0 = 0; _k_0_0 < k_bound - 1; ++_k_0_0) {
    // _k_0_0 是当前 split-K 分片内部的局部 tile 序号。
    int compute_buf = _k_0_0 % 2;      // Buffer to compute from
    int load_buf = (_k_0_0 + 1) % 2;   // Buffer to load into

    // split-K 后，本 block 只访问全局 K32 tile：
    //   blockIdx_z, blockIdx_z+split_k_iters, blockIdx_z+2*split_k_iters, ...
    // 这里求的是下一轮要加载的全局 K32 tile 编号。
    int k_0_0_load = (_k_0_0 + 1) * split_k_iters + blockIdx_z;

    // ===== Compute with current tile (from compute_buf) =====
    // Inner K loop: split K=32 into 2 × K=16 for mma
    // mma.sync 的 K 固定为 16，而 shared stage 保存 K=32，所以分别计算：
    //   k_0_1=0 -> 当前 tile 的 K[0:16]
    //   k_0_1=1 -> 当前 tile 的 K[16:32]
    // 两次结果都累加到同一组 FP32 C_warp 寄存器中。
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      // Load A fragment from shared memory to registers
      {
        // ldmatrix 使用 32-bit shared-memory address。先由通用 64-bit 指针
        // 得到 shared 地址，再通过 cvta/cvt 转成 ldmatrix 接受的 32-bit offset。
        unsigned int addr;
        __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(addr)
          : "l"((void *)((&(A_shared[compute_buf][(k_0_1 * 16)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
        );

        // A_shared[compute_buf] 的逻辑形状是 [16,32]，物理行跨度为 40：
        //   k_0_1*16             选择当前 K16 子块；
        //   (lane & 15)*40       给出 row 0..15 的行地址；
        //   (lane >> 4)*8        后 16 个 lane 指向该行的后 8 列。
        //
        // ldmatrix.x4 是 warp 协同指令：32 个 lane 给出的地址共同描述四个
        // 8x8 FP16 小矩阵，合起来构成 MMA 所需的 A[16,16] fragment。
        // 每个 lane 得到 4 个 b32 寄存器，每个 b32 内含两个 FP16，因此这里
        // 用 half[8] 作存储容器，再按 unsigned[4] 传给后面的 mma。
        __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
          "{%0, %1, %2, %3}, [%4];\n"
          : "=r"(((unsigned *)(A_shared_warp[0] + 0))[0]), 
            "=r"(((unsigned *)(A_shared_warp[0] + 0))[1]), 
            "=r"(((unsigned *)(A_shared_warp[0] + 0))[2]), 
            "=r"(((unsigned *)(A_shared_warp[0] + 0))[3])
          : "r"(addr)
        );
      }

      // Load B fragments and execute MMA
      // 一个 warp 负责当前 CTA 的 N64；将它再切成 4 个 N16 chunk。
      // 每个 ax1_0 对应一个逻辑 B[K16,N16] fragment，稍后由两条 N8 MMA 消费。
      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        // Load B fragment
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[compute_buf][(((k_0_1 * 2176) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 136) + ((((int)threadIdx.x) >> 4) * 8))))
          );

          // B_shared 的逻辑形状是 [K32,N128]，物理行跨度为 136：
          //   k_0_1*2176 = k_0_1*(16*136)，选择 K16 子块；
          //   threadIdx.y*64，选择两个 warp 各自负责的 N64；
          //   ax1_0*16，选择该 warp 内的一个 N16；
          //   (lane&15)*136 与 (lane>>4)*8，提供 ldmatrix 所需的行地址。
          //
          // .trans 在 shared->register 搬运时按转置 fragment 规则分发数据，
          // 使逻辑上按 [K,N] 存放的 B 满足 mma 的 col-major B 操作数布局。
          // .b16 只表示每个元素占 16 bit；这些 bit 已经是反量化后的 FP16。
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[0]), 
              "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[1]), 
              "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[2]), 
              "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[3])
            : "r"(addr)
          );
        }
      }
      
      // Execute 8 MMA instructions (4 N chunks × 2 for each N=16)
      // j_0_4 选择当前 warp 的第 j 个 N16 输出块：
      //   j=0,1,2,3 -> N[0:16], N[16:32], N[32:48], N[48:64]。
      // 每个 N16 再由两条 m16n8k16 指令覆盖左、右两个 N8。
      for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
        {
          // 第一条 MMA 计算当前 N16 的左半边：
          //   D[16,8] = A[16,16] * B[16,8] + C[16,8]
          //
          // 指令后缀 f32.f16.f16.f32 依次表示 D、A、B、C 的类型：
          // A/B 寄存器中的每 16 bit 按 FP16 相乘，结果累加到 FP32 C_warp。
          // 虽然 inline-asm 对 A/B 使用 "r" 约束，它只指定 32-bit 寄存器类别；
          // 真正的数据解释由 mma 指令中的 .f16 决定。
          // 整个 warp 共同产生 16x8 个输出；当前 lane 持有其中 4 个 FP32。
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp[0] + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[3]), 
              "r"(((unsigned *)(B_shared_warp[0] + (j_0_4 * 8)))[0]), 
              "r"(((unsigned *)(B_shared_warp[0] + (j_0_4 * 8)))[1]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }
        {
          // 第二条 MMA 计算当前 N16 的右半边。偏移 +4 的单位取决于数组类型：
          //   B_shared_warp 是 half[]，+4 half 跳过 2 个 packed b32，选择后 N8；
          //   C_warp 是 float[]，+4 float 选择该 N16 的后 4 个 lane-local 累加器。
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp[0] + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[3]), 
              "r"(((unsigned *)(B_shared_warp[0] + ((j_0_4 * 8) + 4)))[0]), 
              "r"(((unsigned *)(B_shared_warp[0] + ((j_0_4 * 8) + 4)))[1]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
      }
    }  // End k_0_1 loop for current tile

    // ===== Load next tile (into load_buf) while computing =====
    // This happens in parallel with the MMA computation above (software pipelining)
    // 更准确地说，本实现先完成上面的 MMA，再越过此 barrier 开始加载；该 barrier
    // 阻止任何线程过早进入下一阶段，并确保随后复用 ping-pong buffer 时没有读写冲突。
    __syncthreads();  // Wait for computation to finish using compute_buf
    
    // Load A for next iteration
    // 每个有效线程从 global A 搬 16 bytes = 8 个连续 FP16 到 load_buf。
    // k_0_0_load*32 定位下一 K32 tile；超出 M 的逻辑行用 0 填充，保证边界
    // tile 参与 MMA 时不会污染输出。
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr[load_buf]) = *(uint4*)(A_ptr + (k_0_0_load * 32));
    } else {
      *(uint4*)(A_shared_ptr[load_buf]) = make_uint4(0, 0, 0, 0);
    }

    // Load scales/zeros for next iteration
    // 下一 tile 的单线程映射与 Prologue 完全相同，只把 tile 编号 k_0_0
    // 换成 k_0_0_load：
    //   group_next     = floor((k_0_0_load*32)/G)
    //   k_global0_next = k_0_0_load*32 + threadIdx.y*2 + threadIdx.x/16
    //   n0              = n_tile*128 + (threadIdx.x%16)*8
    //
    //                        同一个 N[n0:n0+8]
    //                    +--------------------------+
    // k_global0_next+ 0  | 8 weights, loop 0        |
    // k_global0_next+ 4  | 8 weights, loop 1        |
    //         ...        |           ...            |
    // k_global0_next+28  | 8 weights, loop 7        |
    //                    +--------------------------+
    //                         ^
    //                         | 广播同一组 8-wide zero/scale
    //
    // zero 的一个 uint32 打包 8 个 4-bit 值，scale 的一个 uint4 装 8 个
    // FP16。当相邻 K32 tile 属于同一 group 时，它们会从同一 group row
    // 重新读取数值相同的量化参数。
    uint32_t zeros_loaded_next = *(uint32_t*)(zeros_ptr + k_0_0_load * 32 / G * (OC / 8));

    // zero point 也先展开成四个 packed half2，随后与对应的 8 个 weight lane
    // 一一执行 (q-zero)*scale。
    uint4 B_loaded_zero_next = dequantize_s4_to_fp16x2(zeros_loaded_next);
    uint4 B_loaded_scale_next = *(uint4*)(scaling_factors_ptr + k_0_0_load * 32 / G * (OC));

    // qweight 的行跨度为 OC/8 个 uint32；每个 uint32 对应 8 个输出通道。
    int* B_ptr_local_next = B_ptr + k_0_0_load * 32 * (OC / 8);

    // Load and dequantize B for next iteration
    // 每个线程循环 8 次、每次跨 row_stride=4 个 K row。64 个线程共同覆盖
    // 下一块 [K32,N128] 的 packed weight，并直接把它物化到 shared memory，
    // 而不是在 global memory 中生成一份完整的 FP16 权重矩阵。
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {
      // 32-bit global load：得到当前线程负责的 8 个 4-bit weight。
      uint32_t B_loaded = *(uint32_t*)(B_ptr_local_next + ax0_ax1_fused_0 * row_stride * (OC / 8));

      // 快速展开为 uint4 容器中的 8 个 FP16；x/y/z/w 各装一个 half2。
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      
      // 四组 packed-half 运算完成。对 i=ax0_ax1_fused_0、j=0..7：
      //   k = k_global0_next + 4*i
      //   n = n0 + j
      //   B_deq[k,n] = (B_q4[k,n] - zero[group_next,n]) * scale[group_next,n]
      // 每条 sub/fma 同时处理两个 FP16，ZERO 的 bit pattern 是 half2{0,0}。
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_next.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_next.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_next.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_next.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_next.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_next.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_next.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_next.w), "r"(ZERO));
      
      // 128-bit vector store 一次写入 8 个反量化 FP16。136 的物理行跨度包含
      // 8 个 half padding，服务于后续 ldmatrix 的对齐/bank 访问布局。
      *(uint4*)(B_shared_ptr[load_buf] + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
    }
    
    // 发布刚写入 load_buf 的 A/B tile：保证整个 CTA 的所有写入完成后，下一轮
    // 才把该 buffer 作为 compute_buf 交给 ldmatrix 读取。
    __syncthreads();  // Wait for next tile loading to complete before next iteration
  }  // End mainloop

  // ========================================================================
  // EPILOGUE: Process last tile
  // ========================================================================
  // Mainloop 最后一轮已经把 tile k_bound-1 装入对应的 shared buffer，但没有
  // 消费它；这里复用同样的 ldmatrix + MMA 路径完成最后一个 K32 tile。
  // 这不是 GEMM 的输出 epilogue（类型转换/写回），而是软件流水线的 drain 阶段。
  {
    // tile 编号的奇偶性决定它最终停留在哪一个 ping-pong buffer 中。
    int compute_buf = (k_bound - 1) % 2;
    
    // 最后一个 K32 仍拆成两个 K16 fragment，分别累加进已有的 C_warp。
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      // Load A fragment
      {
        unsigned int addr;
        __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(addr)
          : "l"((void *)((&(A_shared[compute_buf][(k_0_1 * 16)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
        );

        // 与 mainloop 相同：warp 用 ldmatrix.x4 从带 40-half 行跨度的 shared A
        // 收集四个 8x8 块，形成一个 row-major A[16,16] MMA fragment。
        __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
          "{%0, %1, %2, %3}, [%4];\n"
          : "=r"(((unsigned *)(A_shared_warp[0] + 0))[0]), 
            "=r"(((unsigned *)(A_shared_warp[0] + 0))[1]), 
            "=r"(((unsigned *)(A_shared_warp[0] + 0))[2]), 
            "=r"(((unsigned *)(A_shared_warp[0] + 0))[3])
          : "r"(addr)
        );
      }

      // Load B fragments and compute
      // 每个 warp 的 N64 被拆为四个 N16；每个 N16 的 B fragment 由
      // ldmatrix.x4.trans 形成，随后供两条 m16n8k16 MMA 使用。
      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[compute_buf][(((k_0_1 * 2176) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 136) + ((((int)threadIdx.x) >> 4) * 8))))
          );

          // .trans 将 shared 中逻辑 [K16,N16] 的 B 数据分发成 mma.row.col
          // 要求的 column-major register fragment；它不改变 FP16 数值。
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[0]), 
              "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[1]), 
              "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[2]), 
              "=r"(((unsigned *)(B_shared_warp[0] + (ax1_0 * 8)))[3])
            : "r"(addr)
          );
        }
      }
      
      // MMA
      // 每轮发射 4*2=8 条 MMA：覆盖当前 warp 的 [M16,N64,K16] 工作量。
      // 两个 k_0_1 迭代完成最后一块 [M16,N64,K32]，并在 FP32 中继续累加。
      for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
        {
          // 当前 N16 的左 N8：C_left = A*B_left + C_left。
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
               "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp[0] + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[3]), 
              "r"(((unsigned *)(B_shared_warp[0] + (j_0_4 * 8)))[0]), 
              "r"(((unsigned *)(B_shared_warp[0] + (j_0_4 * 8)))[1]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), 
              "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }
        {
          // 当前 N16 的右 N8：B 的 half 偏移 +4、C 的 float 偏移 +4。
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
               "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp[0] + 0))[0]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[1]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[2]), 
              "r"(((unsigned *)(A_shared_warp[0] + 0))[3]), 
              "r"(((unsigned *)(B_shared_warp[0] + ((j_0_4 * 8) + 4)))[0]), 
              "r"(((unsigned *)(B_shared_warp[0] + ((j_0_4 * 8) + 4)))[1]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), 
              "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
      }
    }
  }  // End epilogue

  // ========== Write results back ==========
  // 此时每个 warp 的 32 个 lane 合起来持有一个 [M16,N64] FP32 accumulator
  // fragment；每个 lane 在 C_warp 中有 32 个 float（4 个 N16 * 每块 8 个）。
  // ax1_0_1 选择四个 N16，local_id 选择当前 lane 在该 N16 中的 8 个结果。
  for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      // mma.m16n8 的 accumulator fragment 不是按线程连续行存放：
      //   lane/4                 给出基础 row 0..7；
      //   (local_id%4)/2 * 8     在上、下两个 M8 半块之间选择；
      // 因而一个 lane 会写 row r 和 row r+8。blockIdx_y/j_factors1 再加上
      // 当前 CTA 的 M16 全局起点。
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 + ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;

      // M 不一定是 16 的倍数；只让有效逻辑行执行 global store。
      if (row_offset < M) {
        // 列坐标由以下部分组成：
        //   C_ptr                   ：CTA 的 N128 起点 + warp 的 N64 起点
        //   ax1_0_1*16              ：当前 N16 chunk
        //   (local_id/4)*8          ：该 chunk 的左/右 N8
        //   local_id%2              ：N8 内当前 lane 持有的两个相邻列
        // MMA 一直使用 FP32 累加，这里才把单个结果舍入为 FP16 写入输出。
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
}


// ========== PyTorch Interface ==========
// Python/C++ 扩展入口：检查该专用 kernel 的 shape 前提，启动一个
// m16n128k32 CTA kernel，并在 split-K 维度上归约各 block 的部分和。
torch::Tensor awq_gemm_db(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters)
{
    // A 的逻辑 shape 为 [M,IC]；packed B 的逻辑 shape 为 [IC,OC/8]。
    // 每个 B int32 打包 8 个 4-bit 权重，所以真实输出通道数要乘 8。
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);

    // 确保 kernel launch 和 tensor allocation 都发生在输入 A 所在的 CUDA device。
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    // 每个 split-K block 先写一份 [M,OC] 部分和，稍后沿第 0 维求和。
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    // scaling_factors 的第 0 维是 group 数，因此 IC / num_groups 得到量化组大小 G。
    int group_size = num_in_channels / _scaling_factors.size(0);

    // 这些限制来自当前硬编码 tile/layout：每个 CTA 处理 N128，packed INT4
    // 以 8 个通道为一组；group 边界必须与 K32 tile 对齐。
    if (num_out_channels % 64 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 64");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    if (group_size % 32 != 0)
        throw std::invalid_argument("Group size should be a multiple of 32");
    if (num_out_channels % group_size != 0)
        throw std::invalid_argument("OC is not multiple of Group size");

    if (num_out_channels % 128 == 0)
    {
        // 一个线性 blockIdx.x 同时编码：M16 tile、N128 tile、split-K slice。
        int j_factors1 = num_out_channels / 128 / 1;
        dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);

        // 64 threads = 2 warps；两个 warp 各负责 CTA 输出 N128 中的一段 N64。
        dim3 threads_per_block(32, 2);
        
        gemm_forward_4bit_cuda_m16n128k32_db<<<num_blocks, threads_per_block>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, 
            num_in_feats, num_in_channels, num_out_channels, out_feats);
    }
    else
    {
      throw std::invalid_argument("OC is not multiple of 128");
    }
    
    // 每个 split-K slice 只累加自己负责的 K32 tiles；这里把这些 FP16 部分和
    // 沿 split-K 维归约，得到最终 [M,OC] 输出。
    return _out_feats.sum(0);
}
