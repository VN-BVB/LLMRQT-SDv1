#include "sq_gemm.h"
#include <ATen/ATen.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include <iostream>

template <torch::Dtype dtype>
struct ElementOutputType {
    // 默认报错（如果传入非 FP16/BF16 类型）
    static_assert(dtype == torch::kFloat16 || dtype == torch::kBFloat16, "Unsupported dtype");
};

template <>
struct ElementOutputType<torch::kFloat16> {
    using type = cutlass::half_t;
};

template <>
struct ElementOutputType<torch::kBFloat16> {
    using type = cutlass::bfloat16_t;
};

// A[M,K]xB[K,N], A rowmajor B colmajor
torch::Tensor w8a8_int8_linear_bbf16_obf16_per_tensor(torch::Tensor input,  // INT8
                                                    torch::Tensor weight, // INT8
                                                    torch::Tensor bias,   // BF16
                                                    float alpha,          // BF16
                                                    float beta            // BF16
) {
  auto M = input.size(0);
  auto N = weight.size(1); //注意qweight的shape是否转置，这里不要写错了，不然会报illegal memory acess
  auto K = input.size(1);

  using ElementOutput = cutlass::half_t;//cutlass::bfloat16_t;//
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // works on SM80 and SM75
  //   using Gemm = cutlass::gemm::device::Gemm<
  //       int8_t, cutlass::layout::RowMajor,
  //       int8_t, cutlass::layout::ColumnMajor,
  //       ElementOutput, cutlass::layout::RowMajor,
  //       ElementAccumulator,
  //       cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  //       cutlass::gemm::GemmShape<256, 128, 64>, //block tile
  //       cutlass::gemm::GemmShape<64, 64, 64>,  // warp tile
  //       cutlass::gemm::GemmShape<16, 8, 32>, // mma tile
  //       cutlass::epilogue::thread::LinearCombination<
  //           ElementOutput, 8,//128 / cutlass::sizeof_bits<ElementOutput>::value, 输出格式类型
  //           ElementAccumulator,  // 累加器类型
  //           ElementComputeEpilogue>, //乘法精度类型float
  //       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
  using Gemm = cutlass::gemm::device::Gemm<
      // ==================== A matrix ====================
      int8_t,                    // ElementA：A 矩阵元素类型，int8
      cutlass::layout::RowMajor, // LayoutA：A 按 RowMajor 存储 A 的逻辑 shape = [M, K]
      // ==================== B matrix ====================
      int8_t,                       // ElementB：B 矩阵元素类型，int8
      cutlass::layout::ColumnMajor, // LayoutB：B 按 ColumnMajor 存储、B 的逻辑 shape = [K, N]
      // ==================== C / D matrix ====================
      ElementOutput,             // ElementC / ElementD：最终输出元素类型
      cutlass::layout::RowMajor, // LayoutC：输出矩阵 C/D 按 RowMajor 存储、 C/D 的逻辑 shape = [M, N]
      // ==================== accumulator =====================
      ElementAccumulator, // MMA 累加器类型// int8 × int8 通常使用 int32 accumulator
      // ==================== Tensor Core configuration ====================
      cutlass::arch::OpClassTensorOp, // OperatorClass：使用 Tensor Core，而不是普通 CUDA Core
      cutlass::arch::Sm80,            // ArchTag：目标 GPU 架构为 SM80（Ampere）
      // ==================== Block tile ====================
      cutlass::gemm::GemmShape<256, 128, 64>,
      // BlockShape = <M, N, K>、一个 thread block 一次负责： C tile = [256, 128]
      // mainloop 每次沿 K 方向处理 64：A tile = [256, 64]、 B tile = [64, 128]、
      // 即：[256,64] × [64,128] = [256,128]
      // ==================== Warp tile ====================
      cutlass::gemm::GemmShape<64, 64, 64>,
      // WarpShape = <M, N, K>
      // 每个 warp 负责： C warp tile = [64, 64]、 A warp tile = [64,64]、B warp tile = [64,64]
      // 一个 block 的 M/N 方向 warp 数：M：256 / 64 = 4、 N：128 / 64 = 2
      // 所以一个 block 共： 4 × 2 = 8 warps = 8 × 32 = 256 threads
      // ==================== Tensor Core MMA instruction tile ====================
      cutlass::gemm::GemmShape<16, 8, 32>,
      // InstructionShape = <M, N, K>
      // 底层 Tensor Core MMA tile：
      // A mma tile = [16,32] 、B mma tile = [32,8]、 C mma tile = [16,8]
      // 对应 int8 Tensor Core 的 m16n8k32 级别 MMA
      // 一个 WarpShape = [64,64,64]
      // 需要的 MMA tile 数： M：64 / 16 = 4、 N：64 / 8  = 8、K：64 / 32 = 2
      // 总 MMA 数： 4 × 8 × 2 = 64 个 instruction tile / warp-K64
      // ==================== Epilogue ====================
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, // 输出 D 的元素类型
          8,                  // ElementsPerAccess：
                              // 每次 epilogue 向量化访问多少个 ElementOutput
                              // 是“8 个元素”, 这里是half8，一般按照128bit除

          ElementAccumulator, // Accumulator 类型：
                              // Tensor Core mainloop 累加结果的类型
                              // int8 GEMM 通常为 int32

          ElementComputeEpilogue // Epilogue 中 alpha/beta 等计算使用的类型
                                 // 例如 float
                                 // 一般执行：
                                 // D = alpha * Accumulator + beta * C
                                 // 在本算子中 为 （input_scale * output_scale） * int32 + [bias_scale * bias(int32)]
          >,

      // ==================== Threadblock swizzle ====================
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      // ThreadblockSwizzle：
      // 决定 blockIdx 如何映射到 GEMM 的
      // [M,N] threadblock tile 这里是有关不按照内存存储顺序如何多次命中L2cache的
      // Identity = 基本的直接映射策略

      // ==================== Pipeline stages ====================
      3 // Stages = 3
        // mainloop 使用 3-stage pipeline
        // 用多 stage shared-memory buffer
        // 重叠 global→shared 搬运和 MMA 计算
      >;
  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  // bias shape: [OC] -> [1, OC] -> [M, OC]
  auto out = bias.to(device).view({1, -1}).repeat({M, 1}); 
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  // pytorch tensor不认识cutlass::bfloat16_t，所以这里out(torch bf16)需要先转void再强转cutlass::bfloat16_t
  // 这个问题解决了链接问题ImportError: /usr/local/lib/python3.10/dist-packages/runtime-0.1-py3.10-linux-x86_64.egg/runtime/sq_fp8_kernels.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIN7cutlass10bfloat16_tEEEPT_v
  void* out_ptr = out.data_ptr(); //  ElementOutput* out_data = out.data_ptr<ElementOutput>()
  ElementOutput* out_data = static_cast<ElementOutput*>(out_ptr);

  // input_ref：把 PyTorch 的 input tensor 包装成 CUTLASS TensorRef
  // TensorRef = 数据起始指针 + layout/stride 信息
  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), // input 在 GPU 上的首地址，
                                       // 并按 ElementInputA* 类型解释
                                       // 例如 ElementInputA = int8_t，
                                       // 那么这里就是 int8_t*

      LayoutInputA::packed(input_size) // 根据 input_size 自动生成“紧密连续”的 layout
                                       // input_size 通常表示矩阵逻辑 shape，例如 [M, K]
                                       //
                                       // 如果 LayoutInputA = RowMajor：
                                       // stride = K
                                       // 地址 = base + row * K + col
  );
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out_data, LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication这个是真正大矩阵的 MNK。
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device D=A*B+C
                    //因为允许 in-place epilogue。
      {alpha, beta}, 
      1};           // split_k_slices,沿着k维切成几分，如果是1，最后没有k_reduce的操作
  Gemm gemm_op;

  // ==================== 1. 根据 arguments 计算 workspace 大小 ====================
  //
  // workspace 主要用于某些需要额外临时存储的情况，例如：
  //   - Serial Split-K 的 semaphore / synchronization metadata
  //   - 某些特殊 GEMM kernel 的辅助临时空间
  //
  // 如果当前：
  //   split_k_slices = 1
  // 并且 kernel 本身不需要额外 workspace，
  // 那么这里通常会返回：
  //
  //   workspace_size = 0
  //
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // ==================== 2. 分配 device workspace ====================
  //
  // 在 GPU 上申请 workspace_size 字节的临时显存。
  //
  // 类型写成 uint8_t，是因为这里本质上只需要“一块原始字节空间”，
  // CUTLASS 内部会根据自己的需求解释这块内存。
  //
  // 如果：
  //   workspace_size == 0
  //
  // 那么基本不会实际占用额外显存。
  //
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // ==================== 3. 检查当前参数是否能被该 Gemm kernel 实现 ====================
  //
  // can_implement() 不真正执行 GEMM，
  // 只是根据 arguments 做合法性 / 硬件约束检查。
  //
  // 常见检查包括：
  //   1. M / N / K 是否满足某些对齐要求
  //   2. A / B / C / D 的地址是否满足 alignment 要求
  //   3. leading dimension / stride 是否合法
  //   4. Tensor Core 指令对 K、数据类型等的约束是否满足
  //   5. split-K 等配置是否合法
  //
  // 例如 int8 Tensor Core kernel 经常对：
  //   K 的倍数
  //   pointer alignment
  //   elements-per-access
  // 有要求。
  //
  cutlass::Status status = gemm_op.can_implement(arguments);

  if (status != cutlass::Status::kSuccess)
  {
      // 当前 problem_size / layout / alignment 等条件
      // 不满足这个 CUTLASS Gemm kernel 的要求
      throw std::runtime_error("cutlass cannot implement");
  }

  // ==================== 4. 初始化 Gemm operator ====================
  //
  // initialize() 会把 arguments 中的运行时信息整理成
  // kernel 真正需要使用的 Params，并保存在 gemm_op 内部。
  //
  // arguments 中包含：
  //   problem_size        -> 整个 GEMM 的 M/N/K
  //   input_ref           -> A 的 pointer + layout
  //   weight_ref          -> B 的 pointer + layout
  //   out_ref             -> C / D 的 pointer + layout
  //   {alpha, beta}       -> epilogue 参数
  //   split_k_slices      -> Split-K 配置
  //
  // workspace.get()：
  //   返回刚才申请的 workspace 的 GPU pointer。
  //
  // initialize() 本身通常不是在执行完整 GEMM，
  // 它主要完成：
  //   arguments
  //       ↓
  //   内部 Params 初始化
  //       ↓
  //   为后面的真正 kernel launch 做准备
  //
  status = gemm_op.initialize(
      arguments,
      workspace.get());

  if (status != cutlass::Status::kSuccess)
  {
      // arguments 本身可能可实现，
      // 但初始化 kernel 参数 / workspace 时失败
      throw std::runtime_error("cutlass cannot initialize");
  }

  // ==================== 5. 真正执行 CUTLASS GEMM ====================
  //
  // gemm_op() 会根据 initialize() 已经保存好的内部 Params，
  // launch 对应的 CUDA GEMM kernel。
  //
  // 真正发生：
  //
  //   Global Memory
  //       ↓
  //   A / B tile load
  //       ↓
  //   Shared Memory
  //       ↓
  //   ldmatrix / register fragment
  //       ↓
  //   Tensor Core MMA
  //       ↓
  //   accumulator
  //       ↓
  //   Epilogue:
  //       D = alpha * Acc + beta * C
  //       ↓
  //   写回 global memory
  //
  // 就是在这里真正开始 GPU GEMM 计算。
  //
  status = gemm_op();

  if (status != cutlass::Status::kSuccess)
  {
      // kernel launch / execution 状态异常
      throw std::runtime_error("cutlass cannot run");
  }

  // ==================== 6. 返回 PyTorch output Tensor ====================
  //
  // out 对应前面 out_ref 所指向的那块显存。
  // CUTLASS GEMM 已经把最终结果写入 out.data_ptr()。
  //
  // 所以这里只是把原来的 PyTorch Tensor 返回，
  // 并没有再次复制数据。
  //
  return out;
}

//   return out;
// }
