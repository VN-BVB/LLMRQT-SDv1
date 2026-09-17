// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include "sm89_fp8_gemm.h"

#include <cute/tensor.hpp>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/version.h>

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <optional>
// #include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
// Two warninngs in Cutlass included header files
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wset-but-not-used")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-parameter")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wmissing-field-initializers")

// Determine if the architecture supports rowwise scaled mm
// Currently failing on windows with:
// https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000 && defined(CUDA_ARCH) && CUDA_ARCH >= 890

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)

template <typename T>
struct get_torch_DtypeOutput {
    static_assert(sizeof(T) == 0, "Unsupported type for torch::Dtype mapping");
};

// 特化 float -> torch::kFloat32
template <>
struct get_torch_DtypeOutput<float> {
    static constexpr torch::Dtype value = torch::kFloat32;
};

// 特化 cutlass::half_t -> torch::kFloat16
template <>
struct get_torch_DtypeOutput<cutlass::half_t> {
    static constexpr torch::Dtype value = torch::kFloat16;
};

// 特化 cutlass::bfloat16_t -> torch::kBFloat16
template <>
struct get_torch_DtypeOutput<cutlass::bfloat16_t> {
    static constexpr torch::Dtype value = torch::kBFloat16;
};

// common utils
using DtypeScale = float;
using DtypeAccum = float;
using DtypeEpilogue = float;
using DtypeOutput = cutlass::bfloat16_t;
// using torch_DtypeOutput = get_torch_DtypeOutput<DtypeOutput>::value;
constexpr torch::Dtype torch_DtypeOutput = get_torch_DtypeOutput<DtypeOutput>::value;
// for SM90
// using Multiply = cutlass::epilogue::fusion::Sm90Compute<
//     cutlass::multiplies,
//     DtypeEpilogue,
//     DtypeEpilogue,
//     cutlass::FloatRoundStyle::round_to_nearest>;

// using Add = cutlass::epilogue::fusion::Sm90Compute<
//     cutlass::plus,
//     DtypeEpilogue,
//     DtypeEpilogue,
//     cutlass::FloatRoundStyle::round_to_nearest>;

// using Cast = cutlass::epilogue::fusion::Sm90Compute<
//     cutlass::epilogue::thread::Identity,
//     DtypeOutput,
//     DtypeEpilogue,
//     cutlass::FloatRoundStyle::round_to_nearest>;

// template <bool LargeTile, bool FastAccum>
// struct Schedule;

// template <>
// struct Schedule</*LargeTile=*/false, /*FastAccum=*/false> {
//   using type = cutlass::gemm::KernelTmaWarpSpecialized;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
// };

// template <>
// struct Schedule</*LargeTile=*/true, /*FastAccum=*/false> {
//   // For a 128x128x128 tile with fastAccum = false, using
//   // pingpong schedule will lead to spilling, and WarpSpecialized w/o pingpong
//   // is slow
//   using type = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecializedCooperative;
// };

// template <>
// struct Schedule</*LargeTile=*/false, /*FastAccum=*/true> {
//   using type = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
// };

// template <>
// struct Schedule</*LargeTile=*/true, /*FastAccum=*/true> {
//   using type = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
// };

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

// // Cutlass rowwise kernel for SM89
// template <
//     typename ThreadblockShape,
//     typename WarpShape,
//     int NumStages,
//     typename FastAccum,
//     typename DtypeA,
//     typename DtypeB,
//     typename DtypeBias>
// torch::Tensor f8f8bf16_rowwise_impl_sm89(
//     torch::Tensor XQ, // FP8 row major (M,K)
//     torch::Tensor WQ, // FP8 col major (K,N)
//     torch::Tensor x_scale, // FP32
//     torch::Tensor w_scale, // FP32
//     std::optional<torch::Tensor> bias // BF16
// ){
//     // at::Tensor XQ, // FP8 row major (M,K)
//     // at::Tensor WQ, // FP8 col major (K,N)
//     // at::Tensor x_scale,
//     // at::Tensor w_scale,
//     // std::optional<at::Tensor> bias,
//     // at::Tensor out) {
//   int M = XQ.size(0);
//   int N = WQ.size(1);
//   int K = XQ.size(1);

//   using LayoutInputA = cutlass::layout::RowMajor;
//   constexpr int AlignmentInputA = 16 / sizeof(DtypeA); // 16bytes / 1= 16因为最多可向量化 16 个 FP8。

//   using LayoutInputB = cutlass::layout::ColumnMajor;
//   constexpr int AlignmentInputB = 16 / sizeof(DtypeB);

//   using LayoutOutput = cutlass::layout::RowMajor;
//   constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

//   // Tag indicating the minimum SM that supports the intended feature
//   using ArchTag = cutlass::arch::Sm89;
//   using OperatorClass = cutlass::arch::OpClassTensorOp;

//   using ThreadblockSwizzle =
//       cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

//   using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

//   using Operator = std::conditional_t<
//       FastAccum::value,
//       cutlass::arch::OpMultiplyAddFastAccum, // [M,:] * [:,N] ++ tensorcore 一次性加
//       cutlass::arch::OpMultiplyAdd>; // [M,:4] * [4:,N] +[M,:4] * [4:,N] ， cuda core 加
//   constexpr auto NumEVTEpilogueStages = 1;

//   using OutputTileThreadMap =
//       cutlass::epilogue::threadblock::OutputTileThreadLayout<
//           ThreadblockShape,
//           WarpShape,
//           DtypeOutput,
//           AlignmentOutput,
//           NumEVTEpilogueStages>;

//   using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

//   using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
//       OutputTileThreadMap, DtypeScale,
//       cute::Stride<cute::_1, cute::_0, int64_t>>;
//   using XScaleArguments = typename XScale::Arguments;

//   using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
//       OutputTileThreadMap, DtypeScale,
//       cute::Stride<cute::_0, cute::_1, int64_t>/*StrideMNL*/>;
//   using WScaleArguments = typename WScale::Arguments;

//   using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
//       OutputTileThreadMap, DtypeBias,
//       cute::Stride<cute::_0, cute::_1, int64_t>>;
//   using BiasArguments = typename Bias::Arguments;

//   using ApplyXScale = cutlass::epilogue::threadblock::VisitorCompute<
//       cutlass::multiplies, DtypeEpilogue, DtypeEpilogue,
//       cutlass::FloatRoundStyle::round_to_nearest
//   >;
//   using EVTApplyXScale = cutlass::epilogue::threadblock::Sm80EVT<
//       ApplyXScale,// NodeOp即根节点
//       Accum,//childOp
//       XScale>;//childOp

//   using ApplyWScale = cutlass::epilogue::threadblock::VisitorCompute<
//       cutlass::multiplies, DtypeEpilogue/*elementOut*/, DtypeEpilogue/*elementCompute*/,
//       cutlass::FloatRoundStyle::round_to_nearest
//   >;
//   using EVTApplyWScale = cutlass::epilogue::threadblock::Sm80EVT<
//       ApplyWScale, // NodeOp
//       EVTApplyXScale, //childOp
//       WScale>;//childOp

//   using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
//       cutlass::plus, DtypeEpilogue, DtypeEpilogue,
//       cutlass::FloatRoundStyle::round_to_nearest
//   >;
//   using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<
//       ApplyBias,
//       EVTApplyWScale,
//       Bias>;

//   using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
//       OutputTileThreadMap, DtypeOutput,
//       cutlass::FloatRoundStyle::round_to_nearest,
//       cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
//   >;

//   using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
//       Output,
//       EVTApplyBias>;

//   using EVTKernel = // at::cuda::detail::enable_2x_kernel_for_sm89< ATen里面没找到这个函数，干脆注释掉得了
//       typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
//           DtypeA, LayoutInputA, cutlass::ComplexTransform::kNone, AlignmentInputA,
//           DtypeB, LayoutInputB, cutlass::ComplexTransform::kNone, AlignmentInputB,
//           DtypeOutput, LayoutOutput, AlignmentOutput,
//           DtypeAccum,
//           DtypeEpilogue,
//           OperatorClass,
//           ArchTag,
//           ThreadblockShape,
//           WarpShape,
//           InstructionShape,
//           EVTOutput,
//           ThreadblockSwizzle,
//           NumStages,
//           Operator,
//           NumEVTEpilogueStages>::GemmKernel;

//   using Gemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

//   cutlass::gemm::GemmCoord problem_size(M, N, K);
//   constexpr auto SplitKFactor = 1;

//   XScaleArguments x_scale_arguments{
//       (DtypeScale*)x_scale.data_ptr(),
//       DtypeScale(1),
//       {cute::_1{}, cute::_0{}, problem_size.m()}
//   };
//   WScaleArguments w_scale_arguments{
//       (DtypeScale*)w_scale.data_ptr(),
//       DtypeScale(1),
//       {cute::_0{}, cute::_1{}, problem_size.n()}
//   };
//   BiasArguments bias_arguments{
//       bias.has_value() ? reinterpret_cast<DtypeBias*>(bias->data_ptr()) : nullptr,
//       DtypeBias(0),
//       {cute::_0{}, cute::_1{}, problem_size.n()}
//   };

//   auto out = torch::empty({problem_size.m(), problem_size.n()},
//                           torch::dtype(torch_DtypeOutput).device(XQ.device()) );

//   typename Output::Arguments output_arguments{
//     (DtypeOutput*)out.data_ptr(),
//     {problem_size.n(), cute::_1{}, problem_size.mn().product()}
//   };
//   typename EVTOutput::Arguments callback_arguments{
//     {
//       {
//         {
//           {},                 // Accum
//           x_scale_arguments,  // XScale
//           {}                  // ApplyXScale
//         },                    // EVTApplyXScale
//         w_scale_arguments,    // WScale
//         {}                    // ApplyWScale
//       },                      // EVTApplyWScale
//       bias_arguments,         // Bias
//       {}                      // ApplyBias
//     },                        // EVTApplyBias
//     output_arguments          // Output
//   };                          // EVTOutput

//   typename Gemm::Arguments arguments(
//     cutlass::gemm::GemmUniversalMode::kGemm,
//     problem_size,
//     SplitKFactor,
//     callback_arguments,           // arguments of EVT callbacks
//     (DtypeA*)XQ.data_ptr(),
//     (DtypeB*)WQ.data_ptr(),
//     nullptr,                      // ptr C (unused)
//     nullptr,                      // ptr D (unused)
//     problem_size.mk().product(),  // batch stride A
//     problem_size.nk().product(),  // batch stride B
//     0,                            // batch stride C (unused)
//     0,                            // batch stride D (unused)
//     problem_size.k(),             // stride A
//     problem_size.k(),             // stride B
//     0,                            // stride C (unused)
//     0);                           // stride D (unused)

//   Gemm gemm;

//   // Using the arguments, query for extra workspace required for matrix
//   // multiplication computation
//   size_t workspace_size = Gemm::get_workspace_size(arguments);

//   // Allocate workspace memory
//   auto workspace = XQ.new_empty(
//       {static_cast<int64_t>(workspace_size)},
//       at::TensorOptions().dtype(at::kByte));

//   // Check the problem size is supported or not
//   cutlass::Status status = gemm.can_implement(arguments);
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot implement");
//   }

//   // Initialize CUTLASS kernel with arguments and workspace pointer
//   status = gemm.initialize(arguments, workspace.data_ptr());
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot initialize");
//   }

//   status = gemm(at::cuda::getCurrentCUDAStream());
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error(
//         std::string("cutlass cannot run") +
//         cutlass::cutlassGetStatusString(status));
//   }
//   C10_CUDA_KERNEL_LAUNCH_CHECK();
//   return out;
// }
// ============================================================================
// CUTLASS FP8 x FP8 -> BF16 Rowwise GEMM Kernel for SM89
//
// 整个 kernel 实现的数学公式：
//
//   Acc[m,n] = sum_k XQ[m,k] * WQ[k,n]
//
//   Out[m,n]
//       = Acc[m,n]
//       * x_scale[m]        // activation 的 per-token / per-row scale
//       * w_scale[n]        // weight 的 per-channel / per-column scale
//       + bias[n]           // output-channel bias
//
// 整体可以划分成两个阶段：
//
//   1. Mainloop
//
//      FP8 Tensor Core MMA：
//
//          XQ[M,K] x WQ[K,N]
//                   |
//                   v
//              Acc[M,N]
//
//      核心 instruction shape：m16n8k32
//
//
//
//   2. Epilogue
//
//                       Acc[m,n]
//                           |
//                           *
//                         /   \
//                       Acc  x_scale[m]
//                           |
//                           *
//                         /   \
//                   previous w_scale[n]
//                           |
//                           +
//                         /   \
//                   previous bias[n]
//                           |
//                           v
//                       Out[m,n]
//
//
//   EVT 最终描述的公式：
//
//      Out[m,n]
//          = ((Acc[m,n] * x_scale[m])
//                        * w_scale[n])
//                        + bias[n]
//
// ============================================================================

template <
    typename ThreadblockShape,
    typename WarpShape,
    int NumStages,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias>
torch::Tensor f8f8bf16_rowwise_impl_sm89(
    torch::Tensor XQ,                 // FP8 activation，逻辑 shape = [M, K]，RowMajor
    torch::Tensor WQ,                 // FP8 weight，逻辑 shape = [K, N]，ColumnMajor
    torch::Tensor x_scale,            // FP32，shape = [M]，activation per-token/per-row scale
    torch::Tensor w_scale,            // FP32，shape = [N]，weight per-channel scale
    std::optional<torch::Tensor> bias // BF16，shape = [N]，可选
) {

  // ==========================================================================
  // 1. 获取 GEMM problem size
  //
  // XQ : [M, K]
  // WQ : [K, N]
  //
  // Out:
  //
  //     [M,K] x [K,N] = [M,N]
  //
  // ==========================================================================

  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);


  // ==========================================================================
  // 2. 定义矩阵 A（XQ）的 layout 和 alignment
  // ==========================================================================

  // A = XQ[M,K]
  //
  // RowMajor 表示：
  //
  //   XQ[m,k]
  //
  // 在内存中的地址：
  //
  //   base + m*K + k
  //
  // 即同一行 K 方向连续。
  using LayoutInputA = cutlass::layout::RowMajor;


  // CUTLASS 的 AlignmentInputA 单位是“元素个数”，不是 byte。
  //
  // 这里希望一次向量化访问 16 Bytes。
  //
  // 如果 DtypeA = FP8：
  //
  //   sizeof(FP8) = 1 Byte
  //
  // 所以：
  //
  //   AlignmentInputA = 16 / 1 = 16
  //
  // 即要求 A 在合适条件下一次按 16 个 FP8 元素进行向量化访问。
  constexpr int AlignmentInputA = 16 / sizeof(DtypeA);


  // ==========================================================================
  // 3. 定义矩阵 B（WQ）的 layout 和 alignment
  // ==========================================================================

  // B = WQ[K,N]
  //
  // ColumnMajor 表示同一列 K 方向连续。
  //
  // 逻辑上依然是：
  //
  //   B[k,n]
  //
  // 只是物理存储方式是 ColumnMajor。
  using LayoutInputB = cutlass::layout::ColumnMajor;


  // 与 A 同理。
  //
  // FP8 时：
  //
  //   AlignmentInputB = 16
  //
  // 表示 16 Byte / 16 个 FP8 元素的访问对齐要求。
  constexpr int AlignmentInputB = 16 / sizeof(DtypeB);


  // ==========================================================================
  // 4. Output layout
  // ==========================================================================

  // 最终 Out[M,N] 按 RowMajor 保存：
  //
  //   Out[m,n]
  //
  // 地址：
  //
  //   base + m*N + n
  //
  // 所以同一行中 N 方向连续。
  using LayoutOutput = cutlass::layout::RowMajor;


  // 输出也希望按照 16 Byte 做向量化访问。
  //
  // 如果 DtypeOutput = BF16：
  //
  //   sizeof(BF16) = 2 Byte
  //
  // 那么：
  //
  //   AlignmentOutput = 16 / 2 = 8
  //
  // 即一次可对应 8 个 BF16。
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);


  // ==========================================================================
  // 5. GPU architecture + Operator Class
  // ==========================================================================

  // SM89：
  //
  // 例如 Ada 架构 RTX 4090 / L4 / L40 等。
  //
  // SM89 原生支持 FP8 E4M3/E5M2 Tensor Core MMA。
  using ArchTag = cutlass::arch::Sm89;


  // 告诉 CUTLASS：
  //
  //   GEMM mainloop 使用 Tensor Core
  //
  // 而不是普通 SIMT CUDA Core GEMM。
  using OperatorClass = cutlass::arch::OpClassTensorOp;


  // ==========================================================================
  // 6. Threadblock Swizzle
  // ==========================================================================

  // Stream-K GEMM 调度方式。
  //
  // 普通 GEMM 经常是：
  //
  //   一个 threadblock 负责一个 [M_tile, N_tile]
  //
  // Stream-K 会进一步从 K 维工作量角度进行调度，
  // 目的是在某些 GEMM shape 下提高 GPU 负载均衡和利用率。
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;


  // ==========================================================================
  // 7. Tensor Core MMA Instruction Shape
  // ==========================================================================

  // SM89 FP8 Tensor Core 的典型 MMA shape：
  //
  //   m16n8k32
  //
  // 数学意义：
  //
  //   A_tile : [16,32]
  //   B_tile : [32,8]
  //
  //   C_tile += A_tile x B_tile
  //
  // 得到：
  //
  //   C_tile : [16,8]
  //
  // 其中：
  //
  //   16 * 8 * 32 = 4096 MAC
  //
  // 如果按 1 MAC = 2 FLOPs：
  //
  //   8192 FLOPs
  //
  // 注意：
  // 这是 warp-level Tensor Core MMA 的逻辑 instruction shape，
  // 不是说一个普通 CUDA thread 独自计算这个矩阵。
  using InstructionShape =
      cutlass::gemm::GemmShape<16, 8, 32>;


  // ==========================================================================
  // 8. FP8 accumulation operator
  // ==========================================================================

  // FastAccum::value 用于在两种 FP8 Tensor Core accumulation 策略之间选择。
  //
  // 注意：
  //
  //   OpMultiplyAddFastAccum
  //
  // 和
  //
  //   OpMultiplyAdd
  //
  // 这里并不是：
  //
  //   Tensor Core vs CUDA Core
  //
  // 两者仍然是在 TensorOp GEMM 中使用。
  //
  // FastAccum 主要体现 FP8 GEMM 中：
  //
  //   accumulation 精度
  //           vs
  //   accumulation 性能
  //
  // 之间的取舍。
  //
  // FastAccum 通常允许使用更偏性能导向的累加路径；
  // 非 FastAccum 则采用标准 multiply-add 语义。
  using Operator = std::conditional_t<
      FastAccum::value,
      cutlass::arch::OpMultiplyAddFastAccum,
      cutlass::arch::OpMultiplyAdd>;


  // ==========================================================================
  // 9. EVT Epilogue stage 数
  // ==========================================================================

  // 当前 visitor epilogue pipeline 使用 1 stage。
  constexpr auto NumEVTEpilogueStages = 1;


  // ==========================================================================
  // 10. OutputTileThreadMap
  // ==========================================================================
  //
  // 这是整个 EVT 中非常关键的东西。
  //
  // Tensor Core mainloop 算完以后：
  //
  //   Acc[M_tile,N_tile]
  //
  // 并不是：
  //
  //   thread0 -> Acc[0,0]
  //   thread1 -> Acc[0,1]
  //   thread2 -> Acc[0,2]
  //
  // 这么简单地分配。
  //
  // accumulator fragment 分散在不同线程的寄存器中。
  //
  // OutputTileThreadMap 描述：
  //
  //   “每个线程在 epilogue 中负责 output tile 的哪些元素”
  //
  // 后面的：
  //
  //   XScale Broadcast
  //   WScale Broadcast
  //   Bias Broadcast
  //   Output Store
  //
  // 都依赖这个映射关系。
  //
  // 可以粗略理解成它帮助建立：
  //
  //   thread + fragment index
  //              |
  //              v
  //          logical (m,n)
  //
  // ==========================================================================

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          DtypeOutput,
          AlignmentOutput,
          NumEVTEpilogueStages>;


  // ==========================================================================
  // 11. EVT Leaf #1：Accumulator
  // ==========================================================================
  //
  // VisitorAccFetch：
  //
  // 从 mainloop 产生的 accumulator fragment 中取得当前 epilogue
  // 正在处理的 accumulator 数据。
  //
  // 逻辑上：
  //
  //   Accum -> Acc[m,n]
  //
  // 它不是从 global memory 重新 load Acc。
  //
  // Acc 本身就是 GEMM Mainloop 算出来、当前仍由 GEMM kernel
  // 保存和传递到 epilogue 的 accumulator。
  //
  // ==========================================================================

  using Accum =
      cutlass::epilogue::threadblock::VisitorAccFetch;


  // ==========================================================================
  // 12. EVT Leaf #2：XScale
  // ==========================================================================
  //
  // x_scale 是 activation 的 per-token / per-row scale：
  //
  //   x_scale shape = [M]
  //
  // 例如：
  //
  //   x_scale =
  //
  //       [0.10,
  //        0.20,
  //        0.30]
  //
  //
  // 对整个 D[M,N] 来说，我们希望：
  //
  //                  N
  //
  //            n0    n1    n2    n3
  //
  //   m0      0.10  0.10  0.10  0.10
  //   m1      0.20  0.20  0.20  0.20
  //   m2      0.30  0.30  0.30  0.30
  //
  //
  // 即：
  //
  //   XScale(m,n) = x_scale[m]
  //
  //
  // 地址应该：
  //
  //   m 改变 -> 地址改变
  //   n 改变 -> 地址不改变
  //
  // 因此 stride：
  //
  //   Stride<1, 0, L_stride>
  //
  //
  // 第 0 维：M
  //
  //   stride = 1
  //
  // 第 1 维：N
  //
  //   stride = 0
  //
  // 所以 N 方向产生 broadcast。
  //
  // 这就是 VisitorColBroadcast。
  //
  // ==========================================================================

  using XScale =
      cutlass::epilogue::threadblock::VisitorColBroadcast<
          OutputTileThreadMap,
          DtypeScale,
          cute::Stride<
              cute::_1,   // M stride = 1：不同 token 读取不同 x_scale
              cute::_0,   // N stride = 0：同一 token 的所有 N 共用 scale
              int64_t>>;  // batch/L stride：运行时给出


  // XScale visitor 所需要的运行时 Arguments 类型。
  using XScaleArguments = typename XScale::Arguments;


  // ==========================================================================
  // 13. EVT Leaf #3：WScale
  // ==========================================================================
  //
  // w_scale 是 weight 的 per-channel scale：
  //
  //   w_scale shape = [N]
  //
  // 例如：
  //
  //   [0.01, 0.02, 0.03, 0.04]
  //
  //
  // 对输出矩阵 D[M,N] 来说：
  //
  //                 N
  //
  //          n0     n1     n2     n3
  //
  //   m0    0.01   0.02   0.03   0.04
  //   m1    0.01   0.02   0.03   0.04
  //   m2    0.01   0.02   0.03   0.04
  //
  //
  // 即：
  //
  //   WScale(m,n) = w_scale[n]
  //
  // 地址应该：
  //
  //   m 改变 -> 地址不改变
  //   n 改变 -> 地址 +1
  //
  // 所以：
  //
  //   Stride<0,1,L_stride>
  //
  // 这就是 VisitorRowBroadcast。
  //
  // ==========================================================================

  using WScale =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<
          OutputTileThreadMap,
          DtypeScale,
          cute::Stride<
              cute::_0,   // M stride = 0：不同 token 使用同一个 channel scale
              cute::_1,   // N stride = 1：不同 channel 使用不同 scale
              int64_t>>;  // batch/L stride


  using WScaleArguments = typename WScale::Arguments;


  // ==========================================================================
  // 14. EVT Leaf #4：Bias
  // ==========================================================================
  //
  // bias：
  //
  //   shape = [N]
  //
  // 和 w_scale 一样是 output-channel 维度的数据。
  //
  // 比如：
  //
  //   bias = [b0, b1, b2, b3]
  //
  // 对 D[M,N]：
  //
  //             n0   n1   n2   n3
  //
  //   m0        b0   b1   b2   b3
  //   m1        b0   b1   b2   b3
  //   m2        b0   b1   b2   b3
  //
  // 所以：
  //
  //   Bias(m,n) = bias[n]
  //
  // 同样使用：
  //
  //   Stride<0,1,...>
  //
  // ==========================================================================

  using Bias =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<
          OutputTileThreadMap,
          DtypeBias,
          cute::Stride<
              cute::_0,   // M stride = 0
              cute::_1,   // N stride = 1
              int64_t>>;


  using BiasArguments = typename Bias::Arguments;


  // ==========================================================================
  // 15. EVT Compute Node #1：乘 XScale
  // ==========================================================================
  //
  // VisitorCompute 只定义： sm80 89 
  //
  //   “我要执行什么运算”
  //
  // 这里：
  //
  //   cutlass::multiplies
  //
  // 表示：
  //
  //   result = lhs * rhs
  //
  //
  // DtypeEpilogue：
  //
  //   当前 epilogue 节点输出类型
  //
  // 第二个 DtypeEpilogue：
  //
  //   用于实际计算的 compute type
  //
  //
  // round_to_nearest：
  //
  //   如果过程中存在类型转换，使用 round-to-nearest。
  //
  // ==========================================================================

  using ApplyXScale =
      cutlass::epilogue::threadblock::VisitorCompute<
          cutlass::multiplies,
          DtypeEpilogue,
          DtypeEpilogue,
          cutlass::FloatRoundStyle::round_to_nearest>;


  // ==========================================================================
  // 16. EVT subtree #1
  //
  //                 ApplyXScale(*)
  //                   /        \
  //                  /          \
  //               Accum        XScale
  //
  //
  // 数学公式：
  //
  //   T1[m,n]
  //
  //     = Acc[m,n] * x_scale[m]
  //
  //
  // Sm80EVT 的模板形式可以简单理解为：
  //
  //   Sm80EVT<
  //       NodeOperation,
  //       Child0,
  //       Child1,
  //       ...
  //   >
  //
  //
  // 所以这里：
  //
  //   NodeOp = ApplyXScale
  //   Child0 = Accum
  //   Child1 = XScale
  //
  // ==========================================================================

  using EVTApplyXScale =
      cutlass::epilogue::threadblock::Sm80EVT<
          ApplyXScale, // 根节点：乘法
          Accum,       // 左孩子：Acc[m,n]
          XScale>;     // 右孩子：x_scale[m]


  // ==========================================================================
  // 17. EVT Compute Node #2：乘 WScale
  // ==========================================================================

  using ApplyWScale =
      cutlass::epilogue::threadblock::VisitorCompute<
          cutlass::multiplies,
          DtypeEpilogue, // ElementOutput
          DtypeEpilogue, // ElementCompute
          cutlass::FloatRoundStyle::round_to_nearest>;


  // ==========================================================================
  // 18. EVT subtree #2
  //
  //                         ApplyWScale(*)
  //                           /          \
  //                          /            \
  //                 EVTApplyXScale      WScale
  //                       |
  //                       *
  //                     /   \
  //                   Acc   XScale
  //
  //
  // 数学：
  //
  //   T2[m,n]
  //
  //       = T1[m,n] * w_scale[n]
  //
  //       = Acc[m,n]
  //         * x_scale[m]
  //         * w_scale[n]
  //
  // ==========================================================================

  using EVTApplyWScale =
      cutlass::epilogue::threadblock::Sm80EVT<
          ApplyWScale,     // 当前根节点：乘法
          EVTApplyXScale,  // 左孩子：Acc * XScale
          WScale>;         // 右孩子：w_scale[n]


  // ==========================================================================
  // 19. EVT Compute Node #3：加 Bias
  // ==========================================================================
  //
  // cutlass::plus：
  //
  //   result = lhs + rhs
  //
  // ==========================================================================

  using ApplyBias =
      cutlass::epilogue::threadblock::VisitorCompute<
          cutlass::plus,
          DtypeEpilogue,
          DtypeEpilogue,
          cutlass::FloatRoundStyle::round_to_nearest>;


  // ==========================================================================
  // 20. EVT subtree #3
  //
  //                              ApplyBias(+)
  //                              /          \
  //                             /            \
  //                    EVTApplyWScale       Bias
  //                         |
  //                         *
  //                       /   \
  //                      /     \
  //              EVTApplyXScale WScale
  //                    |
  //                    *
  //                  /   \
  //                Acc  XScale
  //
  //
  // 数学：
  //
  //   T3[m,n]
  //
  //      = T2[m,n] + bias[n]
  //
  //      = Acc[m,n]
  //        * x_scale[m]
  //        * w_scale[n]
  //        + bias[n]
  //
  // ==========================================================================

  using EVTApplyBias =
      cutlass::epilogue::threadblock::Sm80EVT<
          ApplyBias,
          EVTApplyWScale,
          Bias>;


  // ==========================================================================
  // 21. EVT Store Node：Output
  // ==========================================================================
  //
  // 前面 EVTApplyBias 只是完成了数学计算：
  //
  //   result
  //      = Acc * XScale * WScale + Bias
  //
  // 现在 VisitorAuxStore 负责把结果真正写到 global memory 中的 out。
  //
  //
  // OutputTileThreadMap：
  //
  //   决定当前线程负责输出 tile 中哪些位置。
  //
  //
  // DtypeOutput：
  //
  //   最终 out 的数据类型，例如 BF16。
  //
  //
  // Stride<int64_t, _1, int64_t>
  //
  // 描述逻辑 D[M,N,L] 的内存 stride：
  //
  //   M stride = runtime
  //   N stride = 1
  //   L stride = runtime
  //
  // 因此 N 方向连续，也就是 RowMajor output。
  //
  // ==========================================================================

  using Output =
      cutlass::epilogue::threadblock::VisitorAuxStore<
          OutputTileThreadMap,
          DtypeOutput,
          cutlass::FloatRoundStyle::round_to_nearest,
          cute::Stride<
              int64_t,    // M stride，运行时指定为 N
              cute::_1,   // N stride = 1
              int64_t>>;  // batch/L stride


  // ==========================================================================
  // 22. 完整 EVT Tree
  //
  //
  //                           Output(Store)
  //                               |
  //                               +
  //                             /   \
  //                            *    Bias[n]
  //                          /   \
  //                         *   WScale[n]
  //                       /   \
  //                  Acc[m,n] XScale[m]
  //
  //
  // 最终实现：
  //
  //   Out[m,n]
  //
  //      = Acc[m,n]
  //        * x_scale[m]
  //        * w_scale[n]
  //        + bias[n]
  //
  //
  // 注意：
  //
  // EVT 的价值不是“帮我们实现乘法和加法”。
  //
  // 真正价值是：
  //
  //   CUTLASS 已经知道：
  //
  //      thread + fragment element
  //                |
  //                v
  //            logical (m,n)
  //
  // 所以对于每个 accumulator fragment：
  //
  //   XScale visitor 可以找到 x_scale[m]
  //   WScale visitor 可以找到 w_scale[n]
  //   Bias visitor   可以找到 bias[n]
  //
  // 我们不用手写：
  //
  //   int m = ???;
  //   int n = ???;
  //
  // ==========================================================================

  using EVTOutput =
      cutlass::epilogue::threadblock::Sm80EVT<
          Output,
          EVTApplyBias>;


  // ==========================================================================
  // 23. 使用 DefaultGemmWithVisitor 生成整个 GEMM Kernel
  // ==========================================================================
  //
  // Mainloop 部分：
  //
  //   A FP8
  //      \
  //       Tensor Core MMA
  //      /
  //   B FP8
  //
  //        |
  //        v
  //
  //   DtypeAccum accumulator
  //
  //
  // Epilogue 部分：
  //
  //   使用我们刚刚定义的 EVTOutput。
  //
  //
  // 所以这个 kernel 的整体结构：
  //
  //      XQ ------\
  //                FP8 MMA mainloop
  //      WQ ------/
  //                     |
  //                     v
  //                   Acc
  //                     |
  //              EVT epilogue
  //                     |
  //                     v
  //                    Out
  //
  // ==========================================================================

  using EVTKernel =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<

          // ------------------------------------------------------------------
          // A
          // ------------------------------------------------------------------
          DtypeA,
          LayoutInputA,
          cutlass::ComplexTransform::kNone,
          AlignmentInputA,

          // ------------------------------------------------------------------
          // B
          // ------------------------------------------------------------------
          DtypeB,
          LayoutInputB,
          cutlass::ComplexTransform::kNone,
          AlignmentInputB,

          // ------------------------------------------------------------------
          // Output
          // ------------------------------------------------------------------
          DtypeOutput,
          LayoutOutput,
          AlignmentOutput,

          // ------------------------------------------------------------------
          // Mainloop accumulator type
          // ------------------------------------------------------------------
          DtypeAccum,

          // ------------------------------------------------------------------
          // Epilogue compute type
          // ------------------------------------------------------------------
          DtypeEpilogue,

          // ------------------------------------------------------------------
          // Tensor Core
          // ------------------------------------------------------------------
          OperatorClass,

          // ------------------------------------------------------------------
          // SM89
          // ------------------------------------------------------------------
          ArchTag,

          // ------------------------------------------------------------------
          // CTA-level GEMM tile
          //
          // 例如：
          //
          //   ThreadblockShape = GemmShape<128,128,64>
          //
          // 表示一个 CTA 负责逻辑上的：
          //
          //   M_tile = 128
          //   N_tile = 128
          //   K_tile = 64
          //
          // ------------------------------------------------------------------
          ThreadblockShape,

          // ------------------------------------------------------------------
          // Warp-level GEMM tile
          // ------------------------------------------------------------------
          WarpShape,

          // ------------------------------------------------------------------
          // Tensor Core MMA shape：
          //
          //   m16n8k32
          // ------------------------------------------------------------------
          InstructionShape,

          // ------------------------------------------------------------------
          // ★ 自定义 EVT Epilogue
          // ------------------------------------------------------------------
          EVTOutput,

          // Stream-K scheduling
          ThreadblockSwizzle,

          // Mainloop pipeline stages
          NumStages,

          // FP8 MMA accumulation operator
          Operator,

          // EVT epilogue stages
          NumEVTEpilogueStages>::GemmKernel;


  // ==========================================================================
  // 24. 使用 GemmUniversalAdapter 把底层 Kernel 包装成 device API
  // ==========================================================================
  //
  // EVTKernel 本质上还是底层 kernel type。
  //
  // GemmUniversalAdapter 提供：
  //
  //   can_implement()
  //   get_workspace_size()
  //   initialize()
  //   operator()()
  //
  // 等比较方便的 host/device 调用接口。
  //
  // ==========================================================================

  using Gemm =
      cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;


  // ==========================================================================
  // 25. GEMM problem size
  // ==========================================================================
  //
  // 描述：
  //
  //   M x N x K
  //
  // 对应：
  //
  //   A[M,K] x B[K,N] -> D[M,N]
  //
  // ==========================================================================

  cutlass::gemm::GemmCoord problem_size(M, N, K);


  // 当前不进行 Split-K。
  //
  // 整个 K 维由正常 GEMM mainloop 完成。
  constexpr auto SplitKFactor = 1;


  // ==========================================================================
  // 26. XScale runtime arguments
  // ==========================================================================
  //
  // 前面：
  //
  //   using XScale = ...
  //
  // 只是定义：
  //
  //   “XScale 是一个什么 visitor”
  //
  // 这里才真正告诉 kernel：
  //
  //   1. x_scale 的 global-memory 地址
  //   2. 默认值
  //   3. 实际 stride
  //
  //
  // x_scale 的逻辑访问：
  //
  //   XScale(m,n,l)
  //
  // 地址：
  //
  //   ptr
  //   + m * 1
  //   + n * 0
  //   + l * M
  //
  //
  // 因此：
  //
  //   m0,n0 -> x_scale[0]
  //   m0,n1 -> x_scale[0]
  //   m0,n2 -> x_scale[0]
  //
  //   m1,n0 -> x_scale[1]
  //   m1,n1 -> x_scale[1]
  //
  // 即沿 N broadcast。
  //
  // ==========================================================================

  XScaleArguments x_scale_arguments{
      // x_scale pointer
      (DtypeScale*)x_scale.data_ptr(),

      // 默认 scalar / fallback value。
      //
      // 对乘法来说，neutral element 是 1：
      //
      //   x * 1 = x
      //
      DtypeScale(1),

      // StrideMNL
      {
          cute::_1{},       // M stride = 1
          cute::_0{},       // N stride = 0
          problem_size.m()  // L/batch stride = M
      }
  };


  // ==========================================================================
  // 27. WScale runtime arguments
  // ==========================================================================
  //
  // WScale(m,n,l) 地址：
  //
  //   ptr
  //   + m * 0
  //   + n * 1
  //   + l * N
  //
  //
  // 所以：
  //
  //   m0,n0 -> w_scale[0]
  //   m0,n1 -> w_scale[1]
  //   m0,n2 -> w_scale[2]
  //
  //   m1,n0 -> w_scale[0]
  //   m1,n1 -> w_scale[1]
  //
  //
  // 即：
  //
  //   WScale[m,n] = w_scale[n]
  //
  // ==========================================================================

  WScaleArguments w_scale_arguments{
      // w_scale pointer
      (DtypeScale*)w_scale.data_ptr(),

      // 乘法的 neutral/default value = 1
      DtypeScale(1),

      // StrideMNL
      {
          cute::_0{},       // M stride = 0
          cute::_1{},       // N stride = 1
          problem_size.n()  // L/batch stride = N
      }
  };


  // ==========================================================================
  // 28. Bias runtime arguments
  // ==========================================================================
  //
  // Bias 和 WScale 一样按 N 维广播：
  //
  //   Bias[m,n] = bias[n]
  //
  //
  // 如果 bias.has_value()：
  //
  //   使用真实 bias pointer。
  //
  // 如果没有 bias：
  //
  //   pointer = nullptr
  //
  // 同时 fallback/default value = 0
  //
  // 因为：
  //
  //   x + 0 = x
  //
  // 所以没有 bias 时：
  //
  //   Out = Acc * XScale * WScale + 0
  //
  // ==========================================================================

  BiasArguments bias_arguments{
      bias.has_value()
          ? reinterpret_cast<DtypeBias*>(bias->data_ptr())
          : nullptr,

      // 加法 neutral value
      DtypeBias(0),

      // StrideMNL
      {
          cute::_0{},       // M stride = 0
          cute::_1{},       // N stride = 1
          problem_size.n()  // L/batch stride = N
      }
  };


  // ==========================================================================
  // 29. 分配最终输出 tensor
  // ==========================================================================
  //
  // shape:
  //
  //   [M,N]
  //
  // dtype:
  //
  //   torch_DtypeOutput
  //
  // device:
  //
  //   与 XQ 相同
  //
  // ==========================================================================

  auto out = torch::empty(
      {problem_size.m(), problem_size.n()},
      torch::dtype(torch_DtypeOutput).device(XQ.device()));


  // ==========================================================================
  // 30. Output Visitor runtime arguments
  // ==========================================================================
  //
  // 告诉 VisitorAuxStore：
  //
  //   1. 最终 D/out 写到哪里
  //   2. D 的内存 stride 是什么
  //
  //
  // out 是 RowMajor [M,N]。
  //
  // 所以：
  //
  //   Out[m,n]
  //
  // 地址：
  //
  //   base + m*N + n
  //
  //
  // 对应：
  //
  //   M stride = N
  //   N stride = 1
  //
  //
  // batch/L stride：
  //
  //   M*N
  //
  // ==========================================================================

  typename Output::Arguments output_arguments{
      (DtypeOutput*)out.data_ptr(),

      {
          problem_size.n(),           // M stride = N
          cute::_1{},                 // N stride = 1
          problem_size.mn().product() // L/batch stride = M*N
      }
  };


  // ==========================================================================
  // 31. 构造完整 EVT runtime arguments
  // ==========================================================================
  //
  // 这一段之所以看起来有很多 {{{ }}}，
  // 是因为它必须严格对应上面 EVT 树的嵌套结构。
  //
  //
  // 我们定义的树是：
  //
  //
  //                         EVTOutput
  //                         /       \
  //                    Output     EVTApplyBias
  //                                  |
  //                                  +
  //                                /   \
  //                      EVTApplyWScale Bias
  //                           |
  //                           *
  //                         /   \
  //               EVTApplyXScale WScale
  //                    |
  //                    *
  //                  /   \
  //               Accum XScale
  //
  //
  // 所以 Arguments 也必须按照完全相同的树嵌套。
  //
  // ==========================================================================

  typename EVTOutput::Arguments callback_arguments{

    // ------------------------------------------------------------------------
    // EVTApplyBias Arguments
    // ------------------------------------------------------------------------
    {
      // ----------------------------------------------------------------------
      // EVTApplyWScale Arguments
      // ----------------------------------------------------------------------
      {
        // --------------------------------------------------------------------
        // EVTApplyXScale Arguments
        // --------------------------------------------------------------------
        {
          {},                 // Accum
                              //
                              // VisitorAccFetch 不需要额外 runtime 参数，
                              // 所以为空 {}。

          x_scale_arguments,  // XScale

          {}                  // ApplyXScale
                              //
                              // VisitorCompute<multiplies>
                              // 本身没有额外 runtime 参数，
                              // 所以也是 {}。
        },

        w_scale_arguments,    // WScale

        {}                    // ApplyWScale
                              //
                              // 第二个 multiply 节点，无额外 runtime 参数。
      },

      bias_arguments,         // Bias

      {}                      // ApplyBias
                              //
                              // plus 节点无额外 runtime 参数。
    },

    // ------------------------------------------------------------------------
    // Output Visitor Arguments
    // ------------------------------------------------------------------------
    output_arguments
  };


  // ==========================================================================
  // 32. 构造整个 GEMM 的运行时 Arguments
  // ==========================================================================
  //
  // GemmUniversalMode::kGemm：
  //
  //   普通单次 GEMM。
  //
  //
  // problem_size：
  //
  //   M,N,K
  //
  //
  // SplitKFactor：
  //
  //   这里 = 1，不进行 Split-K。
  //
  //
  // callback_arguments：
  //
  //   ★ 我们自定义 EVT 的所有参数。
  //
  //
  // A pointer：
  //
  //   XQ
  //
  //
  // B pointer：
  //
  //   WQ
  //
  //
  // C / D pointer：
  //
  //   这里都传 nullptr。
  //
  // 原因：
  //
  //   传统 GEMM epilogue：
  //
  //       D = alpha * Acc + beta * C
  //
  //   会通过普通 Gemm Arguments 中的 ptr_C / ptr_D。
  //
  // 但这里完全使用 EVT：
  //
  //       Acc
  //        *
  //       XScale
  //        *
  //       WScale
  //        +
  //       Bias
  //        |
  //        v
  //   VisitorAuxStore -> out
  //
  // 所以：
  //
  //   C 根本没有使用；
  //   D 也由 VisitorAuxStore 的 output_arguments 提供。
  //
  // ==========================================================================

  typename Gemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,

      problem_size,

      SplitKFactor,

      // ★ EVT callback arguments
      callback_arguments,

      // A = XQ
      (DtypeA*)XQ.data_ptr(),

      // B = WQ
      (DtypeB*)WQ.data_ptr(),

      // ptr C
      //
      // 当前 EVT 完全不读取传统 C。
      nullptr,

      // ptr D
      //
      // 当前 EVT 使用 VisitorAuxStore 自己写 out，
      // 所以传统 ptr_D 不使用。
      nullptr,

      // ----------------------------------------------------------------------
      // batch stride A
      //
      // A 每一批矩阵有 M*K 个元素。
      //
      // 当前 kGemm 实际没有 batch，
      // 但 GemmUniversal Arguments 仍需要这些字段。
      // ----------------------------------------------------------------------
      problem_size.mk().product(),

      // batch stride B = N*K
      problem_size.nk().product(),

      // batch stride C：unused
      0,

      // batch stride D：unused
      0,

      // ----------------------------------------------------------------------
      // leading dimension / stride A
      //
      // A = RowMajor [M,K]
      //
      // A[m,k] 地址：
      //
      //   base + m*K + k
      //
      // 所以 lda = K。
      // ----------------------------------------------------------------------
      problem_size.k(),

      // ----------------------------------------------------------------------
      // leading dimension / stride B
      //
      // B = ColumnMajor [K,N]
      //
      // 对 ColumnMajor [K,N]：
      //
      //   B[k,n] 地址：
      //
      //   base + k + n*K
      //
      // 所以 ldb = K。
      // ----------------------------------------------------------------------
      problem_size.k(),

      // stride C：unused
      0,

      // stride D：unused
      0);


  // ==========================================================================
  // 33. 创建 CUTLASS GEMM device object
  // ==========================================================================

  Gemm gemm;


  // ==========================================================================
  // 34. 查询 workspace size
  // ==========================================================================
  //
  // 某些 CUTLASS GEMM 调度方式需要额外 workspace。
  //
  // 例如：
  //
  //   Stream-K
  //   Split-K
  //   reduction / synchronization metadata
  //
  // 等都可能需要额外空间。
  //
  // 所以不能简单假定 workspace_size = 0。
  //
  // ==========================================================================

  size_t workspace_size =
      Gemm::get_workspace_size(arguments);


  // ==========================================================================
  // 35. 在 GPU 上分配 CUTLASS workspace
  // ==========================================================================
  //
  // 创建 uint8/Byte tensor，
  //
  // 大小：
  //
  //   workspace_size Bytes
  //
  // 因为 XQ.new_empty：
  //
  //   默认跟 XQ 在同一 device。
  //
  // ==========================================================================

  auto workspace = XQ.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));


  // ==========================================================================
  // 36. can_implement()
  // ==========================================================================
  //
  // 在真正启动 kernel 前检查当前参数是否合法。
  //
  // 例如可能检查：
  //
  //   M/N/K 是否满足当前 kernel 要求
  //   pointer alignment
  //   lda / ldb
  //   Tensor Core alignment constraint
  //   problem size
  //
  // 等。
  //
  // 如果不满足：
  //
  //   CUTLASS 不会运行这个 kernel。
  //
  // ==========================================================================

  cutlass::Status status =
      gemm.can_implement(arguments);

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "cutlass cannot implement");
  }


  // ==========================================================================
  // 37. initialize()
  // ==========================================================================
  //
  // 根据 arguments 初始化 kernel 内部 Params，
  // 并绑定 workspace。
  //
  // 可以简单理解成：
  //
  //   host-side Arguments
  //          |
  //          v
  //   kernel-side Params
  //
  // 这里只是初始化，
  // 还没有真正 launch GEMM kernel。
  //
  // ==========================================================================

  status = gemm.initialize(
      arguments,
      workspace.data_ptr());

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "cutlass cannot initialize");
  }


  // ==========================================================================
  // 38. 真正 launch CUTLASS GEMM kernel
  // ==========================================================================
  //
  // 使用 PyTorch 当前 CUDA stream。
  //
  //
  // kernel 内实际经历：
  //
  //   XQ FP8
  //      \
  //       \
  //        FP8 Tensor Core GEMM Mainloop
  //       /
  //      /
  //   WQ FP8
  //
  //       |
  //       v
  //
  //   Accumulator
  //
  //       |
  //       v
  //
  //   EVT:
  //
  //       Acc
  //        |
  //        * x_scale[m]
  //        |
  //        * w_scale[n]
  //        |
  //        + bias[n]
  //        |
  //        v
  //     BF16 out
  //
  // ==========================================================================

  status =
      gemm(at::cuda::getCurrentCUDAStream());

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }


  // ==========================================================================
  // 39. PyTorch CUDA kernel launch error check
  // ==========================================================================
  //
  // 检查 kernel launch 是否出现 CUDA runtime error。
  //
  // 注意它主要检查 launch / runtime error 状态，
  // 并不是这里主动进行 cudaDeviceSynchronize()。
  //
  // ==========================================================================

  C10_CUDA_KERNEL_LAUNCH_CHECK();


  // ==========================================================================
  // 40. 返回最终结果
  // ==========================================================================
  //
  // out shape：
  //
  //   [M,N]
  //
  // 数学上：
  //
  //   out[m,n]
  //
  //     = (
  //         sum_k XQ[m,k] * WQ[k,n]
  //       )
  //       * x_scale[m]
  //       * w_scale[n]
  //       + bias[n]
  //
  // ==========================================================================

  return out;
}
template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_sm89(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale // FP32
){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // at::Tensor out) {
  int M = XQ.size(0);

  if (M <= 16) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<16, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else if (M <= 32) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<32, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else if (M <= 64) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<32, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else if (M <= 256) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 128, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/3,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<128, 128, 64>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  }
}

template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_on_sm(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale){ // FP32
    // float alpha,
    // float beta){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // at::Tensor out) {
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool sm89 = properties != nullptr && properties->major == 8 && properties->minor == 9;
  const bool sm9x = properties != nullptr && properties->major == 9;
  if (!(sm89 || sm9x)) {
    TORCH_CHECK(
        false, "Rowwise scaling only currently supported on SM89 and upper device");
    // placeholder, 无实际意义，避免报错error: return-statement with no value, in function returning ‘at::Tensor’ [-fpermissive]
    auto out = torch::empty({1,1},
                          torch::dtype(torch_DtypeOutput).device(XQ.device()) );
    return out;
//   }

//   if (sm9x) {
//     dispatch_fp8_rowwise_kernel_on_cluster_size_and_transpose<Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return dispatch_fp8_rowwise_kernel_sm89<Types...>(XQ, WQ, bias, x_scale, w_scale);//, out);
  }
}

template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_on_fast_accum(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    // float alpha,
    // float beta,
    bool use_fast_accum){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // bool use_fast_accum,
    // at::Tensor out) {
  if (use_fast_accum) {
    return dispatch_fp8_rowwise_kernel_on_sm<
        std::true_type,
        Types...>(XQ, WQ, bias, x_scale, w_scale);//, out);
  } else {
    return dispatch_fp8_rowwise_kernel_on_sm<
        std::false_type,
        Types...>(XQ, WQ, bias, x_scale, w_scale);//, out);
  }
}

template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_on_input_dtypes(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    bool use_fast_accum){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // bool use_fast_accum,
    // at::Tensor out) {
  if (XQ.dtype() == at::kFloat8_e5m2) {
    return dispatch_fp8_rowwise_kernel_on_fast_accum<
        cutlass::float_e5m2_t,
        cutlass::float_e4m3_t,
        Types...>(XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  } else {
    return dispatch_fp8_rowwise_kernel_on_fast_accum<
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        Types...>(XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  }
}
// 支持bf16和fp32 bias
torch::Tensor dispatch_fp8_rowwise_kernel_on_bias_dtype(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    bool use_fast_accum){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // bool use_fast_accum,
    // at::Tensor out) {
  if (bias.has_value() && bias->dtype() == at::kBFloat16) {
    return dispatch_fp8_rowwise_kernel_on_input_dtypes<
        cutlass::bfloat16_t>
        (XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  } else {
    return dispatch_fp8_rowwise_kernel_on_input_dtypes<
        float>
        //Types...>
        (XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  }
}

void check_inputs(
    torch::Tensor input,  // FP8
    torch::Tensor weight, // FP8
    std::optional<torch::Tensor> bias,   // BF16/FP16
    torch::Tensor scale_a, // FP32
    torch::Tensor scale_b // FP32
    ) {
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.device() == weight.device());
  TORCH_CHECK(scale_a.device() == input.device());
  TORCH_CHECK(scale_b.device() == weight.device());

  TORCH_CHECK(input.dtype() == at::kFloat8_e4m3fn || input.dtype() == at::kFloat8_e5m2);
  TORCH_CHECK(weight.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(scale_a.dtype() == at::kFloat);
  TORCH_CHECK(scale_b.dtype() == at::kFloat);

  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(input.size(1) == weight.size(0)); // a row major b col major
  TORCH_CHECK(scale_a.dim() == 2);
  TORCH_CHECK(scale_b.dim() == 2);
  TORCH_CHECK(scale_a.size(0) == input.size(0));
  TORCH_CHECK(scale_a.size(1) == 1);
  TORCH_CHECK(scale_b.size(0) == 1, "Expected scale_b.size(0) == 1, but got", scale_b.size(0));
  TORCH_CHECK(scale_b.size(1) == weight.size(1));

  TORCH_CHECK(input.stride(1) == 1);
  TORCH_CHECK(input.stride(0) >= input.size(1));
  TORCH_CHECK(weight.stride(0) == 1);
  TORCH_CHECK(weight.stride(1) >= weight.size(0));
  TORCH_CHECK(scale_a.stride(0) == 1);
  TORCH_CHECK(scale_b.stride(1) == 1);

  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == weight.device());
    TORCH_CHECK(bias->dtype() == at::kFloat || bias->dtype() == at::kBFloat16);
    TORCH_CHECK(bias->dim() == 1);
    TORCH_CHECK(bias->size(0) == weight.size(1));
    TORCH_CHECK(bias->stride(0) == 1);
  }

  // TORCH_CHECK(out.device() == a.device());
  // TORCH_CHECK(out.dtype() == at::kBFloat16);
  // TORCH_CHECK(out.dim() == 2);
  // TORCH_CHECK(out.size(0) == a.size(0));
  // TORCH_CHECK(out.size(1) == b.size(1));
  // TORCH_CHECK(out.stride(1) == 1);
  // TORCH_CHECK(out.stride(0) >= out.size(1));
}

// } // namespace

#endif // !defined(USE_ROCM)

// namespace at::cuda::detail {
torch::Tensor f8f8bf16_rowwise(
    torch::Tensor XQ, // FP8, [M, IC] row major
    torch::Tensor WQ, // FP8, [IC, OC] col major
    std::optional<torch::Tensor> bias, // BF16, [OC] 
    torch::Tensor x_scale, // FP32, [M, 1]
    torch::Tensor w_scale, // FP32, [1, OC]
    bool use_fast_accum){
    // at::Tensor XQ, // FP8
    // at::Tensor WQ, // FP8
    // at::Tensor x_scale, // FP32
    // at::Tensor w_scale, // FP32
    // std::optional<at::Tensor> bias, // BF16
    // bool use_fast_accum,
    // at::Tensor& out) {
#if defined(BUILD_ROWWISE_FP8_KERNEL)
  check_inputs(XQ, WQ,  bias, x_scale, w_scale);

  return dispatch_fp8_rowwise_kernel_on_bias_dtype(
      XQ, WQ,  bias, x_scale, w_scale, use_fast_accum);
#else // BUILD_ROWWISE_FP8_KERNEL
  TORCH_CHECK(
      false, "Rowwise scaling is not currently compiled, if you want to use fp8 sm89 kernel, pls add -DBUILD_ROWWISE_FP8_KERNEL");
  return;
#endif
}