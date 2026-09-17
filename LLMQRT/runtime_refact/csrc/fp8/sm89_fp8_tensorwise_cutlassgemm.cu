//adapted from sq
#include "sm89_fp8_gemm.h"

// #include <ATen/ATen.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include <optional>
#include <iostream>

#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000 && defined(CUDA_ARCH) && CUDA_ARCH >= 890

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)
// A[M,K]xB[K,N], A rowmajor B colmajor
template <
    typename ThreadblockShape,
    typename WarpShape,
    int NumStages,
    typename Dinput,
    typename Dweight,
    typename Doutput>
torch::Tensor f8f8bf16_tensorwise_impl_sm89(torch::Tensor input,  // FP8
                                            torch::Tensor weight, // FP8
                                            std::optional<torch::Tensor> bias,   // BF16/FP16
                                            float alpha,          // BF16/FP16
                                            float beta            // BF16/FP16
) {
  auto M = input.size(0);
  auto N = weight.size(1); 
  auto K = input.size(1);

  using ElementOutput = Doutput;//cutlass::bfloat16_t;//cutlass::half_t;//
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = Dinput; // e4m3/e5m2
  using ElementInputB = Dweight; // e4m3

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 890
  // std::cout << "hit GPU arch SM89 " << "\n";
  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, cutlass::layout::RowMajor, ElementInputB, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      ThreadblockShape,/*cutlass::gemm::GemmShape<256, 128, 64>*/
      WarpShape,/*cutlass::gemm::GemmShape<64, 64, 64>*/ cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 8,//128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;
#elif CUDA_ARCH >= 750
  // std::cout << "hit GPU arch SM75 " << "\n";
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, cutlass::layout::RowMajor, ElementInputB, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>>;
#elif CUDA_ARCH >= 700
  //std::cout << "hit GPU arch SM70 " << "\n";
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, cutlass::layout::RowMajor, ElementInputB, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  torch::Tensor bias_tensor;
  if (bias.has_value() && bias->dtype() == torch::kBFloat16) {
    bias_tensor = bias.value();
  } else {
    bias_tensor = torch::zeros({1, N}, torch::dtype(torch::kBFloat16).device(device) );
  }
  auto out = bias_tensor.view({1, -1}).repeat({M, 1});
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  // pytorch tensor不认识cutlass::bfloat16_t/e5m2/e4m3，所以这里out(torch bf16)需要先转void再强转cutlass::bfloat16_t
  // 这个问题解决了链接问题ImportError: /usr/local/lib/python3.10/dist-packages/runtime-0.1-py3.10-linux-x86_64.egg/runtime/sq_fp8_kernels.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIN7cutlass10bfloat16_tEEEPT_v
  void* out_ptr = out.data_ptr();
  ElementOutput* out_data = static_cast<ElementOutput*>(out_ptr);

  void* input_ptr = input.data_ptr();
  ElementInputA* input_data = static_cast<ElementInputA*>(input_ptr);

  void* weight_ptr = weight.data_ptr();
  ElementInputB* weight_data = static_cast<ElementInputB*>(weight_ptr);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input_data, LayoutInputA::packed(input_size));
      //input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight_data, LayoutInputB::packed(weight_size));
      //weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out_data, LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}

template <typename... Types>
torch::Tensor dispatch_fp8_tensorwise_kernel_on_sm89(
    torch::Tensor input,  // FP8
    torch::Tensor weight, // FP8
    std::optional<torch::Tensor> bias,   // BF16/FP16
    float alpha,          
    float beta) {
  int M = input.size(0);

  if (M <= 16) {
    return f8f8bf16_tensorwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<16, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(input, weight, bias, alpha, beta);
  } else if (M <= 32) {
    return f8f8bf16_tensorwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<32, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(input, weight, bias, alpha, beta);
  } else if (M <= 64) {
    return f8f8bf16_tensorwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<32, 64, 64>,
        /*NumStages=*/5,
        Types...>(input, weight, bias, alpha, beta);
  } else if (M <= 256) {
    return f8f8bf16_tensorwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 128, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/3,
        Types...>(input, weight, bias, alpha, beta);
  } else {
    return f8f8bf16_tensorwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<128, 128, 64>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/5,
        Types...>(input, weight, bias, alpha, beta);
  }
}

template <typename... Types>
torch::Tensor dispatch_fp8_tensorwise_kernel_on_input_dtypes(
    torch::Tensor input,  // FP8
    torch::Tensor weight, // FP8
    std::optional<torch::Tensor> bias,   // BF16/FP16
    float alpha,          
    float beta) {
  if (input.dtype() == torch::kFloat8_e5m2) {
    return dispatch_fp8_tensorwise_kernel_on_sm89<
        cutlass::float_e5m2_t,
        cutlass::float_e4m3_t,
        Types...>(input, weight, bias, alpha, beta);
  } else {
    return dispatch_fp8_tensorwise_kernel_on_sm89<
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        Types...>(input, weight, bias, alpha, beta);
  }
}

// 支持bf16和fp16 bias
torch::Tensor dispatch_fp8_tensorwise_kernel_on_bias_dtype(
    torch::Tensor input,  // FP8
    torch::Tensor weight, // FP8
    std::optional<torch::Tensor> bias,   // BF16/FP16
    float alpha,          
    float beta,
    const std::string& dtype_str) {

  if (dtype_str == "bfloat16") {
    if (bias.has_value()) {
      TORCH_CHECK(
        bias->dtype() == torch::kBFloat16,
        "if bias has value, dtype passed in should same as bias dtype "
      ); 
    }
    return dispatch_fp8_tensorwise_kernel_on_input_dtypes<
        cutlass::bfloat16_t>
        (input, weight, bias, alpha, beta);    
  } else if (dtype_str == "float16") {
    if (bias.has_value()) {
      TORCH_CHECK(
        bias->dtype() == torch::kFloat16,
        "if bias has value, dtype passed in should same as bias dtype "
      ); 
    }
    return dispatch_fp8_tensorwise_kernel_on_input_dtypes<
        cutlass::half_t>
        (input, weight, bias, alpha, beta);
  } else {
    TORCH_CHECK(
      false,
      "currently only support bf16 or fp16 output dtype, the passed dtype is ", dtype_str
    ); 
  }
}

void check_inputs(
    torch::Tensor input,  // FP8
    torch::Tensor weight, // FP8
    std::optional<torch::Tensor> bias,   // BF16/FP16
    float alpha,          
    float beta  ) {
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.device() == weight.device());
//   TORCH_CHECK(scale_a.device() == a.device());
//   TORCH_CHECK(scale_b.device() == weight.device());

  TORCH_CHECK(input.dtype() == torch::kFloat8_e4m3fn || input.dtype() == torch::kFloat8_e5m2);
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn);
//   TORCH_CHECK(scale_a.dtype() == torch::kFloat);
//   TORCH_CHECK(scale_b.dtype() == torch::kFloat);

  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(input.size(1) == weight.size(0)); // a row major b col major
//   TORCH_CHECK(scale_a.dim() == 2);
//   TORCH_CHECK(scale_b.dim() == 2);
//   TORCH_CHECK(scale_a.size(0) == a.size(0));
//   TORCH_CHECK(scale_a.size(1) == 1);
//   TORCH_CHECK(scale_b.size(0) == 1);
//   TORCH_CHECK(scale_b.size(1) == b.size(1));

  TORCH_CHECK(input.stride(1) == 1);
  TORCH_CHECK(input.stride(0) >= input.size(1));
  TORCH_CHECK(weight.stride(0) == 1); //COL MAJOR
//   TORCH_CHECK(weight.stride(1) >= weight.size(0));
//   TORCH_CHECK(scale_a.stride(0) == 1);
//   TORCH_CHECK(scale_b.stride(1) == 1);

  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == weight.device());
    TORCH_CHECK(bias->dtype() == torch::kFloat16 || bias->dtype() == torch::kBFloat16);
    TORCH_CHECK(bias->dim() == 1); // 
    TORCH_CHECK(bias->size(0) == weight.size(1));
    TORCH_CHECK(bias->stride(0) == 1);
  }

//   TORCH_CHECK(out.device() == a.device());
//   TORCH_CHECK(out.dtype() == torch::kBFloat16);
//   TORCH_CHECK(out.dim() == 2);
//   TORCH_CHECK(out.size(0) == a.size(0));
//   TORCH_CHECK(out.size(1) == b.size(1));
//   TORCH_CHECK(out.stride(1) == 1);
//   TORCH_CHECK(out.stride(0) >= out.size(1));
}

#endif // !defined(USE_ROCM)

torch::Tensor f8f8bf16_tensorwise(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    float alpha,
    float beta,
    const std::string& dtype_str) {
#if defined(BUILD_ROWWISE_FP8_KERNEL)
  check_inputs(XQ, WQ, bias, alpha, beta);

  return dispatch_fp8_tensorwise_kernel_on_bias_dtype(
      XQ, WQ, bias, alpha, beta, dtype_str);
#else // BUILD_ROWWISE_FP8_KERNEL
  TORCH_CHECK(
      false, "Rowwise scaling is not currently compiled, if you want to use fp8 sm89 kernel, pls add -DBUILD_ROWWISE_FP8_KERNEL");
  
  return;
#endif
}
