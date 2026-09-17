// 把fp8 gemm和sq gemm pybind绑定一下

// #include "fp8/fp8_gemm.h"
#include "smoothquant/sq_gemm.h"
#include "fp8/sm89_fp8_gemm.h"
#include "awq/awq_gemm.h"
#include "awq/gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("w8a8_int8_linear_bbf16_obf16_per_channel", &w8a8_int8_linear_bbf16_obf16_per_channel,
//         "int8 linear per channel");
  m.def("w8a8_int8_linear_bbf16_obf16_per_tensor", 
    &w8a8_int8_linear_bbf16_obf16_per_tensor,
    "int8 linear per tensor");
  m.def("cutlass_f8f8bf16_tensorwise_sm89", 
    &f8f8bf16_tensorwise, 
    "fp8 tensorwise linear on sm89");
  m.def("cutlass_f8f8bf16_rowwise_sm89", 
    &f8f8bf16_rowwise, 
    "fp8 rowwise linear on sm89");
  m.def(
    "awq_gemm",
    &awq_gemm,
    "w4a16 GEMM for AWQ");
  m.def(
    "awq_gemm_db",
    &awq_gemm_db,
    "w4a16 GEMM for AWQ with double buffering");
  m.def(
    "awq_gemm_cutlass",
    &awq_gemm_cutlass,
    "w4a16 GEMM for AWQ using CUTLASS Tensor Core fragments");
  m.def(
    "awq_gemv",
    &awq_gemv,
    "w4a16 GEMV for AWQ");
  m.def(
    "awq_gemv_coalesced",
    &awq_gemv_coalesced,
    "w4a16 GEMV for AWQ with coalesced memory access",
    py::arg("in_feats"),
    py::arg("kernel"),
    py::arg("scaling_factors"),
    py::arg("zeros"),
    py::arg("group_size"),
    py::arg("version") = 2);
}


