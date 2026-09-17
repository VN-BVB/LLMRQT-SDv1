import torch
from runtime_refact.sq_fp8_kernels import cutlass_f8f8bf16_tensorwise_sm89
from icecream import ic

def test_scaled_mm_per_tensor():
    # 设置随机种子保证可复现性
    torch.manual_seed(42)
    
    # 定义输入张量（FP16/BF16 格式，因为 scaled_mm 通常用于低精度计算）
    a = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)  # 激活值
    b = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)  # 权重
    # 传进去的weight，必须是(in feats, out feats) col major
    b = b.t()
    # 定义 per-tensor 的缩放因子（单个标量）
    scale_a = torch.tensor(0.5, dtype=torch.float32, device="cuda")  # 输入张量的缩放因子
    scale_b = torch.tensor(1.2, dtype=torch.float32, device="cuda")  # 权重张量的缩放因子
    scale_output = torch.tensor(1.0, dtype=torch.float32, device="cuda")  # 输出的缩放因子
    
    # bias
    bias = torch.randn(b.shape[1], dtype=torch.bfloat16, device="cuda")
    # bias = None # to test std::optional if work
    # 调用本项目cutlass fp8 kernel
    alpha = scale_a * scale_b
    beta = 1
    output = cutlass_f8f8bf16_tensorwise_sm89(
        a,
        b,
        bias,
        alpha,
        beta,
        "bfloat16"
    ).cpu()
    # 调用 torch._scaled_mm（per-tensor 缩放）
    ref_output = torch._scaled_mm(
        a, b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
        # scale_result=scale_output,
        use_fast_accum=True,  # 是否使用快速累加（FP32 累加）
    ).to(torch.bfloat16).cpu()
    if bias is not None:
        bias = bias.cpu()
        ref_output += bias.reshape(1, -1)    
    # 验证输出
    print("Output shape:", output.shape)  # 应为 (128, 512)
    print("Output dtype:", output.dtype)  # 应为 FP32（默认累加类型）=> BF16
    diff = output - ref_output
    # import pdb;pdb.set_trace()
    ic(torch.allclose(output, ref_output, atol=0.25)) #true,手动看了下误差，最大是0.25, 尚不清楚为何


if __name__ == "__main__":
    test_scaled_mm_per_tensor()
    
    
# @torch.no_grad()
# def test_cutlass_f8f8bf16_tensorwise_sm89():
#     B, M, N = 128, 512, 1024
#     weight = torch.randint(-128, 127, (N, M), dtype=torch.int8) # row major
#     weight_t = weight.t() # col major weight转置为(M,N)
#     bias = torch.randn(N, dtype=torch.float16)
#     x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    
#     alpha, beta = 0.00334, 1
    
#     input_scale = 0.002
#     xx = x.to(torch.float16) * input_scale
    
#     linear = torch.nn.Linear(N, M, bias=True)# out, in
#     linear.weight.data = weight.to(torch.float16) * alpha
#     linear.bias.data = bias.to(torch.float16) * beta
    
#     alpha *= input_scale
#     y_gt = linear(xx)
#     for _ in range(20):
#         y = w8a8_int8_linear_bbf16_obf16_per_tensor( # kernel要求x为B M rowmajor, weight为(M,N) colmajor
#             x.cuda(), weight_t.cuda(), bias.cuda(), alpha, beta)
#     print("y_gt: ", y_gt)
#     print("y:", y.cpu())
#     mval = y_gt-y.cpu()
#     print("y_gt - y max val:", mval.max())
#     ic(torch.allclose(y_gt, y.cpu(), atol=0.1)) #true

