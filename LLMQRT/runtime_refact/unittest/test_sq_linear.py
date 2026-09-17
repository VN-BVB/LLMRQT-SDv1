import torch
from runtime_refact.sq_fp8_kernels import w8a8_int8_linear_bbf16_obf16_per_tensor
from icecream import ic


@torch.no_grad()
def test_quant_linear_a8_w8_bfp16_ofp16():
    B, M, N = 128, 512, 1024
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8) # row major
    weight_t = weight.t() # col major weight转置为(M,N)
    bias = torch.randn(N, dtype=torch.float16)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    
    alpha, beta = 0.00334, 1
    
    input_scale = 0.002
    xx = x.to(torch.float16) * input_scale
    
    linear = torch.nn.Linear(N, M, bias=True)# out, in
    linear.weight.data = weight.to(torch.float16) * alpha
    linear.bias.data = bias.to(torch.float16) * beta
    
    alpha *= input_scale
    # import pdb;pdb.set_trace()
    y_gt = linear(xx)#.to(torch.float16)) fp16 gemm
    # import pdb;pdb.set_trace()
    for _ in range(20):
        y = w8a8_int8_linear_bbf16_obf16_per_tensor( # kernel要求x为B M rowmajor, weight为(M,N) colmajor
            x.cuda(), weight_t.cuda(), bias.cuda(), alpha, beta)
    print("y_gt: ", y_gt)
    print("y:", y.cpu())
    mval = y_gt-y.cpu()
    print("y_gt - y max val:", mval.max())
    ic(torch.allclose(y_gt, y.cpu(), atol=0.1)) #true

# x=[BM] weight_t=[MN] bias=[N] alpha=00334*0.02 beta=1


# @torch.no_grad()
# def test_quant_linear_a8_w8_b8_o8():
#     B, M, N = 128, 512, 1024
#     weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
#     bias = torch.randint(-128, 127, (N,), dtype=torch.int8)
#     x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
#     alpha, beta = 0.001, 0.01
#     linear = torch.nn.Linear(M, N, bias=True)
#     linear.weight.data = weight.float() * alpha
#     linear.bias.data = bias.float() * beta
#     y_gt = linear(x.float()).clamp(-128, 127).round().long()
#     y = linear_a8_w8_b8_o8(x.cuda(), weight.cuda(),
#                            bias.cuda(), alpha, beta).cpu().long()
#     ic(torch.allclose(y_gt.float(), y.float().cpu(), atol=1))



if __name__ == '__main__':
    print('test_quant_linear_a8_w8_bfp16_ofp16')
    test_quant_linear_a8_w8_bfp16_ofp16()

