import torch
import numpy as np
from awq_gemm import awq_gemm  # 编译后的模块

# ---- 超参数 ----
M, IC, OC = 32, 256, 256       # 可改成任意 64/128 倍数
GROUP_SIZE = 128               # 必须满足前面所有检查
SEED = 42
DEVICE = 'cuda'
ATOL = 1e-3                    # fp16 允许误差

torch.manual_seed(SEED)

# ---- 辅助：int4 pack ----
def pack_int4(weights):
    """weights: [IC, OC] int numpy -> [IC, OC//8] uint32"""
    assert weights.shape[1] % 8 == 0
    OC = weights.shape[1]
    weights = weights & 0xf  # 确保 0-15
    out = np.zeros((weights.shape[0], OC // 8), dtype=np.uint32)
    for oc in range(0, OC, 8):
        tmp = (weights[:, oc+7].astype(np.uint32) << 28) | \
              (weights[:, oc+6].astype(np.uint32) << 24) | \
              (weights[:, oc+5].astype(np.uint32) << 20) | \
              (weights[:, oc+4].astype(np.uint32) << 16) | \
              (weights[:, oc+3].astype(np.uint32) << 12) | \
              (weights[:, oc+2].astype(np.uint32) <<  8) | \
              (weights[:, oc+1].astype(np.uint32) <<  4) | \
              (weights[:, oc+0].astype(np.uint32) <<  0)
        out[:, oc//8] = tmp
    return out

# ---- 生成随机数据 ----
X_fp16   = torch.randn((M, IC), dtype=torch.float16, device=DEVICE)
W_fp16   = torch.randn((IC, OC), dtype=torch.float16, device=DEVICE)

# 量化 int4
W_min, W_max = W_fp16.min(dim=1, keepdim=True)[0], W_fp16.max(dim=1, keepdim=True)[0]
scale = (W_max - W_min) / 15.0
zero  = torch.round(-W_min / scale).clamp(0, 15).to(torch.int32)

W_int4 = torch.round((W_fp16 - W_min) / scale).clamp(0, 15).to(torch.int32)

# pack
W_int4_np = W_int4.cpu().numpy()
kernel    = torch.from_numpy(pack_int4(W_int4_np)).to(DEVICE)  # [IC, OC//8]

# scale/zero 也按组折叠
groups_per_ic = IC // GROUP_SIZE
scale_group = scale[:, ::GROUP_SIZE]            # [IC, 1] -> [groups, 1]
zero_group  = zero[:, ::GROUP_SIZE]             # [IC, 1] -> [groups, 1]
scale_group = scale_group.contiguous()
zero_group  = zero_group.contiguous()

# ---- PyTorch 参考实现 ----
def pytorch_ref(X, W_int4, scale, zero, group_size):
    # 先反量化回 fp16
    W_deq = []
    for g in range(IC // group_size):
        rows = slice(g * group_size, (g+1) * group_size)
        W_deq.append((W_int4[rows].float() - zero[rows]) * scale[rows])
    W_deq = torch.cat(W_deq, dim=0).to(torch.float16)
    return X @ W_deq

ref_out = pytorch_ref(X_fp16, W_int4, scale, zero, GROUP_SIZE)

# ---- 调用 CUDA kernel ----
split_k = 2
out_cuda = awq_gemm(X_fp16, kernel, scale_group, zero_group, split_k)

# ---- 比较 ----
err = (out_cuda - ref_out).abs().max().item()
print(f"max abs err = {err:.5f}")
assert err < ATOL, f"err {err} >= {ATOL}"
print("✅ AWQ GEMM unit test passed!")