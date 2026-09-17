import torch

def quantize_fp16_to_int4(fp16_tensor, scale):
    """将FP16_Tensor量化为INT4（对称量化）""" 
    # 1. 缩放：FP16 ~ [-16, 15] 
    scaled_tensor = fp16_tensor / scale 

    # 2. 四舍五入并截断到INT4范围
    int4_tensor = torch.clamp(torch.round(scaled_tensor), min=-16, max=15)

    # 3. 转换为INT4类型（PyTorch默认int4是torch.int4）
    return int4_tensor # .to(torch.int4)

def dequantize_int4_to_fp16(int4_tensor, scale):
    """将INT8反量化为FP16""" 
    return int4_tensor.float() * scale

# 示例输入（FP16_Tensor）
fp16_data = torch.tensor([-1.8, 0.3, 0.9, -0.5], dtype=torch.float16)

# 计算Scale（根据最大值）
scale = torch.max(torch.abs(fp16_data)) / 15  # s = max(|x|) / 15
print(f"Scale: {scale.item():.4f}")  # 输出：Scale: 0.0142 (1.8/127)

# 量化和反量化
int4_data = quantize_fp16_to_int4(fp16_data, scale)
fp16_dequant = dequantize_int4_to_fp16(int4_data, scale)

print("原始FP16:", fp16_data)
print("量化INT8:", int4_data)
print("反量化FP16:", fp16_dequant)
print("量化误差:", torch.abs(fp16_data - fp16_dequant))