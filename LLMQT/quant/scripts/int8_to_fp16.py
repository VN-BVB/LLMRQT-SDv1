import numpy as np

# 假设量化参数
input_scale, weight_scale = 0.1, 0.05
output_scale = 0.02  # 假设下一层的input_scale=0.02

# 随机INT32 GEMM输出
int32 = np.random.randint(-10000, 10000, size=(10,), dtype=np.int32)

# 方案1：INT32直接转FP16
fp16_direct = int32.astype(np.float32) * (input_scale * weight_scale)

# 方案2：INT32→INT8→FP16
int8 = np.clip(np.round(int32 * (input_scale * weight_scale) / output_scale), -128, 127)
fp16_via_int8 = int8.astype(np.float32) * output_scale

# 比较两种结果
print("直接转FP16:\n", fp16_direct)
print("经INT8转FP16:\n", fp16_via_int8)
print("最大误差:", np.max(np.abs(fp16_direct - fp16_via_int8)))