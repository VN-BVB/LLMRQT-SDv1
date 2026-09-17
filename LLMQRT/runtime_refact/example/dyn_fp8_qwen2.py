import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用的GPU编号

from runtime_refact.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer, TextStreamer
from runtime_refact.utils.common_utils import get_best_device
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time

qmodel_path = '/home/LLMQT/quant/examples/Qwen3-8B-dyn'
# qmodel_path = '/home/AutoAWQ/examples/Qwen2.5-14B-Instruct-awq'
# quant_config = {"quant_method": "awq", "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model并且把customizedLinear replace nn.Linear,然后调用HF API generate即可
# 无需额外传入quant config，qmodel_path里面有quant config(config.json)，from_quantized函数会读取它然后parse出对应的quant method，由此拿到对应的linear
# 返回Qwen2AWQForCausalLM(BaseAWQForCausalLM)
# 问题：对于awq triton这里只有设为fp16，不确定对于sq和fp8此处能否设为模型本身的类型bf16
model = AutoQuantForCausalLM.from_quantized(
  qmodel_path,
  torch_dtype=torch.bfloat16, # bf16 or fp16
  device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(qmodel_path, trust_remote_code=True)

# fwd
device = get_best_device()
# model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = [
  {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
  {"role": "user", "content": \
        "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"
        },
]

inputs = tokenizer.apply_chat_template(
  prompt,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors="pt",
  return_dict=True,
  enable_thinking=False
).to(device)

model.to(device)

# ==================== 方案1: 手动分离 Prefill 和 Decode 阶段 ====================
print("\n" + "="*70)
print("开始推理性能测试...")
print("="*70)

# Warmup (可选，避免首次运行的初始化开销)
print("\n[Warmup] 预热中...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=3, do_sample=False)
torch.cuda.synchronize()
print("[Warmup] 完成")

# ========== Prefill 阶段：处理输入 prompt，生成第一个 token ==========
print("\n[Prefill] 测量中...")
torch.cuda.synchronize()
prefill_start = time.perf_counter()

with torch.no_grad():
    # 第一次 forward 是 prefill（处理整个 prompt）
    outputs_prefill = model(**inputs)
    first_token_logits = outputs_prefill.logits[:, -1, :]  # 最后一个位置的 logits
    first_token_id = torch.argmax(first_token_logits, dim=-1)

torch.cuda.synchronize()
prefill_end = time.perf_counter()
prefill_latency = (prefill_end - prefill_start) * 1000  # 转换为毫秒

input_length = inputs['input_ids'].shape[1]
print(f"[Prefill] 延迟: {prefill_latency:.2f} ms")
print(f"[Prefill] 输入长度: {input_length} tokens")
print(f"[Prefill] 吞吐量: {input_length / (prefill_latency / 1000):.2f} tokens/s")

# ========== Decode 阶段：自回归生成后续 tokens ==========
print("\n[Decode] 测量中...")
max_new_tokens = 256
generated_tokens = [first_token_id.item()]
decode_latencies = []

# 构建初始输入（prompt + 第一个生成的 token）
current_input_ids = torch.cat([inputs['input_ids'], first_token_id.unsqueeze(0)], dim=1)
past_key_values = outputs_prefill.past_key_values  # 使用 KV cache

for i in range(max_new_tokens - 1):
    torch.cuda.synchronize()
    decode_start = time.perf_counter()
    
    with torch.no_grad():
        # Decode 阶段：只输入新生成的 token
        outputs = model(
            input_ids=first_token_id.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True
        )
        next_token_logits = outputs.logits[:, -1, :]
        
        # 采样策略
        if True:  # do_sample
            # 简单的 top-p 采样
            next_token_id = torch.argmax(next_token_logits, dim=-1)
        else:
            next_token_id = torch.argmax(next_token_logits, dim=-1)
    
    torch.cuda.synchronize()
    decode_end = time.perf_counter()
    
    decode_latency = (decode_end - decode_start) * 1000
    decode_latencies.append(decode_latency)
    
    # 更新状态
    first_token_id = next_token_id
    past_key_values = outputs.past_key_values
    generated_tokens.append(next_token_id.item())
    
    # 打印生成的 token（可选）
    if i < 10 or i % 50 == 0:
        decoded_text = tokenizer.decode([next_token_id.item()], skip_special_tokens=False)
        print(f"  Step {i+1}: {decoded_text} ({decode_latency:.2f} ms)")
    
    # 检查是否遇到 EOS
    if next_token_id.item() == tokenizer.eos_token_id:
        print(f"  遇到 EOS token，停止生成（第 {i+1} 步）")
        break

# ========== 统计结果 ==========
avg_decode_latency = sum(decode_latencies) / len(decode_latencies)
min_decode_latency = min(decode_latencies)
max_decode_latency = max(decode_latencies)
p50_decode_latency = sorted(decode_latencies)[len(decode_latencies) // 2]
p95_decode_latency = sorted(decode_latencies)[int(len(decode_latencies) * 0.95)]
p99_decode_latency = sorted(decode_latencies)[int(len(decode_latencies) * 0.99)]

print("\n" + "="*70)
print("性能测试结果")
print("="*70)
print(f"\n【Prefill 阶段】")
print(f"  延迟:           {prefill_latency:.2f} ms")
print(f"  输入长度:       {input_length} tokens")
print(f"  吞吐量:         {input_length / (prefill_latency / 1000):.2f} tokens/s")

print(f"\n【Decode 阶段】")
print(f"  生成 tokens:    {len(decode_latencies)} tokens")
print(f"  平均延迟:       {avg_decode_latency:.2f} ms/token")
print(f"  最小延迟:       {min_decode_latency:.2f} ms/token")
print(f"  最大延迟:       {max_decode_latency:.2f} ms/token")
print(f"  P50 延迟:       {p50_decode_latency:.2f} ms/token")
print(f"  P95 延迟:       {p95_decode_latency:.2f} ms/token")
print(f"  P99 延迟:       {p99_decode_latency:.2f} ms/token")
print(f"  吞吐量:         {1000 / avg_decode_latency:.2f} tokens/s")

print(f"\n【总体】")
total_latency = prefill_latency + sum(decode_latencies)
total_tokens = input_length + len(decode_latencies)
print(f"  总延迟:         {total_latency:.2f} ms")
print(f"  总 tokens:      {total_tokens} tokens")
print(f"  端到端吞吐量:   {total_tokens / (total_latency / 1000):.2f} tokens/s")

print("\n【生成文本】")
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"{generated_text}")
