import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用的GPU编号

from runtime_refact.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer, TextStreamer
from runtime_refact.utils.common_utils import get_best_device
import torch
from torch.profiler import profile, record_function, ProfilerActivity
# transformers=4.54.0可以跑起来qwen3，4.57.2和4.46.0都跑不起来
qmodel_path = '/home/lyc/workspace/week78/LLMQRT/models/Qwen3-0.6B-awq'

# Load model并且把customizedLinear replace nn.Linear,然后调用HF API generate即可
model = AutoQuantForCausalLM.from_quantized(
  qmodel_path,
  torch_dtype=torch.float16, # bf16 or fp16
  device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(qmodel_path, trust_remote_code=True)

# fwd
device = get_best_device()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = [
  {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
  {"role": "user", "content": \
        "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"},
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
# 使用torch profiler抓取推理timeline
# 方法 1: 保存到 tensorboard打开
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True,
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs')
# ) as prof:
#     with record_function("model_generate"):
#         outputs = model.generate(
#             **inputs,
#             do_sample=True,
#             max_new_tokens=5,
#             streamer=streamer,
#         )

# 方法 2: 分析并打印 top 10 kernel（推荐）
print("\n" + "="*70)
print("Profiling 并分析 Kernel 性能")
print("="*70)

# Warmup - 确保 CUDA 已初始化
print("Warmup...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=2, do_sample=False)
torch.cuda.synchronize()
print("Warmup 完成")

# 开始 profiling
print("\n开始 profiling...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,  # 关闭以提高性能
    profile_memory=False,
    with_stack=False,
) as prof:
    with record_function("model_generate"):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=10,  # 增加到 10 以便有更多 kernel 调用
                # streamer=streamer,  # 关闭 streamer 以避免输出干扰
            )
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成

print("\nProfiling 完成，开始分析...")

# 获取所有事件
all_events = list(prof.key_averages())
print(f"总共找到 {len(all_events)} 个事件")

# 调试: 打印第一个事件的属性（帮助排查问题）
if len(all_events) > 0:
    first_event = all_events[0]
    print(f"\n调试信息 - 第一个事件的属性:")
    print(f"  key: {first_event.key}")
    print(f"  count: {first_event.count}")
    # 列出所有包含 'time' 的属性
    time_attrs = [attr for attr in dir(first_event) if 'time' in attr.lower() and not attr.startswith('_')]
    print(f"  可用的 time 属性: {time_attrs}")

# 获取所有 CUDA kernel 事件
cuda_events = []
cpu_events_with_cuda = []

for event in all_events:
    # 兼容不同 PyTorch 版本的属性名
    cuda_time = 0
    self_cuda_time = 0
    
    # 尝试不同的属性名
    if hasattr(event, 'cuda_time_total'):
        cuda_time = event.cuda_time_total
        self_cuda_time = getattr(event, 'self_cuda_time_total', 0)
    elif hasattr(event, 'device_time_total'):
        cuda_time = event.device_time_total
        self_cuda_time = getattr(event, 'self_device_time_total', 0)
    elif hasattr(event, 'cuda_time'):
        cuda_time = event.cuda_time
        self_cuda_time = getattr(event, 'self_cuda_time', 0)
    
    # 检查是否有 CUDA 时间
    if cuda_time > 0:
        event_data = {
            'name': event.key,
            'cuda_time': cuda_time,  # 微秒
            'self_cuda_time': self_cuda_time,  # 自身 CUDA 时间
            'count': event.count,
            'avg_time': cuda_time / event.count if event.count > 0 else 0,
            'device_type': str(event.device_type) if hasattr(event, 'device_type') else 'N/A'
        }
        
        # 区分纯 CUDA kernel 和 CPU 操作
        if hasattr(event, 'device_type') and event.device_type == ProfilerActivity.CUDA:
            cuda_events.append(event_data)
        else:
            cpu_events_with_cuda.append(event_data)

print(f"找到 {len(cuda_events)} 个纯 CUDA kernel 事件")
print(f"找到 {len(cpu_events_with_cuda)} 个包含 CUDA 时间的 CPU 操作")

# 如果没有找到纯 CUDA kernel，可能所有操作都被分类为 CPU
# 但它们仍然有 CUDA 时间，我们应该显示它们
if len(cuda_events) == 0 and len(cpu_events_with_cuda) > 0:
    print("\n提示: 所有 CUDA 操作都被分类为 CPU 操作，这是正常的")
    print("      我们将显示所有包含 CUDA 时间的操作")

# 合并所有包含 CUDA 时间的事件
all_cuda_events = cuda_events + cpu_events_with_cuda

# 按总耗时排序
all_cuda_events_sorted = sorted(all_cuda_events, key=lambda x: x['cuda_time'], reverse=True)

print("\n" + "="*70)
print("前10个最耗时的 CUDA Kernel")
print("="*70)

if len(all_cuda_events_sorted) > 0:
    print(f"\n{'排名':<5} {'总耗时(ms)':<15} {'调用次数':<10} {'平均耗时(ms)':<15} {'Kernel 名称'}")
    print("-" * 120)
    
    for idx, event in enumerate(all_cuda_events_sorted[:10], 1):
        total_time_ms = event['cuda_time'] / 1000  # 转换为毫秒
        avg_time_ms = event['avg_time'] / 1000
        kernel_name = event['name'][:80]  # 截断过长的名称
        print(f"{idx:<5} {total_time_ms:<15.2f} {event['count']:<10} {avg_time_ms:<15.4f} {kernel_name}")
    
    # 计算总时间
    total_cuda_time = sum(e['cuda_time'] for e in all_cuda_events) / 1000  # ms
    top10_time = sum(e['cuda_time'] for e in all_cuda_events_sorted[:10]) / 1000  # ms
    print("\n" + "-" * 120)
    print(f"Top 10 总耗时: {top10_time:.2f} ms ({top10_time/total_cuda_time*100:.1f}% of total)")
    print(f"所有 CUDA 操作总耗时: {total_cuda_time:.2f} ms")
else:
    print("\n⚠️  未找到任何 CUDA kernel 事件")
    print("\n尝试使用内置的 table() 方法显示所有事件:")
    print("\n按 CUDA 时间排序:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))

# ========== 可选: 按平均耗时排序 ==========
if len(all_cuda_events) > 0:
    print("\n" + "="*70)
    print("前10个平均耗时最高的 Kernel (单次调用)")
    print("="*70)
    
    all_cuda_events_by_avg = sorted(all_cuda_events, key=lambda x: x['avg_time'], reverse=True)
    
    print(f"\n{'排名':<5} {'平均耗时(ms)':<15} {'调用次数':<10} {'总耗时(ms)':<15} {'Kernel 名称'}")
    print("-" * 120)
    
    for idx, event in enumerate(all_cuda_events_by_avg[:10], 1):
        total_time_ms = event['cuda_time'] / 1000
        avg_time_ms = event['avg_time'] / 1000
        kernel_name = event['name'][:80]
        print(f"{idx:<5} {avg_time_ms:<15.4f} {event['count']:<10} {total_time_ms:<15.2f} {kernel_name}")

# ========== 可选: 保存详细报告到文件 ==========
if len(all_cuda_events) > 0:
    try:
        import json
        import os
        
        output_dir = "./profiler_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 JSON 格式
        profile_data = {
            "total_cuda_time_ms": total_cuda_time,
            "num_cuda_kernels": len(cuda_events),
            "num_cpu_ops_with_cuda": len(cpu_events_with_cuda),
            "top10_by_total_time": [
                {
                    "rank": i+1,
                    "name": e['name'],
                    "total_time_ms": e['cuda_time'] / 1000,
                    "count": e['count'],
                    "avg_time_ms": e['avg_time'] / 1000,
                }
                for i, e in enumerate(all_cuda_events_sorted[:10])
            ],
            "top10_by_avg_time": [
                {
                    "rank": i+1,
                    "name": e['name'],
                    "avg_time_ms": e['avg_time'] / 1000,
                    "count": e['count'],
                    "total_time_ms": e['cuda_time'] / 1000,
                }
                for i, e in enumerate(all_cuda_events_by_avg[:10])
            ]
        }
        
        output_file = os.path.join(output_dir, "kernel_profiling.json")
        with open(output_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"\n详细 profiling 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n⚠️  保存 profiling 结果失败: {e}")
 
print("\n" + "="*70)

# ！以下为性能测试代码，性能测试时需要注释掉以上的profile抓取推理timeline代码
# ==================== 性能测试方案: 手动分离 Prefill 和 Decode 阶段 ====================
# import time

# print("\n" + "="*70)
# print("开始推理性能测试...")
# print("="*70)

# # Warmup (可选，避免首次运行的初始化开销)
# print("\n[Warmup] 预热中...")
# with torch.no_grad():
#     _ = model.generate(**inputs, max_new_tokens=3, do_sample=False)
# torch.cuda.synchronize()
# print("[Warmup] 完成")

# # ========== Prefill 阶段：处理输入 prompt，生成第一个 token ==========
# print("\n[Prefill] 测量中...")
# torch.cuda.synchronize()
# prefill_start = time.perf_counter()

# with torch.no_grad():
#     # 第一次 forward 是 prefill（处理整个 prompt）
#     outputs_prefill = model(**inputs)
#     first_token_logits = outputs_prefill.logits[:, -1, :]  # 最后一个位置的 logits
#     first_token_id = torch.argmax(first_token_logits, dim=-1)

# torch.cuda.synchronize()
# prefill_end = time.perf_counter()
# prefill_latency = (prefill_end - prefill_start) * 1000  # 转换为毫秒

# input_length = inputs['input_ids'].shape[1]
# print(f"[Prefill] 延迟: {prefill_latency:.2f} ms")
# print(f"[Prefill] 输入长度: {input_length} tokens")
# print(f"[Prefill] 吞吐量: {input_length / (prefill_latency / 1000):.2f} tokens/s")

# # ========== Decode 阶段：自回归生成后续 tokens ==========
# print("\n[Decode] 测量中...")
# max_new_tokens = 256
# generated_tokens = [first_token_id.item()]
# decode_latencies = []

# # 构建初始输入（prompt + 第一个生成的 token）
# current_input_ids = torch.cat([inputs['input_ids'], first_token_id.unsqueeze(0)], dim=1)
# past_key_values = outputs_prefill.past_key_values  # 使用 KV cache

# for i in range(max_new_tokens - 1):
#     torch.cuda.synchronize()
#     decode_start = time.perf_counter()
    
#     with torch.no_grad():
#         # Decode 阶段：只输入新生成的 token
#         outputs = model(
#             input_ids=first_token_id.unsqueeze(0),
#             past_key_values=past_key_values,
#             use_cache=True
#         )
#         next_token_logits = outputs.logits[:, -1, :]
        
#         # 采样策略
#         if True:  # do_sample
#             # 简单的 top-p 采样
#             next_token_id = torch.argmax(next_token_logits, dim=-1)
#         else:
#             next_token_id = torch.argmax(next_token_logits, dim=-1)
    
#     torch.cuda.synchronize()
#     decode_end = time.perf_counter()
    
#     decode_latency = (decode_end - decode_start) * 1000
#     decode_latencies.append(decode_latency)
    
#     # 更新状态
#     first_token_id = next_token_id
#     past_key_values = outputs.past_key_values
#     generated_tokens.append(next_token_id.item())
    
#     # 打印生成的 token（可选）
#     if i < 10 or i % 50 == 0:
#         decoded_text = tokenizer.decode([next_token_id.item()], skip_special_tokens=False)
#         print(f"  Step {i+1}: {decoded_text} ({decode_latency:.2f} ms)")
    
#     # 检查是否遇到 EOS
#     if next_token_id.item() == tokenizer.eos_token_id:
#         print(f"  遇到 EOS token，停止生成（第 {i+1} 步）")
#         break

# # ========== 统计结果 ==========
# avg_decode_latency = sum(decode_latencies) / len(decode_latencies)
# min_decode_latency = min(decode_latencies)
# max_decode_latency = max(decode_latencies)
# p50_decode_latency = sorted(decode_latencies)[len(decode_latencies) // 2]
# p95_decode_latency = sorted(decode_latencies)[int(len(decode_latencies) * 0.95)]
# p99_decode_latency = sorted(decode_latencies)[int(len(decode_latencies) * 0.99)]

# print("\n" + "="*70)
# print("性能测试结果")
# print("="*70)
# print(f"\n【Prefill 阶段】")
# print(f"  延迟:           {prefill_latency:.2f} ms")
# print(f"  输入长度:       {input_length} tokens")
# print(f"  吞吐量:         {input_length / (prefill_latency / 1000):.2f} tokens/s")

# print(f"\n【Decode 阶段】")
# print(f"  生成 tokens:    {len(decode_latencies)} tokens")
# print(f"  平均延迟:       {avg_decode_latency:.2f} ms/token")
# print(f"  最小延迟:       {min_decode_latency:.2f} ms/token")
# print(f"  最大延迟:       {max_decode_latency:.2f} ms/token")
# print(f"  P50 延迟:       {p50_decode_latency:.2f} ms/token")
# print(f"  P95 延迟:       {p95_decode_latency:.2f} ms/token")
# print(f"  P99 延迟:       {p99_decode_latency:.2f} ms/token")
# print(f"  吞吐量:         {1000 / avg_decode_latency:.2f} tokens/s")

# print(f"\n【总体】")
# total_latency = prefill_latency + sum(decode_latencies)
# total_tokens = input_length + len(decode_latencies)
# print(f"  总延迟:         {total_latency:.2f} ms")
# print(f"  总 tokens:      {total_tokens} tokens")
# print(f"  端到端吞吐量:   {total_tokens / (total_latency / 1000):.2f} tokens/s")

# print("\n【生成文本】")
# generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
# print(f"{generated_text}")
