#!/usr/bin/env python
"""
手动计算 Perplexity
完全不依赖 lm_eval
"""
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from runtime_refact.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer
from runtime_refact.nn_models.modules.linear.linear_sq import SqW8A8BBF16OBF16PerTensor
from runtime_refact.nn_models.modules.linear.linear_fp8 import FP8DynamicLinear, FP8StaticLinear
try:
    from runtime_refact.nn_models.modules.linear.linear_awq import AWQLinear_GEMM
except ImportError:
    AWQLinear_GEMM = None
    print(f"[warning] awq linear is uninstalled...")

def parse_args():
    parser = argparse.ArgumentParser(description="评估 LLMQRT 量化模型的 perplexity")
    parser.add_argument("--model-path", required=True, help="量化模型目录")
    parser.add_argument("--output-file", default="./eval_results/manual_ppl.json")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--local-text-file", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    args = parser.parse_args()
    if args.max_length <= 1:
        parser.error("--max-length 必须大于 1")
    if args.stride <= 0 or args.stride > args.max_length:
        parser.error("--stride 必须在 [1, max-length] 范围内")
    if args.max_tokens is not None and args.max_tokens <= 1:
        parser.error("--max-tokens 必须大于 1")
    return args


args = parse_args()
model_path = args.model_path
dataset_name = args.dataset_name
dataset_config = args.dataset_config
dataset_split = args.dataset_split
max_length = args.max_length
stride = args.stride
local_text_file = args.local_text_file
torch_dtype = getattr(torch, args.torch_dtype)

print("="*70)
print("手动计算 Perplexity")
print("="*70)

# 加载模型
print(f"\n加载模型: {model_path}")
model = AutoQuantForCausalLM.from_quantized(
    model_path,
    torch_dtype=torch_dtype,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

# ========== 验证自定义 Linear 是否被使用 ==========
print("\n" + "="*70)
print("验证模型中的 Linear 层类型")
print("="*70)

linear_types = {}
total_linears = 0

for name, module in model.model.named_modules():
    if isinstance(module, SqW8A8BBF16OBF16PerTensor):
        linear_types[name] = "SQ Linear (Per-Tensor)"
        total_linears += 1
    elif isinstance(module, FP8DynamicLinear):
        linear_types[name] = "FP8 Dynamic Linear"
        total_linears += 1
    elif isinstance(module, FP8StaticLinear):
        linear_types[name] = "FP8 Static Linear"
        total_linears += 1
    elif AWQLinear_GEMM is not None and isinstance(module, AWQLinear_GEMM):
        linear_types[name] = "AWQ Linear"
        total_linears += 1
    elif 'Linear' in module.__class__.__name__ and 'torch.nn' in str(type(module).__module__):
        linear_types[name] = "Standard nn.Linear (未量化)"
        total_linears += 1

print(f"\n找到 {total_linears} 个 Linear 层")

if total_linears > 0:
    # 统计各类型数量
    type_counts = {}
    for layer_type in linear_types.values():
        type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
    
    print("\n层类型统计:")
    for layer_type, count in type_counts.items():
        print(f"  {layer_type}: {count} 个")
    
    # 显示前几个层作为示例
    custom_layers = [(name, ltype) for name, ltype in linear_types.items() 
                     if "未量化" not in ltype]
    if custom_layers:
        print(f"\n前 5 个自定义量化层示例:")
        for i, (name, layer_type) in enumerate(custom_layers[:5], 1):
            print(f"  [{i}] {name}: {layer_type}")
        if len(custom_layers) > 5:
            print(f"  ... 还有 {len(custom_layers) - 5} 个自定义量化层")
    
    # 检查是否有未量化的层
    standard_linear_count = type_counts.get("Standard nn.Linear (未量化)", 0)
    if standard_linear_count > 0:
        print(f"\n⚠️  警告: 发现 {standard_linear_count} 个未量化的 nn.Linear 层！")
        print("    这些层为LMhead, 未量化")
    else:
        print(f"\n✅ 确认: 所有 Linear 层都已被替换为自定义量化 Linear！")
else:
    print("  ⚠️  未找到任何 Linear 层")

# 运行时验证：测试一次 forward pass
print("\n运行测试 forward pass 以验证自定义 Linear 被调用...")

custom_linear_called = {"count": 0}

def forward_hook(module, input, output):
    custom_linear_called["count"] += 1

# 注册 hook 到第一个自定义 linear 层
first_custom_linear = None
custom_linear_types = (SqW8A8BBF16OBF16PerTensor, FP8DynamicLinear, FP8StaticLinear)
if AWQLinear_GEMM is not None:
    custom_linear_types = (
        SqW8A8BBF16OBF16PerTensor,
        FP8DynamicLinear,
        FP8StaticLinear,
        AWQLinear_GEMM,
    )

for name, module in model.model.named_modules():
    if isinstance(module, custom_linear_types):
        first_custom_linear = module
        handle = module.register_forward_hook(forward_hook)
        print(f"在 '{name}' 上注册了 forward hook")
        break

# 运行一次 forward 测试
if first_custom_linear is not None:
    test_input = tokenizer("Hello world", return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        _ = model.model(**test_input)
    
    if custom_linear_called["count"] > 0:
        print(f"✅ 确认: 自定义 Linear 在 forward pass 中被调用了 {custom_linear_called['count']} 次")
    else:
        print("⚠️  警告: 自定义 Linear 未被调用")
    
    handle.remove()
else:
    print("⚠️  未找到自定义 Linear 层进行验证")

print("="*70)

# 加载数据集
text = None

# 优先使用本地文本文件
if local_text_file is not None:
    print(f"\n使用本地文本文件: {local_text_file}")
    try:
        if os.path.exists(local_text_file):
            with open(local_text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"✅ 成功加载本地文件 ({len(text)} 字符)")
        else:
            print(f"⚠️  文件不存在: {local_text_file}")
    except Exception as e:
        print(f"⚠️  读取本地文件失败: {e}")

# 如果没有本地文件，尝试从 Hugging Face 加载
if text is None:
    print(f"\n加载数据集: {dataset_name} ({dataset_config})")
    try:
        print("尝试从 Hugging Face 加载...")
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
        )
        text = '\n\n'.join(dataset['text'])
        print(f"✅ 成功加载数据集")
    except Exception as e:
        print(f"⚠️  从 Hugging Face 加载失败: {e}")


# 方法 2: 使用简单的测试文本
if text is None:
    print("\n⚠️  无法加载数据集，使用简单测试文本")
    print("建议: 手动下载 wikitext 数据集或使用本地文本文件")
    
    # 创建一个简单的测试文本
    test_text = """
    The quick brown fox jumps over the lazy dog. This is a test sentence for perplexity calculation.
    Natural language processing is a field of artificial intelligence that focuses on the interaction between computers and humans using natural language.
    Machine learning models can be quantized to reduce their size and improve inference speed.
    """ * 100  # 重复多次以获得足够的文本
    
    text = test_text
    print(f"使用 {len(text)} 字符的测试文本")

# 如果有本地文本文件，可以这样加载
# with open('your_test_file.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
tokenize_kwargs = {"return_tensors": "pt", "verbose": False}
if args.max_tokens is not None:
    tokenize_kwargs.update(truncation=True, max_length=args.max_tokens)
encodings = tokenizer(text, **tokenize_kwargs)
print(f"数据集总 token 数: {encodings.input_ids.size(1)}")

# 计算 PPL
device = next(model.model.parameters()).device
seq_len = encodings.input_ids.size(1)
if seq_len <= 1:
    raise ValueError("评估文本至少需要产生 2 个 token")

nlls = []
num_loss_tokens = 0
prev_end_loc = 0

print("\n开始计算...")
with torch.no_grad():
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # 可能不同于步长
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model.model(input_ids, labels=target_ids)
        
        window_loss_tokens = (target_ids[:, 1:] != -100).sum().item()
        neg_log_likelihood = outputs.loss * window_loss_tokens
        nlls.append(neg_log_likelihood)
        num_loss_tokens += window_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

# 计算最终 PPL
ppl = torch.exp(torch.stack(nlls).sum() / num_loss_tokens)

print("\n" + "="*70)
print("评估结果")
print("="*70)
print(f"模型: {model_path}")
print(f"Perplexity: {ppl.item():.2f}")
print(f"总共评估 {num_loss_tokens} 个可预测 tokens")

# 显示验证信息
if 'type_counts' in locals() and type_counts:
    custom_count = sum(count for ltype, count in type_counts.items() if "未量化" not in ltype)
    if custom_count > 0:
        print(f"\n✅ 使用了 {custom_count} 个自定义量化 Linear 层")
        for ltype, count in type_counts.items():
            if "未量化" not in ltype:
                print(f"   - {ltype}: {count} 个")
    else:
        print(f"\n⚠️  警告: 未使用自定义量化层")

print("="*70)

# 保存结果
import json
output_file = args.output_file
Path(output_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

results = {
    "model": model_path,
    "dataset": f"{dataset_name}/{dataset_config}/{dataset_split}",
    "local_text_file": local_text_file,
    "max_length": max_length,
    "stride": stride,
    "torch_dtype": args.torch_dtype,
    "perplexity": ppl.item(),
    "tokens_evaluated": num_loss_tokens,
    "custom_linear_layers": {
        "total": sum(1 for ltype in linear_types.values() if "未量化" not in ltype),
        "types": {k: v for k, v in type_counts.items() if "未量化" not in k} if 'type_counts' in locals() else {},
    },
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: {output_file}")
