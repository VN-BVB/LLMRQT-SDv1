#!/usr/bin/env python
"""
手动计算原始（未量化）模型的 Perplexity
用于对比量化前后的精度损失
"""
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="评估原始未量化模型的 perplexity")
    parser.add_argument("--model-path", required=True, help="原始模型目录或 Hugging Face ID")
    parser.add_argument("--output-file", default="./eval_results/original_ppl.json")
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
        default="bfloat16",
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
print("原始模型 Perplexity 计算")
print("="*70)

# 加载模型
print(f"\n加载原始模型: {model_path}")
print("⚠️  注意: 原始 FP16/BF16 模型会占用较大显存")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

print(f"✅ 模型加载完成")
print(f"   模型类型: {model.__class__.__name__}")
print(f"   数据类型: {model.dtype}")

# 显示模型信息
total_params = sum(p.numel() for p in model.parameters())
print(f"   总参数量: {total_params / 1e9:.2f}B")

# 检查显存使用
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"   GPU 显存使用: {allocated:.2f} GB (已分配) / {reserved:.2f} GB (已保留)")

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

    # 方法 1: 尝试从 Hugging Face 加载
    try:
        print("尝试从 Hugging Face 加载...")
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
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
        Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data.
        Large language models have achieved remarkable performance on various natural language understanding and generation tasks.
        """ * 100  # 重复多次以获得足够的文本
        
        text = test_text
        print(f"使用 {len(text)} 字符的测试文本")

# Tokenize
tokenize_kwargs = {"return_tensors": "pt", "verbose": False}
if args.max_tokens is not None:
    tokenize_kwargs.update(truncation=True, max_length=args.max_tokens)
encodings = tokenizer(text, **tokenize_kwargs)
print(f"数据集总 token 数: {encodings.input_ids.size(1)}")

# 计算 PPL
device = next(model.parameters()).device
seq_len = encodings.input_ids.size(1)
if seq_len <= 1:
    raise ValueError("评估文本至少需要产生 2 个 token")

nlls = []
num_loss_tokens = 0
prev_end_loc = 0

print("\n开始计算...")
print(f"使用滑动窗口: max_length={max_length}, stride={stride}")

with torch.no_grad():
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # 可能不同于步长
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        
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
print("="*70)

# 保存结果
import json
output_file = args.output_file
Path(output_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

results = {
    "model": model_path,
    "model_type": "original_unquantized",
    "dataset": f"{dataset_name}/{dataset_config}/{dataset_split}",
    "local_text_file": local_text_file,
    "max_length": max_length,
    "stride": stride,
    "perplexity": ppl.item(),
    "tokens_evaluated": num_loss_tokens,
    "dtype": str(model.dtype),
    "total_params": total_params,
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: {output_file}")

# 显示最终显存使用
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n最终 GPU 显存使用: {allocated:.2f} GB (已分配) / {reserved:.2f} GB (已保留)")

print("\n" + "="*70)
print("提示: 可以将此结果与量化模型的 PPL 进行对比")
print("      量化模型脚本: eval_ppl_manual.py")
print("="*70)
