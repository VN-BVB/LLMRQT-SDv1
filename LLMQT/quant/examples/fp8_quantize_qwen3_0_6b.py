"""Quantize the local Qwen3-0.6B model with dynamic and/or static FP8."""

import argparse
import gc
import os
import sys
from pathlib import Path

# Reuse the Hugging Face cache used by the existing Qwen3-0.6B examples.
os.environ.setdefault("HF_HOME", r"E:\aiinfra\hf_home")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Allow this file to be run from either the repository root or examples directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from quant.core.api import AutoQuantForCausalLM


DEFAULT_MODEL_PATH = Path(r"E:\aiinfra\llmqt_awq\models\Qwen--Qwen3-0.6B")
DEFAULT_OUTPUT_ROOT = Path(r"E:\aiinfra\llmqt_fp8\quantized")
QUANT_METHODS = {
    "dynamic": "fp8_dynamic_quant",
    "static": "fp8_static_quant",
    "native-dynamic": "fp8_native_dynamic_quant",
    "native-static": "fp8_native_static_quant",
}
DEFAULT_CALIB_TEXTS = [
    "大型语言模型根据上下文预测下一个 token，并能够完成问答、摘要和代码任务。",
    "FP8 量化通过降低权重和激活精度来减少模型推理的显存与计算开销。",
    "静态量化先用代表性文本校准激活范围，推理时复用预先计算的缩放因子。",
    "动态量化在每次前向传播时计算激活缩放因子，对输入分布变化适应性更强。",
    "Transformer 由注意力层、前馈网络、归一化层、词嵌入和输出层组成。",
    "Qwen3-0.6B 是一个紧凑的生成式语言模型，适合在消费级 GPU 上进行实验。",
    "Quantization maps floating-point tensors into a smaller numeric range using scale factors.",
    "A representative calibration set should cover varied sentence lengths and semantic topics.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="对本地 Qwen3-0.6B 执行 FP8 动态量化、静态量化或两者。"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"原始 Qwen3-0.6B 模型目录（默认：{DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"两个量化模型的父目录（默认：{DEFAULT_OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--mode",
        choices=("dynamic", "static", "both", "native-dynamic", "native-static", "native-both"),
        default="both",
        help="量化模式（默认：both）",
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=16,
        help="静态量化的校准样本数（默认：16）",
    )
    parser.add_argument(
        "--max-calib-seq-len",
        type=int,
        default=128,
        help="静态量化的校准序列长度（默认：128）",
    )
    return parser.parse_args()


def selected_modes(mode):
    if mode == "both":
        return ("dynamic", "static")
    if mode == "native-both":
        return ("native-dynamic", "native-static")
    return (mode,)


def output_path(output_root, mode):
    return output_root / f"Qwen3-0.6B-fp8-{mode}"


def validate_args(args):
    if not args.model_path.is_dir():
        raise FileNotFoundError(
            f"找不到原始模型目录：{args.model_path}\n"
            "请通过 --model-path 指定 Qwen3-0.6B 的本地目录。"
        )
    if args.max_calib_samples <= 0:
        raise ValueError("--max-calib-samples 必须大于 0。")
    if args.max_calib_seq_len <= 0:
        raise ValueError("--max-calib-seq-len 必须大于 0。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 FP8 量化实现需要 CUDA GPU，但未检测到可用 CUDA。")

    model_path = args.model_path.resolve()
    for mode in selected_modes(args.mode):
        target = output_path(args.output_root, mode)
        if target.resolve() == model_path:
            raise ValueError(f"输出目录不能与原始模型目录相同：{target}")
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f"输出目录已存在且非空：{target}\n"
                "请移走旧结果或通过 --output-root 指定新目录。"
            )


def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def quantize_one(args, tokenizer, mode):
    target = output_path(args.output_root, mode)
    quant_method = QUANT_METHODS[mode]

    print(f"[info] start {mode} FP8 quantization")
    print(f"[info] quant method: {quant_method}")
    print(f"[info] output path: {target}")

    # Each mode must start from a fresh copy because quantize() replaces Linear modules.
    model = AutoQuantForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
        safetensors=True,
    )
    try:
        quant_config = {
            "quant_method": quant_method,
            "per_tensor": True,
            "modules_to_not_convert": ["lm_head"],
        }
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=DEFAULT_CALIB_TEXTS,
            max_calib_samples=args.max_calib_samples,
            max_calib_seq_len=args.max_calib_seq_len,
        )

        target.mkdir(parents=True, exist_ok=True)
        model.save_quantized(str(target))
        tokenizer.save_pretrained(str(target))
        print(f'[done] {mode} FP8 model saved at "{target}"')
    finally:
        release_model(model)


def main():
    args = parse_args()
    validate_args(args)

    print(f"[info] original model: {args.model_path}")
    print(f"[info] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[info] selected mode: {args.mode}")
    if "static" in selected_modes(args.mode):
        print(
            "[info] static calibration: "
            f"{args.max_calib_samples} samples x {args.max_calib_seq_len} tokens"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    for mode in selected_modes(args.mode):
        quantize_one(args, tokenizer, mode)


if __name__ == "__main__":
    main()
