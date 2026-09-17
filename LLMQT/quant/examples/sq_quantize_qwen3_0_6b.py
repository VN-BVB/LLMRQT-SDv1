"""使用 SmoothQuant 将本地 Qwen3-0.6B 量化为 W8A8 模型。"""

import argparse
import os
import sys
from pathlib import Path

# 复用之前 0.6B 示例已下载的 Hugging Face 缓存。
os.environ.setdefault("HF_HOME", r"E:\aiinfra\hf_home")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoTokenizer


# 允许直接从仓库根目录或 examples 目录运行本脚本。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.api import AutoQuantForCausalLM


DEFAULT_MODEL_PATH = Path(r"E:\aiinfra\llmqt_awq\models\Qwen--Qwen3-0.6B")
DEFAULT_OUTPUT_PATH = Path(r"E:\aiinfra\llmqt_sq\quantized\Qwen3-0.6B-sq")


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 LLMQT SmoothQuant 量化本地 Qwen3-0.6B（W8A8）"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"原始 Qwen3-0.6B 模型目录（默认：{DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"量化模型保存目录（默认：{DEFAULT_OUTPUT_PATH}）",
    )
    parser.add_argument(
        "--calib-data",
        default="pileval",
        help="Hugging Face 校准数据集名称（默认：pileval）",
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=128,
        help="校准样本数（默认：128）",
    )
    parser.add_argument(
        "--max-calib-seq-len",
        type=int,
        default=512,
        help="每条校准样本的最大 token 数（默认：512）",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.model_path.is_dir():
        raise FileNotFoundError(
            f"找不到原始模型目录：{args.model_path}\n"
            "请通过 --model-path 指定 Qwen3-0.6B 的本地目录。"
        )
    if args.model_path.resolve() == args.output_path.resolve():
        raise ValueError("--output-path 不能与 --model-path 相同，以免覆盖原始模型。")
    if args.max_calib_samples <= 0:
        raise ValueError("--max-calib-samples 必须大于 0。")
    if args.max_calib_seq_len <= 0:
        raise ValueError("--max-calib-seq-len 必须大于 0。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 SQ 校准流程需要 CUDA GPU，但未检测到可用的 CUDA。")


def main():
    args = parse_args()
    validate_args(args)

    print(f"[info] original model: {args.model_path}")
    print(f"[info] quantized model: {args.output_path}")
    print(f"[info] calibration data: {args.calib_data}")
    print(f"[info] CUDA device: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    model = AutoQuantForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
        safetensors=True,
    )

    quant_config = {
        "quant_method": "sq",
        "w_bit": 8,
        "zero_point": True,
        "modules_to_not_convert": ["lm_head"],
    }
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=args.calib_data,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.max_calib_seq_len,
    )

    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(args.output_path))
    tokenizer.save_pretrained(str(args.output_path))
    print(f'[done] SmoothQuant model saved at "{args.output_path}"')


if __name__ == "__main__":
    main()
