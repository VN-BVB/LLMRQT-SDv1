import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\aiinfra\hf_home")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from quant.core.api import AutoQuantForCausalLM


DEFAULT_CALIB_TEXTS = [
    "Large language models predict the next token from context and can solve many language tasks.",
    "Quantization reduces model weight precision to lower memory use while preserving useful behavior.",
    "AWQ searches activation-aware scales before packing weights into four-bit groups.",
    "A small calibration set is enough for this local smoke test of the quantization pipeline.",
    "The model should answer concise questions after the quantized weights are loaded from disk.",
    "Transformers models use attention layers, feed-forward layers, embeddings, and normalization.",
    "This sentence adds more tokens so calibration can form at least one complete block.",
    "Running on a consumer GPU needs a compact model and short calibration sequences.",
] * 4


def parse_args():
    parser = argparse.ArgumentParser(description="下载、AWQ 量化并测试运行大语言模型")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face 模型仓库 ID（默认：Qwen/Qwen3-0.6B）",
    )
    parser.add_argument(
        "--work-dir",
        default=r"E:\aiinfra\llmqt_awq",
        help="模型下载、量化结果等文件的工作目录",
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=16,
        help="用于 AWQ 校准的最大文本样本数（默认：16）",
    )
    parser.add_argument(
        "--max-calib-seq-len",
        type=int,
        default=128,
        help="每个校准样本的最大 Token 序列长度（默认：128）",
    )
    parser.add_argument(
        "--calib-data",
        default="builtin",
        help='校准数据源："builtin" 使用脚本内置文本，"pileval" 下载 mit-han-lab/pile-val-backup 数据集',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="模型测试生成时允许生成的最大新 Token 数（默认：32）",
    )
    parser.add_argument(
        "--prompt",
        default="用一句话介绍 AWQ 量化。",
        help="量化模型生成测试所使用的输入提示词",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过模型下载，直接使用工作目录中已有的模型文件",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="跳过量化过程，直接加载工作目录中已有的量化模型",
    )
    return parser.parse_args()


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def download_model(model_id: str, local_model_dir: Path, skip_download: bool):
    if skip_download and local_model_dir.exists():
        return local_model_dir

    local_model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*msgpack*", "*h5*", "optimizer.pt", "*.onnx*"],
    )
    return local_model_dir


def build_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    text = build_prompt(tokenizer, prompt)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    local_model_dir = work_dir / "models" / model_dir_name(args.model_id)
    quant_dir = work_dir / "quantized" / f"{model_dir_name(args.model_id)}-awq"

    print(f"[info] model id: {args.model_id}")
    print(f"[info] local model dir: {local_model_dir}")
    print(f"[info] quantized dir: {quant_dir}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")

    local_model_dir = download_model(args.model_id, local_model_dir, args.skip_download)
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), trust_remote_code=True)

    if not args.skip_quantize:
        quant_config = {
            "quant_method": "awq",
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "modules_to_not_convert": ["lm_head"],
        }
        model = AutoQuantForCausalLM.from_pretrained(
            str(local_model_dir),
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=DEFAULT_CALIB_TEXTS if args.calib_data == "builtin" else args.calib_data,
            max_calib_samples=args.max_calib_samples,
            max_calib_seq_len=args.max_calib_seq_len,
            n_parallel_calib_samples=1,
            max_chunk_memory=128 * 1024 * 1024,
        )
        quant_dir.mkdir(parents=True, exist_ok=True)
        model.save_quantized(str(quant_dir))
        tokenizer.save_pretrained(str(quant_dir))
        print(f"[info] saved quantized model to {quant_dir}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    quantized_model = AutoQuantForCausalLM.from_quantized(
        str(quant_dir),
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    quantized_tokenizer = AutoTokenizer.from_pretrained(
        str(quant_dir),
        trust_remote_code=True,
    )
    output = generate_once(
        quantized_model,
        quantized_tokenizer,
        args.prompt,
        args.max_new_tokens,
    )
    print("[result]")
    print(output)


if __name__ == "__main__":
    main()
