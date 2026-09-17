import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\aiinfra\hf_home")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant.core.api import AutoQuantForCausalLM
from awq_quantize_and_run import DEFAULT_CALIB_TEXTS, model_dir_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", default=r"E:\aiinfra\llmqt_awq")
    parser.add_argument("--model-kind", choices=["original", "quantized", "both"], default="both")
    parser.add_argument("--eval-data", default="pileval", choices=["pileval", "builtin"])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    return parser.parse_args()


def load_eval_texts(eval_data, num_samples):
    if eval_data == "builtin":
        return DEFAULT_CALIB_TEXTS[:num_samples]

    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", revision="main")
    texts = []
    for item in dataset:
        text = item["text"].strip()
        if text:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts


def build_blocks(tokenizer, texts, seq_len, device):
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
    ids = ids[: len(ids) // seq_len * seq_len]
    if not ids:
        raise ValueError("No full evaluation block was produced. Reduce --seq-len or increase --num-samples.")
    input_ids = torch.tensor(ids, dtype=torch.long).view(-1, seq_len).to(device)
    return input_ids


@torch.no_grad()
def perplexity(model, tokenizer, texts, seq_len, device):
    input_ids = build_blocks(tokenizer, texts, seq_len, device)
    losses = []
    for block in input_ids:
        block = block.unsqueeze(0)
        outputs = model(block, labels=block)
        losses.append(outputs.loss.detach().float())
    mean_loss = torch.stack(losses).mean().item()
    return math.exp(mean_loss), mean_loss, input_ids.shape[0]


def load_original(model_dir, device):
    return AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()


def load_quantized(quant_dir, device):
    return AutoQuantForCausalLM.from_quantized(
        str(quant_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    model_dir = work_dir / "models" / model_dir_name(args.model_id)
    quant_dir = work_dir / "quantized" / f"{model_dir_name(args.model_id)}-awq"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = load_eval_texts(args.eval_data, args.num_samples)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print(f"[info] eval_data: {args.eval_data}")
    print(f"[info] num_samples: {len(texts)}")
    print(f"[info] seq_len: {args.seq_len}")
    print(f"[info] device: {device}")

    if args.model_kind in ["original", "both"]:
        model = load_original(model_dir, device)
        ppl, loss, blocks = perplexity(model, tokenizer, texts, args.seq_len, device)
        print(f"[original] blocks={blocks} loss={loss:.6f} ppl={ppl:.6f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model_kind in ["quantized", "both"]:
        model = load_quantized(quant_dir, device)
        ppl, loss, blocks = perplexity(model, tokenizer, texts, args.seq_len, device)
        print(f"[quantized] blocks={blocks} loss={loss:.6f} ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
