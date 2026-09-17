import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\aiinfra\hf_home")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant.core.api import AutoQuantForCausalLM
from awq_quantize_and_run import model_dir_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", default=r"E:\aiinfra\llmqt_awq")
    parser.add_argument("--model-kind", choices=["original", "quantized"], default="quantized")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--system-prompt",
        default="你是一个大语言模型量化助手。回答 AWQ 时，默认解释 Activation-aware Weight Quantization。",
    )
    return parser.parse_args()


def build_prompt(tokenizer, messages):
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
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def load_model(args, device):
    work_dir = Path(args.work_dir)
    model_dir = work_dir / "models" / model_dir_name(args.model_id)
    quant_dir = work_dir / "quantized" / f"{model_dir_name(args.model_id)}-awq"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    if args.model_kind == "original":
        print(f"[info] loading original model from {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()
    else:
        print(f"[info] loading quantized model from {quant_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(quant_dir), trust_remote_code=True)
        model = AutoQuantForCausalLM.from_quantized(
            str(quant_dir),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
    print("[info] model is ready")

    return model, tokenizer


@torch.no_grad()
def chat_once(model, tokenizer, messages, device, max_new_tokens, temperature):
    text = build_prompt(tokenizer, messages)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args, device)
    messages = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []

    print(f"[info] model_kind: {args.model_kind}")
    print(f"[info] device: {device}")
    print("[info] 输入 exit 退出。")

    while True:
        user_text = input("\n你: ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        answer = chat_once(
            model,
            tokenizer,
            messages,
            device,
            args.max_new_tokens,
            args.temperature,
        )
        print(f"模型: {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
