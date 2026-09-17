"""Run an LLMQT FP8 checkpoint for one-shot or interactive inference."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from quant.core.api import AutoQuantForCausalLM


DEFAULT_MODEL_PATH = Path(
    r"E:\aiinfra\llmqt_fp8\quantized\Qwen3-0.6B-fp8-dynamic"
)


def build_prompt(tokenizer, messages):
    if tokenizer.chat_template:
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
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def load_fp8_model(model_path, device="cuda:0"):
    if not torch.cuda.is_available():
        raise RuntimeError("The current LLMQT FP8 runtime requires a CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    model = AutoQuantForCausalLM.from_quantized(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        use_cache=True,
    )
    return model, tokenizer


@torch.inference_mode()
def generate_once(model, tokenizer, messages, device, max_new_tokens, temperature):
    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    started = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_tokens = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), len(new_tokens), elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="请用一句话解释什么是模型量化。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda:0"
    started = time.perf_counter()
    model, tokenizer = load_fp8_model(args.model_path, device)
    print(f"[info] loaded in {time.perf_counter() - started:.2f}s")

    # Build the lazy FP8-rounded weight cache before accepting real requests.
    warmup = tokenizer("你好", return_tensors="pt").to(device)
    with torch.inference_mode():
        model(**warmup)
    torch.cuda.synchronize()
    print("[info] warm-up complete")

    messages = []
    pending = args.prompt
    while True:
        if args.interactive:
            pending = input("\n用户: ").strip()
            if pending.lower() in {"exit", "quit", "q"}:
                break
            if not pending:
                continue
        messages.append({"role": "user", "content": pending})
        answer, token_count, elapsed = generate_once(
            model,
            tokenizer,
            messages,
            device,
            args.max_new_tokens,
            args.temperature,
        )
        print(f"模型: {answer}")
        print(f"[info] {token_count} tokens in {elapsed:.3f}s ({token_count / elapsed:.2f} tok/s)")
        messages.append({"role": "assistant", "content": answer})
        if not args.interactive:
            break


if __name__ == "__main__":
    main()
