#!/usr/bin/env python3
"""Interactive model selector and chat launcher for local Qwen3 models."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from runtime_refact.core.api import AutoQuantForCausalLM
from runtime_refact.utils.common_utils import get_best_device


DEFAULT_MODEL_ROOTS = (
    Path("/mnt/e/aiinfra/llmqt_awq/models"),
    Path("/mnt/e/aiinfra/llmqt_awq/quantized"),
    Path("/mnt/e/aiinfra/llmqt_fp8/quantized"),
)
SUPPORTED_MODEL_TYPES = {"qwen3"}
SUPPORTED_QUANT_METHODS = {
    "awq",
    "fp8_dynamic_quant",
    "fp8_static_quant",
    "smooth_quant",
}
QUANT_LABELS = {
    "original": "Original",
    "awq": "AWQ",
    "fp8_dynamic_quant": "FP8 Dynamic",
    "fp8_static_quant": "FP8 Static",
    "smooth_quant": "SmoothQuant",
}
QUANT_SORT_ORDER = {
    "original": 0,
    "awq": 1,
    "fp8_dynamic_quant": 2,
    "fp8_static_quant": 3,
    "smooth_quant": 4,
}


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    model_type: str
    quant_method: str
    bits: int | None
    torch_dtype: str
    size_label: str
    estimated_parameters: int | None

    @property
    def architecture_label(self) -> str:
        return "Qwen3" if self.model_type == "qwen3" else self.model_type

    @property
    def quant_label(self) -> str:
        label = QUANT_LABELS.get(self.quant_method, self.quant_method)
        if self.quant_method == "awq" and self.bits is not None:
            return f"{label} INT{self.bits}"
        return label

    @property
    def display_name(self) -> str:
        return f"{self.architecture_label} | {self.quant_label} | {self.size_label}"

    @property
    def is_quantized(self) -> bool:
        return self.quant_method != "original"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover, select, load, and chat with local Qwen3 models."
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Model directory. If omitted, an interactive model menu is displayed.",
    )
    parser.add_argument(
        "--model-root",
        action="append",
        type=Path,
        default=None,
        help="Additional/alternative root to scan; may be supplied more than once.",
    )
    parser.add_argument("--list", action="store_true", help="List models and exit.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable the Qwen3 thinking mode in the chat template.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than 0")
    if args.context_length <= args.max_new_tokens + 1:
        parser.error("--context-length must exceed --max-new-tokens by at least 2")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def estimate_qwen3_parameters(config: dict[str, Any]) -> int | None:
    """Estimate dense Qwen3 parameter count from architecture fields."""
    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    if any(config.get(key) is None for key in required):
        return None
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    attention_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    vocab = int(config["vocab_size"])
    head_dim = int(config.get("head_dim") or hidden // attention_heads)

    query_dim = attention_heads * head_dim
    kv_dim = kv_heads * head_dim
    attention = hidden * query_dim + 2 * hidden * kv_dim + query_dim * hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden + 2 * head_dim
    embeddings = vocab * hidden
    if not bool(config.get("tie_word_embeddings", False)):
        embeddings += vocab * hidden
    return embeddings + layers * (attention + mlp + norms) + hidden


def size_from_path(path: Path) -> str | None:
    matches = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])", path.name)
    return f"{matches[-1]}B" if matches else None


def format_size(parameter_count: int | None, path: Path) -> str:
    if parameter_count is not None:
        billions = parameter_count / 1_000_000_000
        if billions >= 0.1:
            return f"{billions:.1f}B"
        return f"{parameter_count / 1_000_000:.0f}M"
    return size_from_path(path) or "Unknown size"


def inspect_model(model_path: Path) -> ModelInfo:
    path = model_path.expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "unknown")).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"Unsupported model_type={model_type!r}; this launcher currently supports: "
            f"{supported}"
        )

    quant_config = config.get("quantization_config") or {}
    quant_method = str(quant_config.get("quant_method") or "original").lower()
    if quant_method not in SUPPORTED_QUANT_METHODS and quant_method != "original":
        raise ValueError(f"Unsupported quant_method={quant_method!r}: {path}")
    bits_value = quant_config.get("bits")
    bits = int(bits_value) if bits_value is not None else None
    default_dtype = "float16" if quant_method == "awq" else "bfloat16"
    config_dtype = str(config.get("torch_dtype") or default_dtype).lower()
    torch_dtype = config_dtype if config_dtype in {"float16", "bfloat16"} else default_dtype
    parameters = estimate_qwen3_parameters(config)
    return ModelInfo(
        path=path,
        model_type=model_type,
        quant_method=quant_method,
        bits=bits,
        torch_dtype=torch_dtype,
        size_label=format_size(parameters, path),
        estimated_parameters=parameters,
    )


def discover_models(roots: list[Path]) -> tuple[list[ModelInfo], list[str]]:
    models: list[ModelInfo] = []
    skipped: list[str] = []
    visited: set[Path] = set()
    for root_value in roots:
        root = root_value.expanduser().resolve()
        if not root.is_dir():
            skipped.append(f"scan root does not exist: {root}")
            continue
        for config_path in sorted(root.rglob("config.json")):
            model_path = config_path.parent.resolve()
            if model_path in visited:
                continue
            visited.add(model_path)
            try:
                models.append(inspect_model(model_path))
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as error:
                skipped.append(f"{model_path}: {error}")
    models.sort(
        key=lambda item: (
            item.model_type,
            item.estimated_parameters or 0,
            QUANT_SORT_ORDER.get(item.quant_method, 99),
            str(item.path),
        )
    )
    return models, skipped


def print_models(models: list[ModelInfo]) -> None:
    print("\n" + "=" * 78)
    print("Available Qwen3 models")
    print("=" * 78)
    for index, model in enumerate(models, 1):
        print(f"  [{index}] {model.display_name}")
        print(f"      {model.path}")
    if not models:
        print("  No supported models found.")


def choose_model(models: list[ModelInfo]) -> ModelInfo | None:
    while True:
        print_models(models)
        print("\n  [P] Enter another model path")
        print("  [Q] Exit")
        try:
            choice = input("\nSelect a model: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        if choice.lower() in {"p", "path"}:
            try:
                custom_path = input("Model directory: ").strip()
                return inspect_model(Path(custom_path))
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as error:
                print(f"Invalid model: {error}")
                continue
        try:
            index = int(choice)
        except ValueError:
            print("Please enter a model number, P, or Q.")
            continue
        if 1 <= index <= len(models):
            return models[index - 1]
        print(f"Please enter a number between 1 and {len(models)}.")


def load_model(model_info: ModelInfo, context_length: int) -> tuple[Any, Any, torch.device]:
    if model_info.is_quantized and not torch.cuda.is_available():
        raise RuntimeError("LLMQRT quantized models require a CUDA GPU")
    dtype = getattr(torch, model_info.torch_dtype)
    print("\n" + "=" * 78)
    print(f"Loading {model_info.display_name}")
    print("=" * 78)
    print(f"Path:         {model_info.path}")
    print(f"Loader route: {'LLMQRT quantized' if model_info.is_quantized else 'Transformers original'}")
    print(f"Dtype:        {model_info.torch_dtype}")
    print(f"Context:      {context_length}")

    if model_info.is_quantized:
        model = AutoQuantForCausalLM.from_quantized(
            str(model_info.path),
            torch_dtype=dtype,
            device_map="auto",
            batch_size=1,
            max_seq_len=context_length,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_info.path),
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_info.path),
        trust_remote_code=True,
    )
    model.to(get_best_device())
    model.eval()
    device = torch.device(get_best_device())
    print(f"Loaded successfully on {device}.")
    return model, tokenizer, device


def build_chat_inputs(
    tokenizer: Any,
    messages: list[dict[str, str]],
    device: torch.device,
    max_input_tokens: int,
    thinking: bool,
) -> tuple[Any, int]:
    tokenizer.truncation_side = "left"
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=thinking,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    return inputs, int(inputs["input_ids"].shape[-1])


@torch.inference_mode()
def generate_reply_streaming(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    max_input_tokens = args.context_length - args.max_new_tokens
    inputs, input_length = build_chat_inputs(
        tokenizer,
        messages,
        device,
        max_input_tokens,
        args.thinking,
    )
    do_sample = args.temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    outputs = model.generate(**inputs, streamer=streamer, **generation_kwargs)
    generated_ids = outputs[0, input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def chat_loop(
    model_info: ModelInfo,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print("\n" + "=" * 78)
    print(f"Chat: {model_info.display_name}")
    print("Commands: /clear clears history, /back returns to menu, /exit quits")
    print("=" * 78)
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "exit"
        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            return "exit"
        if command in {"/back", "/models"}:
            return "back"
        if command == "/clear":
            messages.clear()
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            print("Conversation history cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        print("Assistant> ", end="", flush=True)
        try:
            reply = generate_reply_streaming(
                model,
                tokenizer,
                device,
                messages,
                args,
            )
        except RuntimeError as error:
            messages.pop()
            print(f"\nGeneration failed: {error}")
            if "out of memory" in str(error).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        messages.append({"role": "assistant", "content": reply})


def clear_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    roots = args.model_root if args.model_root is not None else list(DEFAULT_MODEL_ROOTS)
    models, skipped = discover_models(roots)
    if skipped:
        print(f"Skipped {len(skipped)} unsupported or invalid path(s). Use --list to inspect models.")

    if args.list:
        print_models(models)
        if skipped:
            print("\nSkipped paths:")
            for reason in skipped:
                print(f"  - {reason}")
        return 0

    selected_once = args.model is not None
    direct_model = inspect_model(args.model) if args.model is not None else None
    while True:
        model_info = direct_model if direct_model is not None else choose_model(models)
        if model_info is None:
            return 0
        try:
            model, tokenizer, device = load_model(model_info, args.context_length)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"\nModel loading failed: {error}")
            if selected_once:
                return 1
            continue
        action = chat_loop(model_info, model, tokenizer, device, args)
        del model
        del tokenizer
        clear_device_memory()
        if action == "exit" or selected_once:
            return 0
        direct_model = None


if __name__ == "__main__":
    raise SystemExit(main())
