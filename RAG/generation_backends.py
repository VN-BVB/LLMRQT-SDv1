"""RAG 的两种生成后端。

1. VLLMClient：通过 HTTP 调用已经启动的 vLLM 服务；
2. LocalAWQClient：不启动服务，直接用 LLMQRT 加载用户自己的 AWQ checkpoint。

两个类都提供相同的 chat(messages, max_tokens, temperature) 接口，因此检索和
Prompt 代码不需要因为推理后端不同而复制两份。
"""

from __future__ import annotations

import gc
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parent
DEFAULT_AWQ_MODEL = Path(
    "/home/lyc/workspace/lastdata/eagle3/"
    "qwen3_1_7b_awq_eagle3_benchmark/checkpoints/Qwen3-1.7B-W4A16-AWQ"
)
DEFAULT_LLMQRT_ROOT = Path("/home/lyc/workspace/week78/LLMQRT")


class ChatBackend(Protocol):
    """只要实现这个方法，就可以作为 RAG 的生成后端。"""

    model: str

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str: ...

    def chat_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_concurrency: int = 1,
    ) -> list[str]: ...

    def unload(self) -> None: ...


class VLLMClient:
    """通过 OpenAI 兼容 HTTP API 调用 vLLM。"""

    def __init__(self, api_base: str, model: str | None = None, api_key: str = "EMPTY") -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        # 开发机常配置 HTTP_PROXY。127.0.0.1 请求不应绕到代理，否则可能得到 502。
        hostname = urlparse(self.api_base).hostname
        if hostname in {"127.0.0.1", "localhost", "::1"}:
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        else:
            self._opener = urllib.request.build_opener()
        self.model = model or self._discover_model()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.api_base + path,
            data=body,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with self._opener.open(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM 返回 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"无法连接 vLLM：{self.api_base}。请先按照 README 启动服务。"
            ) from exc

    def _discover_model(self) -> str:
        response = self._request("GET", "/models")
        models = response.get("data", [])
        if not models:
            raise RuntimeError("vLLM /models 没有返回模型，请用 --model 显式指定")
        return models[0]["id"]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        current_max_tokens = max_tokens
        while True:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": current_max_tokens,
                "temperature": temperature,
                # Qwen3 默认可能先输出很长的思考内容。RAG 短答案评测关闭 thinking，
                # 否则 64/128 token 很可能全消耗在 <think> 中。
                "chat_template_kwargs": {"enable_thinking": False},
            }
            try:
                response = self._request("POST", "/chat/completions", payload)
                break
            except RuntimeError as exc:
                # 个别 OCR/LaTeX 页的字符/token 比异常。仅在服务明确报告
                # context overflow 时减小输出预算，其他 HTTP 错误照常抛出。
                if (
                    "maximum context length" not in str(exc)
                    or current_max_tokens <= 64
                ):
                    raise
                current_max_tokens = max(64, current_max_tokens // 2)
        return response["choices"][0]["message"]["content"].strip()

    def chat_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_concurrency: int = 8,
    ) -> list[str]:
        """并发发送请求，让 vLLM 服务端通过 continuous batching 合批。"""

        if max_concurrency <= 0:
            raise ValueError("max_concurrency 必须大于 0")
        if not messages_batch:
            return []

        def generate(messages: list[dict[str, str]]) -> str:
            return self.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        worker_count = min(max_concurrency, len(messages_batch))
        if worker_count == 1:
            return [generate(messages) for messages in messages_batch]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(generate, messages_batch))

    def unload(self) -> None:
        """HTTP 客户端不持有模型，无需释放显存。"""


class LocalAWQClient:
    """直接加载本地 LLMQRT W4A16 AWQ checkpoint，不经过 vLLM。

    这复用了现有 benchmark.py 已验证过的加载路径：
    `runtime_refact.core.api.AutoQuantForCausalLM.from_quantized`。
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_AWQ_MODEL,
        llmqrt_root: str | Path = DEFAULT_LLMQRT_ROOT,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> None:
        model_path = Path(model_path).expanduser().resolve()
        llmqrt_root = Path(llmqrt_root).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"AWQ checkpoint 不存在：{model_path}")
        if not llmqrt_root.is_dir():
            raise FileNotFoundError(f"LLMQRT 源码目录不存在：{llmqrt_root}")

        # LLMQRT 是工作区源码而不是 pip 包，所以显式加入模块搜索路径。
        if str(llmqrt_root) not in sys.path:
            sys.path.insert(0, str(llmqrt_root))

        try:
            import torch
            from runtime_refact.core.api import AutoQuantForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("无法导入 LLMQRT/torch/transformers，请使用项目的 vLLM Python 环境") from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("本地 AWQ CUDA kernel 需要可用的 NVIDIA GPU")
        if batch_size <= 0:
            raise ValueError("本地 AWQ batch_size 必须大于 0")

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = str(model_path)
        print(f"正在直接加载本地 AWQ 模型：{model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        wrapper = AutoQuantForCausalLM.from_quantized(
            str(model_path),
            max_seq_len=4096,
            torch_dtype=torch.float16,
            fuse_layers=False,
            device_map=device,
            batch_size=batch_size,
        )
        self._model = wrapper.model.eval()
        self._model.config._attn_implementation = "sdpa"
        self._model.config.use_cache = True

    def _next_token(self, logits: Any, temperature: float) -> Any:
        """temperature=0 做确定性 greedy；大于 0 时做基础随机采样。"""

        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probabilities = self.torch.softmax(logits / temperature, dim=-1)
        return self.torch.multinomial(probabilities, num_samples=1)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """应用 Qwen3 对话模板，并用与 benchmark 相同的 KV-cache 循环生成。"""

        if max_tokens <= 0:
            return ""

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        eos_ids = self._model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids or [])

        generated = []
        with self.torch.inference_mode():
            first = self._model(input_ids, use_cache=True, return_dict=True)
            token = self._next_token(first.logits[:, -1], temperature)
            past_key_values = first.past_key_values

            for step in range(max_tokens):
                token_id = int(token[0, 0])
                if token_id in eos_ids:
                    break
                generated.append(token_id)
                if step + 1 >= max_tokens:
                    break
                output = self._model(
                    token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = output.past_key_values
                token = self._next_token(output.logits[:, -1], temperature)

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_concurrency: int = 1,
    ) -> list[str]:
        """使用一次 ``generate`` 同时处理多个 Prompt，提升离线 GPU 利用率。"""

        del max_concurrency  # 本地模型依靠 batch 并行，不能由多个线程同时调用。
        if not messages_batch:
            return []
        if max_tokens <= 0:
            return [""] * len(messages_batch)

        results: list[str] = []
        for start in range(0, len(messages_batch), self.batch_size):
            current = messages_batch[start : start + self.batch_size]
            # batch=1 时继续复用已经验证过的手写 KV-cache 路径。
            if len(current) == 1:
                results.append(
                    self.chat(
                        current[0],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                continue

            prompts = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for messages in current
            ]
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            tokens = self.tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature

            with self.torch.inference_mode():
                outputs = self._model.generate(**tokens, **generation_kwargs)
            prompt_width = tokens["input_ids"].shape[1]
            generated = outputs[:, prompt_width:]
            results.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            )
        return results

    def unload(self) -> None:
        """释放本地生成模型，为后续 Embedding 建库腾出 GPU 显存。"""

        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if self.device.type == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def create_backend(
    backend: str,
    *,
    api_base: str,
    model: str | None,
    awq_model: str | Path,
    llmqrt_root: str | Path,
    local_batch_size: int = 1,
) -> ChatBackend:
    """根据命令行选项创建后端，集中处理分支。"""

    if backend == "vllm":
        return VLLMClient(api_base, model)
    if backend == "local-awq":
        return LocalAWQClient(awq_model, llmqrt_root, batch_size=local_batch_size)
    raise ValueError(f"未知生成后端：{backend}")
