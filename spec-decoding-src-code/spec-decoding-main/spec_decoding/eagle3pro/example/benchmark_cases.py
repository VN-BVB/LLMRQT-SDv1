"""Shared deterministic prompts for EAGLE3Pro/vLLM comparison benchmarks."""

from __future__ import annotations


BASE_PROMPTS = (
    "请用中文解释投机解码为什么能够保持输出无损，并给出一个简单例子。",
    "Write a short Python function that checks whether a string is a palindrome, then explain it.",
    "Solve step by step: A shop discounts an 800 yuan item by 15%, then applies a 5% coupon. What is the final price?",
    "请比较线程、进程和协程的区别，并分别给出一个适用场景。",
    "Explain why the sky appears blue during the day and red near sunset.",
    "A train travels 180 km in 2.5 hours. Compute its average speed and explain each step.",
    "请写一个 Python 二分查找函数，并分析它的时间复杂度。",
    "Summarize the main trade-offs between latency and throughput in an inference server.",
)


def select_prompts(count: int) -> list[str]:
    """Return ``count`` stable prompts, adding a case tag after one full cycle."""
    if count < 1:
        raise ValueError("num_prompts must be >= 1")
    prompts: list[str] = []
    for index in range(count):
        prompt = BASE_PROMPTS[index % len(BASE_PROMPTS)]
        cycle = index // len(BASE_PROMPTS)
        prompts.append(prompt if cycle == 0 else f"[case {index}] {prompt}")
    return prompts
