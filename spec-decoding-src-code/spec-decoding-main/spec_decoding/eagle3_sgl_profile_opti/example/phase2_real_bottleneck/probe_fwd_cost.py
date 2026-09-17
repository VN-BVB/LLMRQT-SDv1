#!/usr/bin/env python3
"""微基准：target 前向墙钟随新 token 数的变化（测试 eager 瓶颈是否为launch-bound，如果是，则使用cuda graph）。"""
import time

import torch
from transformers import AutoModelForCausalLM

mid = "NousResearch/Meta-Llama-3.1-8B-Instruct"
dev, dt = "cuda", torch.float16
m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).to(dev).eval()

with torch.inference_mode():
    base = torch.randint(0, 32000, (1, 512), device=dev)
    print("target forward wall (ms): prefix=512 in KV, decode N new tokens (best of 15)")
    for n in [1, 2, 3, 4, 8, 16, 32, 64]:
        best = 1e9
        for _ in range(15):
            oo = m(base, use_cache=True)
            nt = torch.randint(0, 32000, (1, n), device=dev)
            am = torch.ones((1, 512 + n), device=dev, dtype=torch.long)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = m(nt, past_key_values=oo.past_key_values, use_cache=True, attention_mask=am)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        print(f"  N={n:>3}: {best:6.2f} ms")
