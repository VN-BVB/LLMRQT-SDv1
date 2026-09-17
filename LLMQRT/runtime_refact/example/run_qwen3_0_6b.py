import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用的GPU编号

import torch
from transformers import AutoTokenizer, TextStreamer

from runtime_refact.core.api import AutoQuantForCausalLM
from runtime_refact.utils.common_utils import get_best_device


qmodel_path = '/home/lyc/workspace/week78/LLMQRT/models/Qwen3-0.6B-awq'

# 读取 AWQ 模型，并替换成 LLMQRT 的量化 Linear
model = AutoQuantForCausalLM.from_quantized(
    qmodel_path,
    torch_dtype=torch.float16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(qmodel_path, trust_remote_code=True)

device = get_best_device()
model.to(device)

prompt = [
    {'role': 'user', 'content': 'What is 2 + 2? Answer briefly.'},
]

inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors='pt',
    return_dict=True,
    enable_thinking=False,
).to(device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

with torch.no_grad():
    model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=20,
        streamer=streamer,
    )

