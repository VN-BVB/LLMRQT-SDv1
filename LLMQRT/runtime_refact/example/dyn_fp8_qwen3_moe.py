import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用的GPU编号

from runtime_refact.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer, TextStreamer
from runtime_refact.utils.common_utils import get_best_device
import torch

qmodel_path = '/home/LLMQT/quant/examples/Qwen3-30B-A3B-dyn'

model = AutoQuantForCausalLM.from_quantized(
  qmodel_path,
  torch_dtype=torch.bfloat16, # bf16 or fp16
  device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(qmodel_path, trust_remote_code=True)

# fwd
device = get_best_device()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = [
  {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
  {"role": "user", "content": \
        "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"
        },
]

inputs = tokenizer.apply_chat_template(
  prompt,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors="pt",
  return_dict=True,
  enable_thinking=False
).to(device)

model.to(device)

outputs = model.generate(
    **inputs,
    do_sample=True,
    max_new_tokens=256,
    streamer=streamer,
)
