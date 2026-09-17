import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用的GPU编号

from runtime_refact.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer
from runtime_refact.common_utils import get_best_device
import torch

qmodel_path = 'Qwen3-0.5B-Instruct-awq'
# quant_config = {"quant_method": "awq", "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model并且把customizedLinear replace nn.Linear,然后调用HF API generate即可
# 无需额外传入quant config，qmodel_path里面有quant config(config.json)，from_quantized函数会读取它然后parse出对应的quant method，由此拿到对应的linear
model = AutoQuantForCausalLM.from_quantized(
  qmodel_path,
  torch_dtype=torch.bfloat16,
  device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(qmodel_path, trust_remote_code=True)

# fwd
device = get_best_device()
# model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = [
  {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
  {"role": "user", "content": \
        "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"},
]

inputs = tokenizer.apply_chat_template(
  prompt,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors="pt",
  return_dict=True,
).to(device)

model.to(device)

outputs = model.generate(
    **inputs,
    do_sample=True,
    max_new_tokens=256,
    streamer=streamer,
)
