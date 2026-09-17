# import sys
# sys.path.append("../")
from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer

# model_path = 'Qwen/Qwen2.5-14B-Instruct'
# quant_path = 'Qwen2.5-14B-Instruct-awq'
model_path = 'Qwen/Qwen3-8B' # 若无法访问huggingface，则在modelscope下载好模型，然后此处用本地模型路径
quant_path = 'Qwen3-8B-awq'
# model_path = 'Qwen/Qwen3-30B-A3B'
# quant_path = 'Qwen3-30B-A3B-awq'
# model_path = 'v2ray/DeepSeek-V3-1B-Test'
# quant_path = 'DeepSeek-V3-1B-Test-awq'
quant_config = {"quant_method": "awq", "zero_point": True, "q_group_size": 128, "w_bit": 4, "modules_to_not_convert": ["lm_head"]} 

# Load model
model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # 加载 tokenizer（分词器）

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')