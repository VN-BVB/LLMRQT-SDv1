from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer
# model_path = 'meta-llama/Llama-3.1-8B-Instruct'
# quant_path = 'Llama-3.1-8B-Instruct-fp8-static'
model_path = 'Qwen/Qwen3-8B' # 若无法访问huggingface，则在modelscope下载好模型，然后此处用本地模型路径
quant_path = 'Qwen3-8B-fp8-static'
# model_path = 'Qwen/Qwen2.5-14B-Instruct'
# quant_path = 'Qwen2.5-14B-Instruct-fp8-static'
# model_path = 'Qwen/Qwen3-30B-A3B'
# quant_path = 'Qwen3-30B-A3B-static'
#model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
#quant_path = 'DeepSeek-R1-Distill-Qwen-32B-static'
quant_config = {"quant_method": "fp8_static_quant"}

# Load model
model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
