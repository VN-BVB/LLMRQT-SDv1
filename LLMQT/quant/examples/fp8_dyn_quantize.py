from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer

# model_path = 'Qwen/Qwen2.5-14B-Instruct'
# quant_path = 'Qwen2.5-14B-Instruct-fp8-dyn'
# model_path = 'Qwen/Qwen3-8B'
# quant_path = 'Qwen3-8B-dyn'
# model_path = 'Qwen/Qwen3-30B-A3B'
# quant_path = 'Qwen3-30B-A3B-dyn'
# model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
# quant_path = 'DeepSeek-R1-Distill-Qwen-32B-dyn'
model_path = 'meta-llama/Llama-4-Scout-17B-16E' # 若无法访问huggingface，则在modelscope下载好模型，然后此处用本地模型路径
quant_path = 'Llama-4-Scout-17B-16E-dyn'
quant_config = {"quant_method": "fp8_dynamic_quant"} 

# Load model
model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)#
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')