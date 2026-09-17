from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer

# model_path = 'Qwen/Qwen2.5-14B-Instruct'
# quant_path = 'Qwen2.5-14B-Instruct-sq'
# model_path = 'facebook/opt-13b'
# quant_path = 'opt-13b-sq'
model_path = 'meta-llama/Llama-3.1-8B-Instruct' # 若无法访问huggingface，则在modelscope下载好模型，然后此处用本地模型路径
quant_path = 'Llama-3.1-8B-Instruct-sq'

quant_config = {"quant_method": "sq", "zero_point": True} 

# Load model
model = AutoQuantForCausalLM.from_pretrained(model_path, safetensors=True) 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')