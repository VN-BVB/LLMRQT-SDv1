import transformers
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    '/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5',
    trust_remote_code=True,
    torch_dtype='auto',
    use_safetensors=True,
)