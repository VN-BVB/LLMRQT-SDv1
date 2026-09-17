import os
import torch
import logging
from transformers import AutoConfig
from quant.nn_models import *
from .base import BaseModelForCausalLM

Quant_CAUSAL_LM_MODEL_MAP = {
    "llama": LlamaModelForCausalLM, 
    "opt": OptModelForCausalLM,
    "qwen2": Qwen2ModelForCausalLM,
    "qwen3": Qwen3ModelForCausalLM, 
    "qwen3_moe": Qwen3MoeModelForCausalLM, 
    "deepseek_v3": DeepseekV3ModelForCausalLM,
    "qwen2_distilled_r1": DeepseekR1DistillQwen2ModelForCausalLM,
    "llama4": Llama4MoeModelForCausalLM
}

def check_and_get_model_type(model_dir, trust_remote_code=True, **model_init_kwargs):
    config = AutoConfig.from_pretrained( # 返回模型路径中的基于config.json抽象的config对象
        model_dir, trust_remote_code=trust_remote_code, **model_init_kwargs
    )
    print("model type is ", config.model_type)
    if config.model_type not in Quant_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    if model_dir[:8] == "deepseek" and model_type == "qwen2":
        model_type = "qwen2_distilled_r1"
    return model_type


class AutoQuantForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "You must instantiate AutoQuantForCausalLM with\n"
            "AutoQuantForCausalLM.from_quantized or AutoQuantForCausalLM.from_pretrained, rather directly init this class"
        )

    @classmethod # 类方法
    def from_pretrained(
        self,
        model_path, # 只有这一个入参，其他都取默认值
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs, # 传递额外的模型初始化参数
    ) -> BaseModelForCausalLM:
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        return Quant_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            model_type,
            torch_dtype=torch_dtype, # 模型权重的数据类型
            trust_remote_code=trust_remote_code,#相信远端的code或模型
            safetensors=safetensors, # 模型权重格式
            device_map=device_map, # 指定模型加载的设备
            low_cpu_mem_usage=low_cpu_mem_usage, # 控制模型加载时是否尽量减少 CPU 内存占用。如果为 True，模型会分步加载权重到 GPU，避免一次性占用大量 CPU 内存，有利于小幅提升推理速度
            use_cache=use_cache, # 控制模型在生成文本时是否使用KV Cache加速推理
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        self,
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ) -> BaseModelForCausalLM:
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        return Quant_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            model_path,
            model_type,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            safetensors=safetensors,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_cache=use_cache,
            **model_init_kwargs,
        )
