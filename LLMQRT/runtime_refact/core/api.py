import os
import torch
import logging
from transformers import AutoConfig
from runtime_refact.nn_models.models.qwen2 import Qwen2ModelForCausalLM
from runtime_refact.nn_models.models.qwen3 import Qwen3ModelForCausalLM
from runtime_refact.nn_models.models.qwen3_moe import Qwen3MoeModelForCausalLM
from runtime_refact.nn_models.models.llama import LlamaModelForCausalLM
from runtime_refact.nn_models.models.opt import OptModelForCausalLM
from runtime_refact.nn_models import * # 有时用这个不行，我也不知道为啥，只有用上面那个
from .base import BaseModelForCausalLM


QUANT_CAUSAL_LM_MODEL_MAP = {
    "llama": LlamaModelForCausalLM,
    "qwen2": Qwen2ModelForCausalLM,
    "opt": OptModelForCausalLM,
    "qwen3": Qwen3ModelForCausalLM,
    "qwen3_moe": Qwen3MoeModelForCausalLM,
}


def check_and_get_model_type(model_dir, trust_remote_code=True, **model_init_kwargs):
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, **model_init_kwargs
    )
    print("model type is ", config.model_type)
    if config.model_type not in QUANT_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


class AutoQuantForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "You must instantiate AutoQuantForCausalLM with\n"
            "AutoQuantForCausalLM.from_quantized or AutoQuantForCausalLM.from_pretrained"
        )

    @classmethod
    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        download_kwargs=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ) -> BaseModelForCausalLM:
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        return QUANT_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            model_type,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            safetensors=safetensors,
            device_map=device_map,
            download_kwargs=download_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_cache=use_cache,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        self,
        quant_path,
        max_seq_len=2048,
        trust_remote_code=True,
        fuse_layers=True,
        batch_size=1,
        safetensors=True,
        device_map="balanced",
        max_memory=None,
        offload_folder=None,
        download_kwargs=None,
        **config_kwargs,
    ) -> BaseModelForCausalLM:
        os.environ["AWQ_BATCH_SIZE"] = str(batch_size)
        model_type = check_and_get_model_type(quant_path, trust_remote_code)

        if config_kwargs.get("max_new_tokens") is not None:
            max_seq_len = config_kwargs["max_new_tokens"]
            logging.warning(
                "max_new_tokens argument is deprecated... gracefully "
                "setting max_seq_len=max_new_tokens."
            )

        return QUANT_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path,
            model_type,
            max_seq_len,
            trust_remote_code=trust_remote_code,
            fuse_layers=fuse_layers,
            safetensors=safetensors,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )
