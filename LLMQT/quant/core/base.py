import os
import warnings
import torch
import transformers
import torch.nn as nn

from tqdm import tqdm
from typing import List, Union, Dict
from typing_extensions import Doc, Annotated
from huggingface_hub import snapshot_download, save_torch_state_dict
from safetensors.torch import load_file as safe_load_file
from transformers.trainer_utils import load_sharded_checkpoint


from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
)

from .config import QuantConfig # both need in quantizer and runtime
from quant.quantization import get_concrete_quantizer_cls
from quant.nn_models.modules.linear import get_concrete_linear_module
from quant.utils.common_utils import (
    exclude_layers_to_not_quantize,
    get_named_linears,
    set_op_by_name,
)

TRANSFORMERS_AUTO_MAPPING_DICT = { # if you want support VLM or other non-CausalLM model, add the class here
    "llama": "AutoModelForCausalLM",
    "opt": "AutoModelForCausalLM",
    "qwen2": "AutoModelForCausalLM",
    "qwen3": "AutoModelForCausalLM",
    "qwen3_moe": "AutoModelForCausalLM",
    "deepseek_v3": "AutoModelForCausalLM",
    "qwen2_distilled_r1": "AutoModelForCausalLM",
    "llama4": "AutoModelForCausalLM",
}


class BaseModelForCausalLM(nn.Module):
    def __init__(
        self,
        model, # The pretrained or quantized model
        model_type, # The model type, found in config.json
        is_quantized, # Indicates if the current model is quantized
        config, # The config of the model
        quant_config, # The quantization config of the model
    ):
        """The base model for all models."""
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: QuantConfig = quant_config

    # self.model to到device
    def to(self, device: Annotated[str, Doc("The device to move your model to.")]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    # torch.nn.Module的fwd方法
    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    # wrapper HF的generate方法
    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(
        self,
        model_path,
        model_type,
        torch_dtype = torch.float16,
        trust_remote_code = True,
        safetensors = True,
        device_map = "auto",
        low_cpu_mem_usage = True,
        use_cache = False,
        **model_init_kwargs,
    ):
        """A method for initialization of pretrained models, usually in FP16."""
        # Get weights path and quant config by AutoConfig and QuantConfig
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            "",
            safetensors,
            trust_remote_code=trust_remote_code,
        )
        
        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name) # 动态获取transformers里面的target_cls_name类
        print("target cls name is ", target_cls_name)
        if model_init_kwargs.get("low_cpu_mem_usage") is None:
            model_init_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if model_init_kwargs.get("use_cache") is None and model_type != "llama4" and not ((target_cls_name == "AutoModelForVision2Seq") or (target_cls_name == "AutoModelForTextToWaveform")):
            model_init_kwargs["use_cache"] = use_cache

        # torch.nn.Module
        model = target_cls.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs,
        )

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=False,
            config=config,
            quant_config=quant_config,
        )

    @classmethod
    def from_quantized(
        self,
        model_path,
        model_type,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ):
        """Load a model saved by save_quantized()."""
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            "",
            safetensors,
            trust_remote_code=trust_remote_code,
        )

        supported_methods = {
            "awq",
            "fp8_dynamic_quant",
            "fp8_static_quant",
            "fp8_native_dynamic_quant",
            "fp8_native_static_quant",
        }
        if quant_config.quant_method not in supported_methods:
            raise NotImplementedError(
                "from_quantized supports AWQ and FP8 models; got "
                f"{quant_config.quant_method!r}."
            )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)# 用字符串形式的名字，从一个对象里取属性。
        print("target cls name is ", target_cls_name)

        # Transformers does not know LLMQT's custom FP8 method names. Remove the
        # field only while constructing the unquantized skeleton, then restore it.
        serialized_quant_config = getattr(config, "quantization_config", None)
        is_fp8 = quant_config.quant_method in {
            "fp8_dynamic_quant",
            "fp8_static_quant",
            "fp8_native_dynamic_quant",
            "fp8_native_static_quant",
        }
        if is_fp8 and hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        try:
            construction_kwargs = dict(model_init_kwargs)
            if torch_dtype != "auto":
                construction_kwargs["torch_dtype"] = torch_dtype
            model = target_cls.from_config(
                config,
                trust_remote_code=trust_remote_code,
                **construction_kwargs,
            )
        finally:
            if is_fp8:
                config.quantization_config = serialized_quant_config
        if torch_dtype != "auto":
            model = model.to(dtype=torch_dtype)

        q_linear_module = get_concrete_linear_module(quant_config.quant_method)
        for layer in self.get_model_layers(model):
            named_linears = get_named_linears(layer)
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )
            for name, linear_layer in named_linears.items():
                if is_fp8:
                    q_linear = q_linear_module(
                        linear_layer.in_features,
                        linear_layer.out_features,
                        linear_layer.bias is not None,
                        dev=linear_layer.weight.device,
                        dtype=linear_layer.weight.dtype,
                        per_tensor=quant_config.per_tensor,
                    )
                else:
                    q_linear = q_linear_module.from_linear(
                        linear=linear_layer,
                        w_bit=quant_config.w_bit,
                        group_size=quant_config.q_group_size,
                        init_only=True,
                    )
                set_op_by_name(layer, name, q_linear)

        single_safe_file = os.path.join(model_weights_path, "model.safetensors")
        single_bin_file = os.path.join(model_weights_path, "pytorch_model.bin")
        if os.path.exists(single_safe_file):
            print(f"[info] loading checkpoint: {single_safe_file}")
            state_dict = safe_load_file(single_safe_file, device="cpu")
            print("[info] loading state_dict into quantized model")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        elif os.path.exists(single_bin_file):
            print(f"[info] loading checkpoint: {single_bin_file}")
            state_dict = torch.load(single_bin_file, map_location="cpu")
            print("[info] loading state_dict into quantized model")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[info] loading sharded checkpoint from: {model_weights_path}")
            missing_keys, unexpected_keys = load_sharded_checkpoint(
                model,
                model_weights_path,
                strict=False,
                prefer_safe=safetensors,
            )
        allowed_missing_keys = {"lm_head.weight"}
        if set(missing_keys) - allowed_missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Unexpected checkpoint mismatch. Missing: {missing_keys}; unexpected: {unexpected_keys}"
            )
        if missing_keys and hasattr(model, "tie_weights"):
            model.tie_weights()

        if device_map is not None:
            print(f"[info] moving quantized model to {device_map}")
            model = model.to(device_map)
        if use_cache is not None:
            model.config.use_cache = use_cache

        model.eval()

        print("[info] quantized model loaded")

        return self(
            model,
            model_type,
            is_quantized=True,
            config=config,
            quant_config=quant_config,
        )
        
    @torch.no_grad()
    def quantize(
        self,
        tokenizer = None,
        quant_config = {},
        calib_data = "pileval",
        duo_scaling = True, # Whether to scale using both w/x or just x, quantizer.py#496
        fake_quant = False,
        apply_clip = True,
        n_parallel_calib_samples= None, # The number of parallel samples to run through the model. A high number of parallel samples can result in OOM during quantization if max_calib_samples is high enough. You can set this to a low number for more memory efficient quantization.
        max_calib_samples = 128, # The maximum number of samples to run through the model
        max_calib_seq_len= 512, # The maximum sequence length of the calibration dataset. will discard samples greater than max_calib_seq_len.
        max_chunk_memory = 1024 * 1024 * 1024, # The loss computation and per-channel mean of AWQ is optimized into chunked computations. Default is 1GB (1024 * 1024 * 1024).
        **kwargs,
    ):
        self.quant_config: QuantConfig = QuantConfig.from_dict(quant_config)
        
        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert
        # dispatch to dedicated quantizer 
        quantizer_cls = get_concrete_quantizer_cls(self.quant_config.quant_method)
        self.quantizer = quantizer_cls(
            self, # models/下面的Qwen2ModelForCausal, only use for awq
            self.model,
            self.model_type,
            tokenizer,
            self.quant_config,
            self.quant_config.quant_method,
            self.quant_config.w_bit,
            self.quant_config.q_group_size, # only for awq
            self.quant_config.zero_point, # only for awq, sq is for homework
            calib_data, # use for sq and awq, dataset name
            duo_scaling, # only for awq
            modules_to_not_convert=self.quant_config.modules_to_not_convert,
            fake_quant=fake_quant, # use for awq and sq, but can change name to fake_quant
            apply_clip=apply_clip, # only for awq
            n_parallel_calib_samples=n_parallel_calib_samples, # only for awq
            max_calib_samples=max_calib_samples,
            max_calib_seq_len=max_calib_seq_len,
            max_chunk_memory=max_chunk_memory, # only for awq
            **kwargs,
        )

        self.quantizer.quantize()

        self.is_quantized = True
        
    def save_quantized(
        self,
        save_dir,
        safetensors = True, # Whether to save the model as safetensors or torch files
        shard_size= "5GB", # The shard size for sharding large models into multiple chunks
    ):
        save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # Save model and config files with empty state dict
        self.model.config.quantization_config = self.quant_config.to_transformers_dict()
        self.model.generation_config.do_sample = True
        self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

        # Remove empty state dict
        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        save_torch_state_dict( # 主要是这个
            state_dict=self.model.state_dict(),
            save_directory=save_dir,
            max_shard_size=shard_size,
            safe_serialization=safetensors,
            force_contiguous=True,
            shared_tensors_to_discard=self.model._tied_weights_keys,
        )

    def _load_config(
        self,
        model_path,
        model_filename,
        safetensors=True,
        trust_remote_code=True,
        max_seq_len=4096,
        download_kwargs=None,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt", "*.onnx*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            # 下载模型到/root/.cache/huggingface/hub
            model_path = snapshot_download(
                model_path, ignore_patterns=ignore_patterns,
            )

        if model_filename != "":
            model_weights_path = model_path + f"/{model_filename}"
        else:
            model_weights_path = model_path

        # [STEP 2] Load config and set sequence length
        quant_config = QuantConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_seq_len is None and hasattr(self, "max_seq_len_key"):
            # 其实就下面这一行最有用。。
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            if hasattr(config, "text_config"):
                config.text_config.max_seq_len = getattr(
                    config, self.max_seq_len_key, 2048
                )
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = max_seq_len

        return model_weights_path, config, quant_config
