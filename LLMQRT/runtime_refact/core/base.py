import os
import gc
import warnings
import torch
import transformers
import torch.nn as nn

from tqdm import tqdm
from typing import List, Union, Dict
from typing_extensions import Doc, Annotated
from huggingface_hub import snapshot_download, save_torch_state_dict

from runtime_refact.nn_models.modules.linear import (
    get_concrete_linear_module,
)
from runtime_refact.utils.common_utils import (
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    BaseImageProcessor,
    ProcessorMixin,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from runtime_refact.core.config import QuantConfig # both need in quantizer and runtime

TRANSFORMERS_AUTO_MAPPING_DICT = {
    "llama": "AutoModelForCausalLM",
    "qwen2": "AutoModelForCausalLM",
    "opt": "AutoModelForCausalLM",
    "qwen2_vl": "AutoModelForVision2Seq",
    "qwen3": "AutoModelForCausalLM",
    "qwen3_moe": "AutoModelForCausalLM",
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BaseModelForCausalLM(nn.Module):
    def __init__(
        self,
        model, # The pretrained or quantized model
        model_type, # The model type, found in config.json
        is_quantized, # Indicates if the input model is quantized
        config, # The config of the model
        quant_config, # The quantization config of the model
        processor, # An optional processor, e.g. for vision models
    ):
        """The base model for all models."""
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: QuantConfig = quant_config
        self.processor: ProcessorMixin = processor

    def to(self, device: Annotated[str, Doc("The device to move your model to.")]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @staticmethod
    def fuse_layers(model):
        pass

    @classmethod
    def from_quantized(
        self,
        model_path, # A Huggingface path or local path to a model
        model_type, #: Annotated[str, Doc("The model type, loaded from config.json.")],
        max_seq_len=None,
        torch_dtype=torch.float16, # 模型量化前的参数类型
        trust_remote_code=True, # Useful for Huggingface repositories that have not been integrated into transformers yet."
        safetensors=True, # Whether to download/load safetensors instead of torch weights
        fuse_layers=True, # 是否做图优化
        device_map="auto", # 加载模型的时候，指定weight加载到哪个device
        max_memory=None, # A dictionary device identifier to maximum memory which will be passed onto the model loading method from transformers. For example：{0: "4GB",1: "10GB"'
        offload_folder=None, # 没啥用，但load_checkpoint_and_dispatch又要
        download_kwargs=None, # 也没啥用，None就行
        **config_kwargs, # Additional kwargs that are passed to the config during initialization."
    ):
        """即输入量化weight路径,返回含qweight和qlinear的quantized model"""
        # [STEP 1-2] Load weights path and configs
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            safetensors,
            trust_remote_code,
            max_seq_len=max_seq_len,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name) # autoModelForCausalLM

        # [STEP 3] 构造模型架子，不含真实weight，为标准的transformer qwen2/qwen3/llama2/llama3等等
        # 基于你提供的配置（config）和数据类型（torch_dtype）进行初始化,根据config里写的，返回transformers.Qwen2/3/LLaMa2/3ForCausalLM的实例
        # 这个函数挺牛逼，有了后，不需要像vllm那样需要手写完整的llama.py/llama3.py/qwen2.py/qwen3.py了,https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py
        with init_empty_weights():
            model = target_cls.from_config(
                config=config,
                torch_dtype=torch_dtype, 
                trust_remote_code=trust_remote_code,
            )
        # 用本项目的nn_models/modules/linear/linear_xx.py来替代输入模型包含的LLMQT量化工具项目中的naive linear_awq/fp8/sq
        self._load_quantized_modules(
            self,
            model,
            quant_config,
            dtype=torch_dtype
        )
        # 这一步只是“换模块并准备空参数”，不是加载真实权重。
        model.tie_weights() # input和output embedding共享一个table
        # 加载权重并应用量化 设备分发与内存优化, 但我们无法控制权重到具体设备，所以经过这个后，weight全都到cuda7去了
        # ！！！！所以为了避免这种情况，我们设置CUDA VISIBLE DEVICES为0，强制都到0去做单卡推理
        native_fp8 = quant_config.quant_method in {
            "fp8_native_dynamic_quant",
            "fp8_native_static_quant",
        }
        # Accelerate's global dtype conversion would turn an E4M3 checkpoint
        # Parameter back into BF16/FP16. Native FP8 checkpoints already contain
        # the desired dtypes (E4M3 weight, FP32 scale, BF16/FP16 bias), so retain
        # them exactly. Legacy checkpoints keep the historical conversion path.
        checkpoint_dtype = None if native_fp8 else torch_dtype
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_weights_path,  # safetensors 文件或权重目录
            device_map="auto", # 、自动分配 GPU、CPU，必要时磁盘
            max_memory=max_memory, # GPU 和 CPU 最多允许使用多少内存。
            no_split_module_classes=[self.layer_type],
            offload_folder=offload_folder,
            dtype=checkpoint_dtype,
        )
        # 此时经过load_checkpoint_and_dispatch后的model变成了加载了fp8/sq/awq quantizer信息(scale, zeropoint等等)的model
        # [STEP 4]图优化, 貌似只有打开了fuse_layers才会用本项目包含的norm,attn等，不然还是用hf那一套，目前llama还没打开，后续可以打开，使用本项目的norm attn等
        if fuse_layers:
            # if llm_quant_runtime is None:
            #     warnings.warn("Skipping fusing modules because AWQ extension is not installed." + msg)
            # else:
            self.fuse_layers(model) # 调用llama.py/llama3.py/qwen2.py/qwen3.py里面的fuse layers
        # 切换为推理模式
        model.eval()
        return self(
            model,
            model_type,
            is_quantized=True,
            config=config,
            quant_config=quant_config,
            processor=None,
        )

    def _load_config(
        self,
        model_path,
        safetensors=True,
        trust_remote_code=True,
        max_seq_len=4096,
        download_kwargs=None,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory,永远不会走到这
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt", "*.onnx*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            if download_kwargs is None:
                download_kwargs = {}

            if "ignore_patterns" in download_kwargs:
                download_kwargs_ignore_patterns = download_kwargs.pop("ignore_patterns")

                if isinstance(download_kwargs_ignore_patterns, str):
                    ignore_patterns.append(download_kwargs_ignore_patterns)
                elif isinstance(download_kwargs_ignore_patterns, list):
                    ignore_patterns.extend(download_kwargs_ignore_patterns)

            model_path = snapshot_download(
                model_path, ignore_patterns=ignore_patterns, **download_kwargs
            )

        model_weights_path = model_path

        # [STEP 2] 加载model_path中的config json，
        # 读取出quant config，包括quant method，scale，zp等（这些是quantizer的时候写入的
        quant_config = QuantConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_seq_len is None and hasattr(self, "max_seq_len_key"):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            # generate support of Multi-modal models
            # if hasattr(config, "text_config"):
            #     config.text_config.max_seq_len = getattr(
            #         config, self.max_seq_len_key, 2048
            #     )
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = max_seq_len

        return model_weights_path, config, quant_config
    
    # inplace load
    def _load_quantized_modules(
        self, model, quant_config, dtype=torch.float16
    ):
        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            # layer为原始的fp16版本未量化版本的linear
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to include
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )

            # Replace activation functions
            self._scale_activations(self, layer)

            # 根据量化方法(awq/sq/fp8)dispatch到相应的linear(awq/sq/fp8)
            q_linear_module = get_concrete_linear_module(quant_config.quant_method) # AWQLinear_GEMM
            # 用本项目的nn_models/modules/linear/linear_xx.py来替代输入模型包含的LLMQT量化工具项目中的naive linear_awq/fp8/sq
            for name, module in named_linears.items():
                q_linear = q_linear_module.from_linear( # q_linear_module必须是确定好的特定量化方法的linear class
                    module, quant_config.w_bit, quant_config.q_group_size, True, dtype=dtype, per_tensor=quant_config.per_tensor 
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

                torch.cuda.empty_cache()
            gc.collect()

    # called in _loaded_quantized_modules
    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer) # 返回一个空dict，目前autoAWQ中暂未看到scale_dict["is_scalable"]为true的情况

        if scale_dict["is_scalable"]:
            if not isinstance(scale_dict["scale_layer"], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(
                    scale_dict["scale_shape"], dtype=param.dtype, device=param.device
                )

                # scale activation
                scaled_act = ScaledActivation(scale_dict["scale_layer"], scale_like)
                set_op_by_name(layer, scale_dict["scale_name"], scaled_act)
