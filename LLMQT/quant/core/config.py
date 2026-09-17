import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from transformers.utils.hub import PushToHubMixin

# 量化工具量化完后，把量化配置写到此类，而后runtime读取此类拿到量化配置
#1、用于将config.json中读config，反序列画用来初始化本类；
#2、在保存的时候，将量化配置序列化为dict；
@dataclass
#@dataclass 是 Python 标准库 dataclasses 提供的装饰器，用来自动为类生成常用方法（例如 __init__、__repr__、__eq__ 等），根据类型注解和默认值创建构造函数，减少样板代码。
class QuantConfig(PushToHubMixin): # 专门用于将模型、配置或分词器等对象一键上传到 Hugging Face Hub，QuantConfig自动获得上传到 Hub 的能力
    quant_method: str = field(default="awq") # fp8, sq
    zero_point: bool = field(default=False) # only use in awq
    q_group_size: int = field(default=0) # only use in awq
    w_bit: int = field(default=8) # only awq is 4
    config_file_name = "config.json"
    modules_to_not_convert: list = field(default_factory=lambda: ["lm_head"])
    fp8_static_quant: bool = field(default=False)
    kv_cache_quant_layers: list = field(default_factory=list) # when enabled, quantize_output=true
    per_tensor: bool = field(default=True)
    
    # base.py#164读取传进来的quant config来初始化本类成员
    @classmethod
    def from_dict(cls, quant_config: Dict = {}):
        if not quant_config:
            quant_config = cls()
        else:
            quant_config = cls(**quant_config)

        return quant_config
    
    # runtime/base.py/from_quantized的时候用到
    # quant的时候api.py#20里面的from_pretrained调base.py
    # from_pretrained的load config里面的from pretrained
    # 用于从config.json读config以反序列化用来初始化本类
    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        assert os.path.isdir(save_dir), "model path should be dir!!"
        resolved_config_file = os.path.join(save_dir, cls.config_file_name)

        quant_config = None
        if os.path.exists(resolved_config_file):
            with open(resolved_config_file, "r", encoding="utf-8") as file:
                loaded_config = json.loads(file.read())

            quant_config = loaded_config.get("quantization_config")
            # 生成quant_config的dict
            if quant_config is not None:
                config_dict = cls.from_transformers_dict(cls, quant_config)
                quant_config = cls(**config_dict)

        if quant_config is None:
            quant_config = cls()

        return quant_config
    
    # 暂时没看到用
    def to_dict(self):
        return {
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "fp8_static_quant": self.fp8_static_quant,
            "kv_cache_quant_layers": self.kv_cache_quant_layers,
            "modules_to_not_convert": self.modules_to_not_convert,
            "per_tensor": self.per_tensor,
        }
        
    # base.py#save_quantized将quant config序列化为dict
    def to_transformers_dict(self):
        return {
            "quant_method": self.quant_method,
            "zero_point": self.zero_point,
            "group_size": self.q_group_size,
            "bits": self.w_bit,
            "fp8_static_quant": self.fp8_static_quant,
            "kv_cache_quant_layers": self.kv_cache_quant_layers,
            "modules_to_not_convert": self.modules_to_not_convert,
            "per_tensor": self.per_tensor,
        }
    # runtime的时候读
    # 本文件的from pretained调到，把json load进来的序列化config dict重新换个key，然后用来初始化本类
    def from_transformers_dict(self, transformers_dict: Dict):
        return {
            "quant_method": transformers_dict.get("quant_method"),
            "zero_point": transformers_dict.get("zero_point"),
            "q_group_size": transformers_dict.get("group_size"),
            "w_bit": transformers_dict.get("bits"),
            "fp8_static_quant": transformers_dict.get("fp8_static_quant"),
            "kv_cache_quant_layers": transformers_dict.get("kv_cache_quant_layers"),
            "modules_to_not_convert": transformers_dict.get("modules_to_not_convert"),
            "per_tensor": transformers_dict.get("per_tensor"),
        }
