import transformers
import torch
import copy
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from quant.nn_models.modules.linear import (
    get_concrete_linear_module,
    FP8StaticLinearQuantizer
)
from functools import partial
from quant.nn_models.modules.linear.linear_fp8 import (
    per_tensor_quantize,
    static_per_tensor_quantize,
    replace_module
)
from quant.utils.common_utils import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    clear_memory, 
    get_best_device
)
from quant.utils.fp8_calib_utils import *
from quant.quantization.base.quantizer import BaseQuantizer
import threading
# fp8不需要sq那么麻烦需要qdq搞来搞去，因为我们可以让fp8 gemm的output直接为fp16

# 多进程，在fp8多卡量化中未用到，最终采用的是多线程
_pool = None
_pool_lock = threading.Lock()

def _get_pool(num_proc):
    import torch.multiprocessing as mp
    global _pool
    with _pool_lock:
        if _pool is None:
            # 关键：延迟到真正需要时才设置 start_method
            mp.set_start_method('spawn', force=True)   # 第一次调用时才执行
            _pool = mp.Pool(processes=num_proc)
    return _pool

class Fp8Quantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM,
        model,
        model_type,
        tokenizer,
        quant_config,
        quant_method,
        w_bit,
        group_size,
        zero_point,
        calib_data,
        duo_scaling,
        modules_to_not_convert=None,
        fake_quant=False,
        apply_clip=False,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        super(BaseQuantizer, self).__init__()
        self.modelforCausalLM = modelforCausalLM
        self.model = model
        self.quant_method = quant_method
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        # 后续修改：保存调用方传入的校准数据，供静态 FP8 激活校准使用。
        self.calib_data = calib_data
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.device = get_best_device() 
        self.parallel = True # 总是开启多卡动态量化
        native_storage = quant_method in {
            "fp8_native_dynamic_quant",
            "fp8_native_static_quant",
        }
        dynamic_method = (
            "fp8_native_dynamic_quant" if native_storage else "fp8_dynamic_quant"
        )
        self.dynamic_quant_linear = get_concrete_linear_module(dynamic_method)
    
    def parallel_quantize_layers(self):
        """使用线程池进行多卡并行量化"""
        layers = self.modelforCausalLM.get_model_layers(self.model)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_layers = len(layers)
        
        # 准备参数：每个layer分配到不同的设备
        tasks = []
        for i, layer in enumerate(layers):
            device_idx = i % num_devices  # 轮询分配设备
            tasks.append((layer, device_idx))

        # 使用线程池
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(tasks)  # 预分配结果列表
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(
                    self.quantize_layer_on_device_thread_safe, 
                    layer, device_idx, self.quant_config, self.dynamic_quant_linear
                ): i
                for i, (layer, device_idx) in enumerate(tasks)
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(tasks), desc="FP8 Quantizing weights in parallel") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                        completed_count += 1
                    except Exception as e:
                        print(f"Error quantizing layer {index}: {e}")
                        results[index] = None
                    finally:
                        pbar.update(1)
                        pbar.set_postfix(completed=f"{completed_count}/{len(tasks)}")
        
        return results

    def quantize_layer_on_device_thread_safe(self, layer, device_idx, quant_config, dynamic_quant_linear):
        """线程安全的量化函数"""
        try:
            # 关键：在每个线程中明确设置CUDA设备
            torch.cuda.set_device(device_idx)
            
            # 将layer数据移动到对应的GPU
            layer = layer.to(f'cuda:{device_idx}')
            
            named_modules = get_named_linears(layer)
            
            for name, linear in named_modules.items():
                if (
                    not isinstance(linear, torch.nn.Linear)
                    or name in quant_config.modules_to_not_convert
                ):
                    print(f"=== Device {device_idx}: skipping {name}")
                    continue
                
                print(f"=== Device {device_idx}: Dynamic Quantizing {name}")
                            
                q_linear = dynamic_quant_linear.from_linear(
                    linear, 
                    per_tensor=quant_config.per_tensor
                )
                
                replace_module(layer, name, q_linear)
                del linear.weight
                del linear.bias
                del linear
            
            # 量化完成后移回CPU
            layer.cpu()
            clear_memory()
            return layer
            
        except Exception as e:
            print(f"Error on device {device_idx}: {e}")
            raise

    def quantize_layer_on_device(self, layer, device_idx, quant_config, dynamic_quant_linear):
        """
        在指定设备上量化单个layer的函数
        """
        # 将layer移动到目标设备
        device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
        layer = layer.to(device)
        
        named_modules = get_named_linears(layer)
        
        for name, linear in named_modules.items():
            if (
                not isinstance(linear, torch.nn.Linear)
                or name in quant_config.modules_to_not_convert
            ):
                print(f"=== Device {device_idx}: skipping {name}")
                continue
            
            print(f"=== Device {device_idx}: Dynamic Quantizing {name}")
                        
            q_linear = dynamic_quant_linear.from_linear(
                linear, 
                per_tensor=quant_config.per_tensor
            )
            
            replace_module(layer, name, q_linear)
            del linear.weight
            del linear.bias
            del linear
        
        # 量化完成后移回CPU
        layer.cpu()
        clear_memory()
        return layer

    # 修改主循环部分
    def parallel_quantize_layers_deprecated(self):
        """
        并行量化所有layers
        """
        layers = self.modelforCausalLM.get_model_layers(self.model)
        num_devices = 8#torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_layers = len(layers)
        #import torch.multiprocessing as mp 
        # 使用多进程并行量化, with block自动管理pool资源，退出with block时自动调用pool.join和pool.close
        #with mp.Pool(processes=num_devices) as pool:
        pool = _get_pool(num_devices)
        try:
        # 准备参数：将layers分组，每个设备处理一组
            layer_groups = []
            device_indices = []
            
            for i in range(0, num_layers, num_devices):
                for j in range(num_devices):
                    if i + j < num_layers:
                        layer_groups.append(layers[i + j])
                        device_indices.append(j)
            
            # 创建偏函数，固定除layer和device_idx外的所有参数
            quantize_func = partial(
                self.quantize_layer_on_device,
                quant_config=self.quant_config, # 非固定参数
                dynamic_quant_linear=self.dynamic_quant_linear, # 非固定参数
            )
            
            # 使用tqdm显示进度
            results = []
            with tqdm(total=len(layer_groups), desc="FP8 Quantizing weights in parallel") as pbar:
                for i, result in enumerate(pool.starmap(quantize_func, zip(layer_groups, device_indices))):
                    results.append(result)
                    pbar.update(1)
        finally:
            pass
        return 


    # fake multi-gpu quantize
    def quantize(self):
        if self.parallel:
            self.parallel_quantize_layers()
        else:
            layers = self.modelforCausalLM.get_model_layers(self.model)
            # 动态量化
            for i in tqdm(range(len(layers)),desc="FP8 Quantizing weights"):
                # 获取当前layer该被分到第几个device
                common_device = next(layers[i].parameters()).device # next的意思是获取该module的第一个参数
                if common_device is None or str(common_device) == "cpu":
                    if torch.cuda.is_available():
                        best_device = "cuda:" + str(i % torch.cuda.device_count()) # 将当前第i个module的weight移动到第i个device
                    else:
                        best_device = get_best_device()

                    layers[i] = layers[i].to(best_device)
                    common_device = next(layers[i].parameters()).device
                # 1.拿到当前layer的所有linear
                named_modules = get_named_linears(layers[i])
                for name, linear in named_modules.items():
                    if (
                        not isinstance(linear, torch.nn.Linear)
                        or name in self.quant_config.modules_to_not_convert
                    ):
                        print("=== skipping ", name)
                        continue
                    print("=== Dynamic Quantizing ", name)
                    # quant_weight, weight_scales = per_tensor_quantize(linear.weight) # 这一步多余
                    # bias = copy.deepcopy(linear.bias) if linear.bias is not None else None
                    # q_linear = self.dynamic_quant_linear.from_linear(linear, weight=quant_weight, weight_scales=weight_scales, bias=bias, per_tensor=self.quant_config.per_tensor)
                    # 2.初始化dynamic quant linear，包括量化weight
                    q_linear = self.dynamic_quant_linear.from_linear(linear, weight=self.quant_weight, weight_scales=self.weight_scales, bias=self.bias, per_tensor=self.quant_config.per_tensor)
                    # 3.用q_linear替换掉layers[i]里面的名为name的torch.nn.Linear
                    replace_module(layers[i], name, q_linear)
                    del linear.weight
                    del linear.bias
                    del linear  
                layers[i].cpu()  
                clear_memory()     
                   
        # 后续修改：仅在静态 FP8 模式准备校准数据，避免动态量化无意义地下载和处理数据集。
        static_methods = {"fp8_static_quant", "fp8_native_static_quant"}
        if self.quant_config.per_tensor and (
            self.quant_method in static_methods or self.quant_config.fp8_static_quant
        ):
            # 后续修改：将调用方传入的校准数据继续传给校准 token 构造函数。
            calib_tokens = prepare_calib_tokens(
                self.tokenizer,
                self.device,
                self.max_calib_samples,
                self.max_calib_seq_len,
                self.calib_data,
            )
            self._apply_quant_act(self.quant_config, calib_tokens)
        else:
            print("[info] skip static quant, since per_tensor=False or quant method is not static quant")
        clear_memory()            


    def _apply_quant_act(self, quant_config, calib_tokens):
        # 1. calibration的准备
        # Replace weight quantizer with a dynamic activation quantizer observer
        for name, dynamic_quant_linear in self.model.named_modules():
            if (
                not isinstance(dynamic_quant_linear, self.dynamic_quant_linear)
                or name in quant_config.modules_to_not_convert
            ):
                continue
            # 找到FP8DynamicLinear，用FP8StaticLinearQuantizer替换FP8DynamicLinear
            # FP8StaticLinearQuantizer负责calibration            
            quantizer = FP8StaticLinearQuantizer(
                in_features=dynamic_quant_linear.in_features,
                out_features=dynamic_quant_linear.out_features,
                qdtype=dynamic_quant_linear.qdtype,
                weight=dynamic_quant_linear.weight,
                weight_scale=dynamic_quant_linear.weight_scale,
                bias=dynamic_quant_linear.bias,
                quantize_output=(
                    hasattr(quant_config, "kv_cache_quant_layers")
                    and name in quant_config.kv_cache_quant_layers
                ),
            )
            replace_module(self.model, name, quantizer)
            del dynamic_quant_linear
        clear_memory()
        # 2.启动calibration
        self.model.to(self.device)
        with torch.inference_mode():
            with tqdm(total=calib_tokens.shape[0], desc="Calibrating activation scales") as pbar:
                for row_idx in range(calib_tokens.shape[0]):
                    self.model(calib_tokens[row_idx].reshape(1, -1))
                    clear_memory()
                    pbar.update(1)
        # 3.用真正的FP8StaticLinear替换掉FP8StaticLinearQuantizer
        static_method = (
            "fp8_native_static_quant"
            if self.quant_method == "fp8_native_static_quant"
            else "fp8_static_quant"
        )
        static_quant_linear = get_concrete_linear_module(static_method)
        for name, quantizer in self.model.named_modules():
            if (
                not isinstance(quantizer, FP8StaticLinearQuantizer)
                or name in quant_config.modules_to_not_convert
            ):
                print("=== skipping ", name)
                continue
            print("=== static Quantizing ", name)
            static_proj = static_quant_linear.from_linear(
                in_features=quantizer.in_features,
                out_features=quantizer.out_features,
                fp8_weight=quantizer.qweight,
                weight_scales=quantizer.weight_scale,
                bias=quantizer.bias,
                input_scale=quantizer.input_scale,
                output_scale=quantizer.output_scale,
                quantize_output=(
                    hasattr(quant_config, "kv_cache_quant_layers")
                    and name in quant_config.kv_cache_quant_layers
                ),
            )
            replace_module(self.model, name, static_proj)
            del quantizer
        clear_memory()

        # 4. 量化kv cache，就目前而言先存scale
        # store kv cache quant scale in the parent attention module as `k_scale` and `v_scale`
        if quant_config.kv_cache_quant_layers:
            # Assumes that list is ordered such that [layer0.k_proj, layer0.v_proj, layer1.k_proj, layer1.v_proj, ...]
            # so we make a list of tuples [(layer0.k_proj, layer0.v_proj), (layer1.k_proj, layer1.v_proj), ...]
            kv_proj_pairs = zip(*[iter(quant_config.kv_cache_quant_layers)]*2)
            for k_proj_name, v_proj_name in kv_proj_pairs:
                # layer 0
                parent_module_name = ".".join(k_proj_name.split(".")[:-1])
                assert parent_module_name == ".".join(v_proj_name.split(".")[:-1])
                parent_module = dict(model.named_modules())[parent_module_name]

                k_proj = dict(model.named_modules())[k_proj_name]
                v_proj = dict(model.named_modules())[v_proj_name]
                # ！！！核心：量化kv在于把kv cache scale保存到k proj和v proj的parent module的属性中
                parent_module.k_scale = torch.nn.Parameter(k_proj.output_scale, requires_grad=False)
                parent_module.v_scale = torch.nn.Parameter(v_proj.output_scale, requires_grad=False)

                # Remove output_scale from k_proj and v_proj
                k_proj.output_scale = None
                v_proj.output_scale = None
        clear_memory()
            
