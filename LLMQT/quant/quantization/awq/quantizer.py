import transformers
import torch
import inspect
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict

from .scale import apply_scale, apply_clip

from quant.nn_models.modules.linear import (
    get_concrete_linear_module
)
import time
from quant.utils.awq_calib_utils import get_calib_dataset
from quant.utils.common_utils import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    clear_memory, 
    get_best_device
)
from quant.quantization.base.quantizer import BaseQuantizer

# 整体步骤: init_quant拿到第0个decoderlayer的input act作为self.inps，
# 然后quantize对每个decoderlayer循环处理(calib，search best scale，apply scale，search best clip，apply clip，real quant)
class AwqQuantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM, # 各模型的模型类比如Qwen2ModelForCausalLM，它的from_pretained返回的model，下面model的自定义包装类
        model, # huggingface : AutoModelForCausalLM.from_pretained返回的
        model_type,
        tokenizer,
        quant_config,
        quant_method,
        w_bit,
        group_size,
        zero_point,
        calib_data, # str
        duo_scaling, # true of false
        modules_to_not_convert=None,
        fake_quant=False, # true时为fake quant，false为real quant
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024, # used in seearch best scale and compute loss, 避免在校准 token 数量很大时一次性分配两个巨大的 FP32/FP16 张量导致 OOM。通过 max_chunk_memory（默认通常设几百 MB）动态决定 chunk size，既能跑在 24 GB 显卡上，也能跑在 80 GB 显卡上，实现“一次代码，到处可跑”
        scipy_refine=False,
        scipy_maxiter=10_000_000,
        scipy_scale_bound=1.25,
        scipy_max_scale_change=0.10,
        scipy_clip_refine=False,
        scipy_clip_maxiter=10_000_000,
        scipy_clip_max_threshold_change=0.10,
    ) -> None:
        super(BaseQuantizer, self).__init__()
        self.awq_model = modelforCausalLM 
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.quant_method = quant_method
        self.w_bit = 4
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.duo_scaling = duo_scaling
        self.fake_quant = fake_quant
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        if self.model_type == "qwen3_moe":
            self.max_calib_samples = max_calib_samples + 128 # increase calib nums for qwen3 moe in case line 763 assert error
        else:
            self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.scipy_refine = scipy_refine
        self.scipy_maxiter = scipy_maxiter
        self.scipy_scale_bound = scipy_scale_bound
        self.scipy_max_scale_change = scipy_max_scale_change
        self.scipy_refine_stats = []
        self.scipy_clip_refine = scipy_clip_refine
        self.scipy_clip_maxiter = scipy_clip_maxiter
        self.scipy_clip_max_threshold_change = scipy_clip_max_threshold_change
        self.scipy_clip_refine_stats = []
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        # 返回值这里不能命名为self.modules，因为BaseQuantizer是一个torch.nn.Module，它也有self.modules成员
        # self.inps表示捕获到的第一个layer的输入，用于layer1-layern的calib
        # self.target_modules表示模型的所有layers
        self.target_modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )
    # fake quant
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape #[5120,5120]
        if self.group_size > 0: # for deepseek and group size > 0
            assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
            w = w.reshape(-1, self.group_size) #[5120x40,128]
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        # 1.非对称量化的scale和zp的计算公式
        # scale = (absmax - absmin) / 255, zp = clip(-round(absmin/scale), 0, 255)
        # qx = clip(round(x / scale - zp))
        # 2.对称量化的scale和zp的计算公式
        # scale = absmax / 127, zp = 0
        # qx = clip(round(x / scale), -128, 127)     
        # 如果是int4，max_int = 2 ** (self.w_bit - 1) - 1 = 2 **（4-1） - 1 = 7, 则scale = absmax / 7, qx = clip(round(x / scale), -8, 7)  
        
        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True) # [5120x40,1]即列上每128个元素的最大值
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1  
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales
            #量化后又反量化，包含量化误差 

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0
        scales = scales.view(org_w_shape[0], -1) # [5120x40,1]=>[5120,40]
        w = w.reshape(org_w_shape) #[5120x40,128]=>[5120,5120]

        return w, scales, zeros#[5120,40]

    def quantize(self):
        # 遍历每个decoderLayer 。tqdm进度条
        for i in tqdm(range(len(self.target_modules)), desc="AWQ"): # init_quant返回了第0个decoder layers的input act
            start = time.perf_counter()
            #@@@@ Move module and inputs to correct device（multi gpu）分发module到不同的GPU，实现多GPU量化
            common_device = next(self.target_modules[i].parameters()).device # next的意思是获取该module的第一个参数
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count()) # 将当前第i个module的weight移动到第i个device
                else:
                    best_device = get_best_device()

                self.target_modules[i] = self.target_modules[i].to(best_device)
                common_device = next(self.target_modules[i].parameters()).device
            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)
            # 把第0个decoder layer的输入传到GPU、best device
            self.inps = self.inps.to(common_device)# init quant得到得第0个decoder layers的input act
            # 把emb table传到GPU、best device
            # 目的：减少内存占用，避免重复计算。
            # 副作用：量化时需要显式确保 rotary_embed 和设备同步。
            # 如果transformers 4.45.0后, rotary_embed 是全局的，而某些层被移动到其他设备，会导致 设备不匹配错误，所以每个layer都需要显式移动rotary embed到指定设备
            self.awq_model.move_embed(self.model, common_device)
            if (transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get('attention_mask') is None):
                self.module_kwargs['attention_mask'] = None

            for k, v in self.module_kwargs.items():
                # position embeddings found in tuple
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module)) 
                        else item for item in v
                    )
            # 返回第i个decoder layer上所有linear的name和torch.nn.Linear的映射字典
            # {'self_attn.q_proj': Linear(in_features=5120, out_features=5120, bias=True), 
            # 'self_attn.k_proj': Linear(in_features=5120, out_features=1024, bias=True), 
            # 'self_attn.v_proj': Linear(in_features=5120, out_features=1024, bias=True), 
            # 'self_attn.o_proj': Linear(in_features=5120, out_features=5120, bias=False), 
            # 'mlp.gate_proj': Linear(in_features=5120, out_features=13824, bias=False), 
            # 'mlp.up_proj': Linear(in_features=5120, out_features=13824, bias=False), 
            # 'mlp.down_proj': Linear(in_features=13824, out_features=5120, bias=False)}
            named_linears = get_named_linears(self.target_modules[i])
            #@@@@@@ Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            #@@@@@@ calib，返回每个decoderlayer中每个linear的input features,送去apply scale后再送去决定clip
            # dict_keys(['self_attn.q_proj', 'self_attn.k_proj','self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'])
            input_feat = self._get_input_feat(self.target_modules[i], named_linears)
            end0 = time.perf_counter()
            clear_memory()
            print("[info] the get_input_feat time per layer is ", end0 - start)
            # 返回第i个decoder layer的config：qkvo gate up down
            # (Pdb) module_config[0].keys()
            # dict_keys(['prev_op', 'layers', 'inp', 'module2inspect', 'kwargs'])
            # (Pdb) len(module_config)
            # 3 attn qkvo为1个，gate和up为1个，down为1个
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.target_modules[i], input_feat, self.module_kwargs
            )
            # (Pdb) scales_list[0]
            # ('input_layernorm' prev op name, ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj') layer name,
            #   best scales: tensor([1.3828, 1.2344, 1.2500,  ..., 1.2812, 1.1250, 1.1406], dtype=torch.bfloat16))
            #@@@@@@搜寻每个linear的best awq scale
            scales_list = [ 
                self._search_best_scale(self.target_modules[i], **layer) # 数据都在module2inspect.parameter的device上面，即attn mlp这些module
                for layer in module_config
            ]
            #@@@@@@把上面搜索出来的best scale乘到当前layer的weight，activation对应的scale融合到prev op
            # W=(W*S)，X=(S^-1 * X) 
            apply_scale(self.target_modules[i], scales_list, input_feat_dict=input_feat) 
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.target_modules[i]) + "."
            )
            end1 = time.perf_counter()
            print("[info] the apply_scale time per layer is ", end1 - end0)
            #@@@@@@Compute and apply clipping list
            if self.apply_clip:
                #@@@@@@搜寻每个linear的best awq clip
                clip_list = self._search_best_clip(
                    self.target_modules[i], named_linears, input_feat, common_device
                ) 
                #@@@@@@把上面搜索出来的best clip乘到当前layer的weight
                apply_clip(self.target_modules[i], clip_list) # 开头，weight to到GPU，末尾，weight再一次一个个的卸载到cpu, 节约量化占用的显存
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.target_modules[i]) + "."
                )
            end2 = time.perf_counter()
            print("[info] the apply_clip time per layer is ", end2 - end1)
            # 真正的量化的地方，在这之前上面的代码都是fp16的weight
            # fp16 weight => int4 weight
            # self.target_modules[i]目前是一个fp16 decoder layer，apply_quant后原地变为了int4 weight的decoder layer
            if not self.fake_quant:
                self._apply_quant(self.target_modules[i], named_linears, common_device)
            
            clear_memory()
            end3 = time.perf_counter()

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear], common_device):
        # named_linears 保存当前 decoder layer 内所有待量化的 Linear，
        # key 是在 module 内部的层路径（如 self_attn.q_proj），value 是原始 nn.Linear。
        for name, linear_layer in named_linears.items():
            print("[info] hit ", name)
            # 将当前 Linear 移到本层所在的计算设备，并转成 fp16。
            # pseudo_quantize_tensor 和后续 pack 权重都在同一设备上完成，避免 CPU/GPU 来回拷贝。
            linear_layer = linear_layer.to(common_device).half() # 这里to common device，时间会减小1/3-1/2
            
            # 对 weight 做“伪量化”：先按 group_size 计算 scale/zero，再量化并反量化回 fp16。
            # 返回值中：
            #   weight.data 是带量化误差的 fp16 权重，用于后续打包成真实 int 权重；
            #   scales 形状为 [out_features, in_features / group_size]；
            #   zeros 仅 zero_point=True 时存在，形状同 scales。
            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )
            # 前面的伪量化只是为了拿到量化的scale和zero，其实直接用公式算也一样，无非就是目前存在封装问题的缘故
            scales = scales.t().contiguous()
            if zeros is not None:
                zeros = zeros.t().contiguous()# (out channel, in channel)=>(in channel, out channel)

            # 根据当前量化方法（如 AWQ）选择具体的 QuantLinear 实现类。
            q_linear_module = get_concrete_linear_module(self.quant_method) 

            # 用原始 Linear 的结构、bias、已计算好的 scale/zero 创建量化 Linear。
            # init_only=False 表示立即把 fp16 权重按照 w_bit/group_size 打包为推理用的低比特权重。
            # 在from linear中量化
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,# [in channels/128,out channels ]
                zeros=zeros, # [in channels/128,out channels ]
            )

            # 原始 Linear 已经被量化模块替代，先搬回 CPU 释放当前 GPU 上的 fp16 权重占用。
            linear_layer.cpu()

            # 将新建的 QuantLinear 放回当前 decoder layer 所在设备，
            # 保证替换后 module 内部参数设备一致，后续 forward 不会出现 device mismatch。
            q_linear.to(next(module.parameters()).device)

            # 在 module 中按 name 路径原地替换原来的 nn.Linear，例如 self_attn.q_proj -> QuantLinear。
            set_op_by_name(module, name, q_linear)

            # 每替换完一个 Linear 就清理一次缓存，降低逐层量化时的峰值显存。
            clear_memory()

    # 对每个layer作calib
    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad() # 数据都在module2inspect的device上面
    def _search_best_scale(
        self,
        module,# transformers layer 
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: 计算[out channels, in channels]下，每列的weight均值
        # 行维度拼接,concat([out1,in],[out2,in])=>[out1+out2,in]
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        weight = weight.view(-1, self.group_size) # [(out1+out2)*in/128,128]
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6) # weights的per group归一化
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape) # [out1+out2,in]
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0) # [in]，对[out,in]这个shape的每列归一化后的均值，即对每个in channel的outputchannel个weight求均值，论文上的图也是这样的
        clear_memory(weight)

        # [STEP 2]: 计算[bs*tokens, in channels]下，每列的activation均值
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0) # 行=bs*numtokens
        num_channels = inp_flat.size(1) # 列=hiddensize
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory 可容纳行数 = 总内存字节数 ÷ 每行字节数
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation, 一个channel的所有token求和和均值，保存在x_sum,确实论文上的图也是这样的
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        # 用分块求每一列的和sum(dim=0)
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)
        # [in], [bs,in channels]的x shape下的每列mean
        x_mean = (x_sum / num_elements).to(inp.dtype) 
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module, 对module2inspect（对于qkv是attn，对于gateupdown是mlp）做fwd，求它的out
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            # 下面是enable deepseek v3加上的
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4]: Compute loss，基于x和w的均值，linear对应的module2inspect的fwd结果
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    # 数据都在x的device上面
    def _compute_best_scale(
        self,
        x: torch.Tensor, # calib input
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,  # attn module
        linears2scale: List[nn.Linear], # qkv linear, gateupdown linear
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        # 求出了channel level的x mean和w mena后
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)
        # x mean和w mean是每个channel的mean
        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid
            # AWQ 原文确实只拿 activation 的 magnitude 作为「重要性」的 proxy
            # 但 open-source 的 auto-awq 后来为了稳定收敛给了一个可选的 duo_scaling 分支，把 weight 的均值也拉进来做一个几何加权
            # 好处有：
            # 1.防止极端通道的activation 很小，按论文方法几乎把 scale 压到 0，导致量化后权重直接“消失”
            # 2.搜索空间平滑：用 (x^ratio) / (w^(1-ratio)) 的形式，把原来只有一条 activation 曲线变成 ratio∈[0,1] 的连续曲线，网格搜索更稳定，不容易出现 loss 突然爆炸
            # 3.在 Llama / DeepSeek 这类大模型上，作者发现 activation 极度稀疏，只靠 activation 会低估某些“权重很大但很少被激活”的通道的重要性，于是给 weight 也留了一个“话语权”。
            if self.duo_scaling: # Whether to scale using both w/x or just x
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            # 但是为什么最终scale是下式？作用是1.平衡scale的范围，x_mean.pow(ratio) 得到的值可能会有较大的动态范围，即最大值最小值之间的差距可能很大，通过除以 (scales.max() * scales.min()).sqrt()，可以将缩放因子的范围调整到一个更合理的区间，避免某些通道的缩放因子过大或过小，从而导致量化后的权重分布不均匀
            # 2.保持数值稳定性：如果scale的范围过大，可能会导致数值不稳定，例如在后续的乘法和除法操作中出现溢出或下溢            
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s) * s^-1: 对应论文提到的fuse to previous op，和smoothquant的处理一样
            for fc in linears2scale:
                fc.weight.mul_(scales_view) # w * s
                fc.weight.data = ( # Q(w*s) * s^-1
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view # fp16--int4--fp16
                )

            #（Q(W * s) * s^-1） * X，对module2inspect(attn mlp）里面的这些linear做fwd
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            # 下面是enable deepseek v3加上的
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)
            # fp16_output = W * X
            # compute mean squared error (L2 norm) between fp16_output and int_w_output
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            # 对module2inspect的各种linear weight作了修改:对w*s fakequant + 乘了个s^-1
            # 所以下面这一行需要复原，以做下一个iter的grid search
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        if self.scipy_refine:
            refinement = self._refine_best_scale_scipy(
                x=x,
                module2inspect=module2inspect,
                linears2scale=linears2scale,
                fp16_output=fp16_output,
                kwargs=kwargs,
                initial_scales=best_scales,
                initial_loss=best_error,
            )
            diagnostics = refinement.diagnostics()
            diagnostics["grid_best_ratio"] = float(best_ratio)
            diagnostics["module"] = module2inspect.__class__.__name__
            diagnostics["layers"] = self._relative_layer_names(
                module2inspect, linears2scale
            )
            self.scipy_refine_stats.append(diagnostics)
            best_scales = refinement.scales
            print(
                "[scipy-awq] "
                f"accepted={refinement.accepted} "
                f"grid_loss={refinement.initial_loss:.6e} "
                f"candidate_loss={refinement.candidate_loss:.6e} "
                f"improvement={refinement.relative_improvement * 100:.4f}% "
                f"rollback={refinement.callback_triggered} "
                f"iterations={refinement.iterations} "
                f"evals={refinement.function_evaluations}"
            )

        return best_scales.detach().cpu()

    @staticmethod
    def _relative_layer_names(module2inspect, linears2scale):
        module_names = {
            id(submodule): name or "<self>"
            for name, submodule in module2inspect.named_modules()
        }
        return [
            module_names.get(id(layer), layer.__class__.__name__)
            for layer in linears2scale
        ]

    def _evaluate_scale_exact(
        self,
        x,
        module2inspect,
        linears2scale,
        fp16_output,
        kwargs,
        scales,
    ):
        """Evaluate the original, non-differentiable AWQ module MSE."""

        device = x.device
        original_weights = [layer.weight.detach().clone() for layer in linears2scale]
        try:
            with torch.no_grad():
                for layer, original_weight in zip(linears2scale, original_weights):
                    scales_view = scales.to(
                        device=original_weight.device,
                        dtype=original_weight.dtype,
                    ).view(1, -1)
                    scaled_weight = original_weight * scales_view
                    quantized_weight = self.pseudo_quantize_tensor(scaled_weight)[0]
                    layer.weight.copy_(quantized_weight / scales_view)

                output = self._module_forward(x, module2inspect, kwargs)
                output = output.clip(
                    torch.finfo(output.dtype).min,
                    torch.finfo(output.dtype).max,
                )
                return self._compute_loss(fp16_output, output, device)
        finally:
            with torch.no_grad():
                for layer, original_weight in zip(linears2scale, original_weights):
                    layer.weight.copy_(original_weight)

    def _refine_best_scale_scipy(
        self,
        x,
        module2inspect,
        linears2scale,
        fp16_output,
        kwargs,
        initial_scales,
        initial_loss,
    ):
        """Jointly refine every input-channel scale using SciPy L-BFGS-B.

        The forward value uses the real groupwise fake quantizer. Since round is
        discontinuous, the backward value uses a straight-through estimator.
        The candidate is accepted only after evaluation by
        ``_evaluate_scale_exact``, which is the same objective used by the AWQ
        grid search.
        """

        try:
            from torch.func import functional_call
            from .scale_scipy import (
                channel_scales_from_log_correction,
                refine_channel_scales,
            )
        except ImportError as error:
            raise ImportError(
                "scipy_refine=True requires SciPy and torch.func.functional_call"
            ) from error

        device = x.device
        initial_scales = initial_scales.detach().to(device)
        target_output = fp16_output.detach().to(device)
        original_weights = {
            id(layer): layer.weight.detach().clone().to(device)
            for layer in linears2scale
        }
        module_names = {
            id(submodule): name for name, submodule in module2inspect.named_modules()
        }

        parameter_names = {}
        for layer in linears2scale:
            relative_name = module_names.get(id(layer))
            if relative_name is None:
                raise RuntimeError(
                    "A layer selected for AWQ scaling is not contained in "
                    "module2inspect"
                )
            parameter_names[id(layer)] = (
                f"{relative_name}.weight" if relative_name else "weight"
            )

        def differentiable_objective(log_correction):
            scales = channel_scales_from_log_correction(
                initial_scales.float(), log_correction
            )
            replacements = {}
            for layer in linears2scale:
                original_weight = original_weights[id(layer)]
                scales_view = scales.to(original_weight.dtype).view(1, -1)
                scaled_weight = original_weight * scales_view
                rounded_weight = self.pseudo_quantize_tensor(scaled_weight)[0]
                # Real quantized values in the forward pass, identity derivative
                # in the backward pass. Division by the channel scale still
                # exposes the scale-dependent quantization error to L-BFGS-B.
                quantized_ste = scaled_weight + (
                    rounded_weight - scaled_weight
                ).detach()
                replacements[parameter_names[id(layer)]] = (
                    quantized_ste / scales_view
                )

            output = functional_call(
                module2inspect,
                replacements,
                args=(x,),
                kwargs=kwargs,
                strict=False,
            )
            if isinstance(output, tuple):
                output = output[0]
            output = output.clip(
                torch.finfo(output.dtype).min,
                torch.finfo(output.dtype).max,
            )
            return (output.float() - target_output.float()).pow(2).mean()

        def exact_objective(scales):
            return self._evaluate_scale_exact(
                x,
                module2inspect,
                linears2scale,
                fp16_output,
                kwargs,
                scales,
            )

        return refine_channel_scales(
            initial_scales=initial_scales,
            initial_loss=initial_loss,
            differentiable_objective=differentiable_objective,
            exact_objective=exact_objective,
            maxiter=self.scipy_maxiter,
            relative_bound=self.scipy_scale_bound,
            max_scale_change=self.scipy_max_scale_change,
        )

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat, common_device):
        """
        为当前 decoder layer 中的每个可量化 Linear 搜索最佳裁剪阈值。

        返回值示例：
            [
                ("self_attn.v_proj", max_val_v),
                ("self_attn.o_proj", max_val_o),
                ("mlp.gate_proj", max_val_gate),
                ...
            ]

        其中 max_val 的 shape 为 [out_channels, n_group, 1]。也就是说，
        每个 Linear、每个输出通道、每个输入 group 都有自己的裁剪阈值，
        并不是整个 Linear 共用一个 max_val。
        """
        clip_list = []
        # Q/K 对 attention score 较敏感，裁剪后容易放大 QK^T 的误差，因此跳过。
        # Wqkv 用于匹配某些模型中融合后的 QKV Linear 名称。
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]
        
        for name in named_linears:
            if any([_ in name for _ in avoid_clipping]):
                continue

            # weight 和对应的校准输入必须位于同一计算设备上。
            named_linears[name].to(common_device)
            max_val = self._compute_best_clip(
                named_linears[name].weight,
                input_feat[name],
                layer_name=name,
            )

            # 后续 apply_clip() 会根据 name 找回 Linear，并将 weight 截断到
            # [-max_val, max_val]。
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
        layer_name=None,
    ):
        """
        通过网格搜索，为一个 Linear 找到量化误差最小的裁剪阈值。

        搜索单位：
            一个输出通道 × 一个输入 group。

        小例子：
            w.shape = [4, 8]，group_size = 4
            => out_channels = 4，n_group = 8 / 4 = 2
            => 最终会独立搜索 4 × 2 = 8 个裁剪阈值
            => 返回值 shape 为 [4, 2, 1]

        对某个输出通道、某个 group，比较的是：
            原始局部输出：sum(X_group * W_group)
            量化局部输出：sum(X_group * Q(clip(W_group)))

        然后在采样 token 维度上计算 MSE，选择误差最小的阈值。
        """
        assert w.dim() == 2
        org_w_shape = w.shape
        # co = out_channels，ci = in_channels。
        #
        # reshape 后：
        # w:
        #   [co, ci] -> [co, 1, n_group, group_size]
        #   中间的 1 用于和 token 维广播。
        #
        # input_feat:
        #   [batch, seq_len, ci] 或 [n_token, ci]
        #   -> [1, n_token, n_group, group_size]
        #   开头的 1 用于和输出通道 co 广播。
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]

        # 将 batch、sequence 等前置维度全部展平为 token 维。
        # 例如 [128, 512, 4096] -> [65536, 4096]->[1, 65536, 32 ,128] 。
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # 均匀下采样 token，减少搜索计算量。
        # 例如共有 65536 个 token，n_sample_token=512：
        # step_size=128，选择 token 0、128、256、...。
        #
        # 注意：使用切片后，token 数通常约为 512；当总 token 数不能整除
        # n_sample_token 时，实际数量可能略大于 512。
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        # [co, ci] -> [co, 1, n_group, group_size]
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        # 分批处理输出通道，避免同时生成所有候选量化权重导致 OOM。
        # 这里只改变计算批次，不改变每个输出通道独立搜索阈值的语义。
        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []
        scipy_batch_stats = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            # 当前 w.shape:
            # [oc_batch_size, 1, n_group, group_size]
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            # 每个“输出通道 × group”的原始绝对最大值。
            # 例如当前 batch 有 256 个输出通道、n_group=32：
            # org_max_val.shape = [256, 1, 32, 1]。
            org_max_val = w.abs().amax(dim=-1, keepdim=True)

            # 默认先使用“不裁剪”的阈值；后面遇到更小误差时逐元素更新。
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)

            # 原始 FP 权重的局部点积，作为 ground truth。
            # 广播过程：
            # input_feat [1, token, n_group, group_size]
            # w          [oc, 1,     n_group, group_size]
            # 相乘求和 -> [oc, token, n_group]
            #
            # 这里保留 n_group 维，没有立刻对所有 group 求和，因为我们需要为
            # 每个 group 独立选择裁剪阈值。
            org_out = (input_feat * w).sum(dim=-1)

            # 搜索 shrink ratio。默认 n_grid=20、max_shrink=0.5 时：
            # i_s = 0, 1, ..., 9
            # 阈值比例 = 1.00, 0.95, 0.90, ..., 0.55
            #
            # i_s=0 不是完全“不处理”：虽然不裁剪权重，但仍会执行伪量化，
            # 因此它表示“不裁剪情况下的量化误差基线”。
            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val

                # 将离群权重截断到当前候选范围。
                cur_w = torch.clamp(w, min_val, max_val)

                # 伪量化：量化后立即反量化，q_w 仍是浮点 Tensor，
                # 但已经包含低比特量化造成的舍入误差。
                #
                # cur_w 虽然是 4 维，但当 self.group_size > 0 时，
                # pseudo_quantize_tensor() 内部会先将它 reshape 为
                # [-1, group_size]，量化后再恢复为 cur_w 的原始 shape，
                # 因此这里不需要额外展平。
                #
                # 只有 self.group_size <= 0 时，内部不会执行该 reshape，
                # 4 维 cur_w 才会在 w.dim() == 2 的断言处失败。
                q_w = self.pseudo_quantize_tensor(cur_w)[0]

                # 使用候选量化权重计算局部点积：
                # cur_out.shape = [oc, token, n_group]。
                cur_out = (input_feat * q_w).sum(dim=-1)

                # 对 token 维求均方误差：
                # err[oc, group] =
                #     mean_token((quant_local_out - fp_local_out)^2)
                #
                # reshape 后与 max_val/min_errs 保持相同 shape：
                # [oc, 1, n_group, 1]。
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out

                # 逐“输出通道 × group”比较，而不是整个 Linear 只比较一次。
                # 因此同一轮候选可能只更新其中一部分 group 的最佳阈值。
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]

            if self.scipy_clip_refine:
                try:
                    from .clip_scipy import refine_clip_ratios
                except ImportError as error:
                    raise ImportError(
                        "scipy_clip_refine=True requires SciPy"
                    ) from error

                # The original 20-point clip search remains authoritative and
                # supplies the initial ratio. SciPy only explores the continuous
                # interval [1 - max_shrink, 1] around that discrete result.
                safe_org_max = org_max_val.float().clamp_min(1e-12)
                initial_ratios = (
                    best_max_val.float() / safe_org_max
                ).clamp(1.0 - max_shrink, 1.0)

                def clip_errors(ratios, use_ste):
                    candidate_max = org_max_val.float() * ratios
                    candidate_max = candidate_max.to(w.dtype)
                    clipped_w = torch.clamp(w, -candidate_max, candidate_max)
                    quantized_w = self.pseudo_quantize_tensor(clipped_w)[0]
                    if use_ste:
                        quantized_w = clipped_w + (
                            quantized_w - clipped_w
                        ).detach()
                    candidate_out = (input_feat * quantized_w).sum(dim=-1)
                    return (candidate_out - org_out).float().pow(2).mean(
                        dim=1
                    ).view(min_errs.shape)

                refinement = refine_clip_ratios(
                    initial_ratios=initial_ratios,
                    initial_errors=min_errs,
                    differentiable_objective=lambda ratios: clip_errors(
                        ratios, use_ste=True
                    ).mean(),
                    exact_errors=lambda ratios: clip_errors(
                        ratios, use_ste=False
                    ),
                    maxiter=self.scipy_clip_maxiter,
                    min_ratio=1.0 - max_shrink,
                    max_ratio=1.0,
                    max_threshold_change=self.scipy_clip_max_threshold_change,
                )
                best_max_val = (
                    org_max_val.float() * refinement.ratios
                ).to(org_max_val.dtype)
                min_errs = clip_errors(
                    refinement.ratios, use_ste=False
                ).detach().to(min_errs.dtype)
                scipy_batch_stats.append(refinement.diagnostics())

            best_max_val_all.append(best_max_val)

        # 拼回全部输出通道：
        # [out_channels, 1, n_group, 1]。
        best_max_val = torch.cat(best_max_val_all, dim=0)

        if scipy_batch_stats:
            total_groups = sum(item["total_groups"] for item in scipy_batch_stats)
            initial_loss = sum(
                item["initial_loss"] * item["total_groups"]
                for item in scipy_batch_stats
            ) / total_groups
            final_loss = sum(
                item["final_loss"] * item["total_groups"]
                for item in scipy_batch_stats
            ) / total_groups
            clip_stat = {
                "layer": layer_name,
                "output_channels": org_w_shape[0],
                "groups_per_output": org_w_shape[1] // group_size,
                "total_groups": total_groups,
                "accepted_groups": sum(
                    item["accepted_groups"] for item in scipy_batch_stats
                ),
                "callback_rollbacks": sum(
                    int(item["callback_triggered"])
                    for item in scipy_batch_stats
                ),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "relative_improvement": (
                    (initial_loss - final_loss) / initial_loss
                    if initial_loss > 0
                    else 0.0
                ),
                "iterations": sum(
                    item["iterations"] for item in scipy_batch_stats
                ),
                "function_evaluations": sum(
                    item["function_evaluations"] for item in scipy_batch_stats
                ),
                "max_observed_threshold_change": max(
                    item["max_threshold_change"] for item in scipy_batch_stats
                ),
                "callback_reasons": [
                    item["callback_reason"]
                    for item in scipy_batch_stats
                    if item["callback_reason"] is not None
                ],
            }
            self.scipy_clip_refine_stats.append(clip_stat)
            print(
                "[scipy-clip] "
                f"layer={layer_name} "
                f"accepted={clip_stat['accepted_groups']}/"
                f"{clip_stat['total_groups']} "
                f"improvement={clip_stat['relative_improvement'] * 100:.4f}% "
                f"rollbacks={clip_stat['callback_rollbacks']}"
            )

        clear_memory(input_feat)
        clear_memory(org_out)

        # 去掉为了广播添加的维度 1，供 apply_clip() 使用。
        return best_max_val.squeeze(1)  # [out_channels, n_group, 1]

    def init_quant(self, n_samples=128, max_seq_len=512):
        # n x decoder layers
        modules = self.awq_model.get_model_layers(self.model) # from pretained返回的model
        samples = get_calib_dataset(
            data=self.calib_data,
            # data="wikitext-2-v1",
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            # split=self.split,
            split="validation",
        )
        samples = torch.cat(samples, dim=0)

        inps = []# list,里面只有一个元素，shape为[59,512,5120]
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)
        # @@@@@@捕获到第0个module（第0个decoder layers）的input act到inps作为全局inps用到quantize
        # 如何捕获到layer0的input
        # 方法：hack一个hook
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.ori_module = module
                
                # 继承原始模块的所有属性, 以防calib的时候报AttributeError: 'Catcher' object has no attribute 'attention_type'
                for name, value in module.__dict__.items():
                    setattr(self, name, value)
                    
            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                #layer0(hidden_states, attention_mask, position_ids)
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference
        original_module = modules[0]
        #@@@@@@ patch layer 0 to catch input and kwargs
        #modules[0] = Catcher(modules[0])
        modules[0] = Catcher(original_module)
        try: # calibration以catch第0个module的输入,存到inps和layer_kwargs
            # 这里的to device才能保证samples和model params都在同一device，避免出现一个在cpu一个在gpu
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        #import pdb;pdb.set_trace()
        modules[0] = original_module
        # modules[0] = modules[0].ori_module  # restore
        # prepare_inputs_for_generation的解释：输入samples，根据当前状态动态调整模型的输入
        # ep.prefill阶段，inputs如下：
        # inputs = {
        #     "input_ids": initial_input,  # 初始输入 [batch_size, seq_len]
        #     "attention_mask": mask,      # 初始掩码
        #     "use_cache": True
        # }
        # model_inputs = model.prepare_inputs_for_generation(**inputs)
        # decode阶段，inputs如下：有了kv cache，且input_ids的shape不一样了
        # inputs = {
        #     "input_ids": new_token,      # 新生成的 token [batch_size, 1]
        #     "past_key_values": past,     # 上一轮的 KV Cache
        #     "attention_mask": updated_mask,  # 扩展掩码
        # }
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs) # 这个应该是hf里面modeling_qwen3.py里的方法
        # layer_kwargs: dict_keys(['cache_position', 'input_ids', 'inputs_embeds', 'position_ids', 'past_key_value', 'output_attentions', 'use_cache', 'position_embeddings'])
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]
        # 为什么要搬回CPU？答：省显存
        modules[0] = modules[0].cpu() # modules为list，里面是每个decoderlayer module
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()
        return modules, layer_kwargs, inps
    
    # 参数:1.decoder layer 2.所有linear的name:nn.Module映射
    def _get_input_feat(self, layer, named_linears):
        # 在每个linear module上面注册这个钩子，等到forward的时候，运行到了该linear，将自动触发这个钩子
        # m:module, x:输入, y:输出, name:linear的名字, feat_dict:存储每个linear输入的字典
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x) # {linear name: input act}

        input_feat = defaultdict(list)
        handles = []
        # 这里是qwen3的改动，加了这行才能跑，猜测gate up的输入为named_linears[mlp]的值,后面打印一下layer看看
        if self.awq_model.model_type == "qwen3_moe" or self.awq_model.model_type == "deepseek_v3":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }
        if self.awq_model.model_type == "llama4":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }            
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        # 返回的self.inps为当前入参layer的input
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu

        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)
        # self.inps是第一个layer的输入，self.inp为当前layer的输入
        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()

        def cat_and_assert(k, v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] != 0, (
                f"{k} has a zero dimension. This can happen if no data was passed through (e.g. an expert in MoE not being activated). "
                "Try increasing max_calib_samples (warning: this can significantly increase quantization time and memory usage.)"
            )
            return x

        input_feat = {k: cat_and_assert(k, v) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        过滤掉目标模块（module）的 forward 方法不支持的参数，确保传入的关键字参数（inputs_kwargs）
        不会因为transformers版本差异或参数不匹配导致模块的前向传播（forward）失败

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """

        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs


# =============================================================================
# 补充说明：self.target_modules 的结构，以及 next(...parameters()).device
# =============================================================================
#
# 1. self.target_modules 从哪里来？
#
# 在 __init__() 中：
#
#     self.target_modules, self.module_kwargs, self.inps = self.init_quant(...)
#
# init_quant() 中继续调用：
#
#     modules = self.awq_model.get_model_layers(self.model)
#
# 每个模型 adapter 会返回 Transformer 的 decoder layers，例如：
#
#     Qwen2/Qwen3/LLaMA:
#         model.model.layers
#
#     OPT:
#         model.model.decoder.layers
#
# 当前这些对象通常是 torch.nn.ModuleList。它不是重新复制出来的一份模型，
# 里面保存的就是 self.model 中原来的 DecoderLayer 对象。因此，对
# self.target_modules[i] 内部 Linear 的替换，也会直接修改 self.model。
#
# 以一个 32 层的 Qwen2-like 模型为例，结构可以粗略理解为：
#
#     self.target_modules = ModuleList(
#         [
#             Qwen2DecoderLayer(                 # self.target_modules[0]
#                 input_layernorm=...,
#                 self_attn=Qwen2Attention(
#                     q_proj=nn.Linear(...),
#                     k_proj=nn.Linear(...),
#                     v_proj=nn.Linear(...),
#                     o_proj=nn.Linear(...),
#                 ),
#                 post_attention_layernorm=...,
#                 mlp=Qwen2MLP(
#                     gate_proj=nn.Linear(...),
#                     up_proj=nn.Linear(...),
#                     down_proj=nn.Linear(...),
#                 ),
#             ),
#             Qwen2DecoderLayer(...),             # self.target_modules[1]
#             ...
#             Qwen2DecoderLayer(...),             # self.target_modules[31]
#         ]
#     )
#
# 因此：
#
#     len(self.target_modules)
#
# 通常等于 config.num_hidden_layers；而：
#
#     self.target_modules[i]
#
# 表示第 i 个完整 DecoderLayer，不是第 i 个 Linear。随后
# get_named_linears(self.target_modules[i]) 才会递归找出该 DecoderLayer 内的
# q/k/v/o_proj、gate/up/down_proj 等 nn.Linear。
#
# 对 MoE 模型，self.target_modules[i] 仍是一个完整 DecoderLayer，只是其
# mlp 内还包含 router、多个 experts 及各 expert 的 Linear。
#
#
# 2. next(self.target_modules[i].parameters()).device 是什么意思？
#
# nn.Module.parameters() 返回一个“参数迭代器”，它默认会递归遍历当前模块及
# 所有子模块中注册的 nn.Parameter。可以把原代码拆开理解：
#
#     layer = self.target_modules[i]       # 第 i 个 DecoderLayer
#     parameter_iterator = layer.parameters()
#     first_parameter = next(parameter_iterator)
#     common_device = first_parameter.device
#
# next(iterator) 的意思是从迭代器中取出下一个元素。因为这里是刚创建的迭代器，
# 所以取到的是它产出的第一个 nn.Parameter，通常会是当前 DecoderLayer 中较早
# 注册的某个 weight，例如某个 attention projection 的 weight；具体是哪一个取决于
# 模型类的子模块注册顺序，业务逻辑不应该依赖它的名字。
#
# .device 再读取这个 Parameter 所在的设备，例如：
#
#     torch.device("cpu")
#     torch.device("cuda:0")
#     torch.device("cuda:1")
#
# 这行代码的目的不是使用“第一个参数”的数值，而是用它所在的 device 作为
# 整个 DecoderLayer 所在 device 的代表：
#
#     common_device = next(self.target_modules[i].parameters()).device
#
# 后续会把该层的 calibration input、attention_mask、position_ids、scale 等移动到
# common_device，避免执行当前层 forward 时发生 CPU/CUDA 或跨 GPU device mismatch。
#
# 该写法隐含两个假设：
#
#   1) DecoderLayer 至少注册了一个 nn.Parameter；否则 next(...) 会抛 StopIteration。
#   2) 同一个 DecoderLayer 的所有参数都在同一设备；如果一层内部参数被拆到多个
#      设备，第一个 Parameter 的 device 不能代表整层。
#
# 对本项目“以完整 DecoderLayer 为单位搬到一张 GPU”的量化流程，这两个假设通常
# 成立。注意 parameters() 不返回 register_buffer() 注册的 buffer；它只遍历
# nn.Parameter。如果还要检查 buffer 的设备，需要另外遍历 layer.buffers()。
# =============================================================================
