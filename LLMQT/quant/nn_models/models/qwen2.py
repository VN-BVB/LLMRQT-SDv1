import tqdm
from typing import List, Tuple
from quant.core.base import BaseModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer as OldQwen2DecoderLayer,
    Qwen2ForCausalLM as OldQwen2ForCausalLM,
)

class Qwen2ModelForCausalLM(BaseModelForCausalLM):
    layer_type = "Qwen2DecoderLayer"

    # @staticmethod
    # def fuse_layers(model: OldQwen2ForCausalLM):
    #     fuser = Qwen2Fuser(model)
    #     fuser.fuse_transformer()

    # awq/fp8 quantizer里面会用到
    # 看源码可知
    # self.layers = nn.ModuleList([Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
    @staticmethod
    def get_model_layers(model: OldQwen2ForCausalLM):
        return model.model.layers

    # no use
    # @staticmethod
    # def get_act_for_scaling(module: OldQwen2DecoderLayer):
    #     return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldQwen2ForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    # awq quantizer里面会用到
    @staticmethod
    def get_layers_for_scaling(module: OldQwen2DecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # GQA与MQA
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers


# =============================================================================
# Qwen2 AWQ 阅读笔记：model.model.layers、target_modules 与 scaling 配置
# =============================================================================
#
# 1. model.model.layers 到底是什么？
#
# model 是 Hugging Face 的 Qwen2ForCausalLM，主要层级可以简化理解为：
#
# Qwen2ForCausalLM                          # 带语言模型输出头的完整模型
# ├── model: Qwen2Model                    # Transformer 主干
# │   ├── embed_tokens
# │   ├── layers: nn.ModuleList            # 所有 DecoderLayer 都在这里
# │   │   ├── [0]: Qwen2DecoderLayer
# │   │   ├── [1]: Qwen2DecoderLayer
# │   │   └── ...
# │   ├── norm
# │   └── rotary_emb
# └── lm_head
#
# 因此，get_model_layers(model) 返回的 model.model.layers 通常是：
#
# nn.ModuleList([
#     Qwen2DecoderLayer(
#         input_layernorm=...,
#         self_attn=(q_proj, k_proj, v_proj, o_proj),
#         post_attention_layernorm=...,
#         mlp=(gate_proj, up_proj, down_proj),
#     ),
#     ...
# ])
#
# 它不是权重 Tensor，也不是 Linear 列表，而是“完整 DecoderLayer”的列表。
# 列表长度通常等于 model.config.num_hidden_layers。
#
#
# 2. self.target_modules 长什么样？
#
# AWQQuantizer 初始化时会执行类似：
#
#     self.target_modules = self.awq_model.get_model_layers(self.model)
#
# 对 Qwen2 而言，等价于：
#
#     self.target_modules = self.model.model.layers
#
# 所以：
#
#     self.target_modules[i]
#
# 是第 i 个 Qwen2DecoderLayer。例如它内部可以继续访问：
#
#     self.target_modules[i].self_attn.q_proj
#     self.target_modules[i].self_attn.k_proj
#     self.target_modules[i].self_attn.v_proj
#     self.target_modules[i].self_attn.o_proj
#     self.target_modules[i].mlp.gate_proj
#     self.target_modules[i].mlp.up_proj
#     self.target_modules[i].mlp.down_proj
#
# 这里保存的是原模型子模块的引用，不是深拷贝。因此，通过 target_modules
# 替换或修改某个 Linear，也会直接修改 self.model 中的对应模块。
#
#
# 3. next(self.target_modules[i].parameters()).device 是什么意思？
#
# module.parameters() 返回一个“参数迭代器”，递归遍历该 DecoderLayer 中的
# nn.Parameter（例如 q_proj.weight、k_proj.weight 等）。它不会一次性返回列表。
#
# next(iterator) 表示从迭代器中取出第一个元素。因此：
#
#     first_parameter = next(self.target_modules[i].parameters())
#     device = first_parameter.device
#
# 是用该层第一个参数所在的设备，代表整个 DecoderLayer 所在的设备，随后可将
# 校准输入移动到相同设备。它不是“取第一个子模块”，而是“取第一个参数”。
# 这种写法默认该层至少有一个参数，并且该 DecoderLayer 的参数位于同一设备。
#
#
# 4. input_feat 和 module_kwargs 是什么？
#
# input_feat 是量化器通过 forward hook 收集到的 Linear 输入，形式为 Dict：
#
#     {
#         "self_attn.q_proj": Tensor[..., hidden_size],
#         "self_attn.k_proj": Tensor[..., hidden_size],
#         "self_attn.v_proj": Tensor[..., hidden_size],
#         "self_attn.o_proj": Tensor[..., value_projection_size],
#         "mlp.gate_proj":    Tensor[..., hidden_size],
#         "mlp.up_proj":      Tensor[..., hidden_size],
#         "mlp.down_proj":    Tensor[..., intermediate_size],
#     }
#
# Tensor 前面的维度来自校准样本/序列，最后一维才是对应 Linear 的输入通道数。
#
# module_kwargs 是执行 DecoderLayer/Attention 时需要的额外关键字参数，例如
# attention_mask、position_ids、position_embeddings、cache_position 等。后续代码
# 会根据待调用模块的 forward 签名过滤无效参数，并非所有版本都会包含完全相同的键。
#
#
# 5. get_layers_for_scaling(...) 返回什么？
#
# 返回值 module_config 是 List[Dict]，不是 QuantConfig。每个 Dict 表示一次
# AWQ 最优缩放因子的搜索任务。Qwen2 一般返回下面 3 组；满足特定形状时返回 4 组：
#
# module_config = [
#     # (1) Attention 输入：同一个缩放因子同时作用于 q/k/v 的输入通道
#     {
#         "prev_op": module.input_layernorm,
#         "layers": [q_proj, k_proj, v_proj],
#         "inp": input_feat["self_attn.q_proj"],
#         "module2inspect": module.self_attn,
#         "kwargs": module_kwargs,
#     },
#
#     # (2) Attention 输出：只有 v_proj 与 o_proj 的 weight.shape 相等才加入
#     {
#         "prev_op": v_proj,
#         "layers": [o_proj],
#         "inp": input_feat["self_attn.o_proj"],
#     },
#
#     # (3) MLP 输入：同一个缩放因子同时作用于 gate_proj/up_proj
#     {
#         "prev_op": module.post_attention_layernorm,
#         "layers": [gate_proj, up_proj],
#         "inp": input_feat["mlp.gate_proj"],
#         "module2inspect": module.mlp,
#     },
#
#     # (4) MLP 输出：搜索 up_proj 与 down_proj 之间的缩放
#     {
#         "prev_op": up_proj,
#         "layers": [down_proj],
#         "inp": input_feat["mlp.down_proj"],
#     },
# ]
#
# Qwen2 常使用 GQA，v_proj 和 o_proj 的权重形状可能不同，所以第 (2) 组经常
# 不会加入。最终列表长度应以模型配置和实际 weight.shape 判断，不能固定认为是 4。
#
#
# 6. 每个 Dict 字段的作用
#
# prev_op:
#     当前 Linear 之前的算子。确定 scale 后，它负责吸收“除以 scale”的变换。
#     可能是 RMSNorm，也可能是上一层 Linear。
#
# layers:
#     要参与权重伪量化/缩放搜索的下游 Linear。AWQ 会把这些 Linear 权重的
#     输入通道乘以 scale；同组多个 Linear（如 q/k/v）共享同一组 scale。
#
# inp:
#     从真实校准数据中捕获的该组输入激活，用来比较缩放前后模块输出误差。
#
# module2inspect:
#     用来做前向计算和 MSE 对比的完整模块。例如 q/k/v 共同影响 Attention，
#     所以比较 self_attn 的输出；gate/up 共同影响 MLP，所以比较 mlp 的输出。
#     若省略且 layers 只有一个元素，搜索代码通常直接检查 layers[0]。
#
# kwargs:
#     调用 module2inspect.forward(...) 时需要附带的关键字参数，Attention 组需要。
#
#
# 7. module_config 后面怎样被使用？
#
# quantizer.py 中的逻辑可简化为：
#
#     module_config = get_layers_for_scaling(current_layer, input_feat, module_kwargs)
#     scales_list = [
#         self._search_best_scale(current_layer, **layer_config)
#         for layer_config in module_config
#     ]
#
# **layer_config 是 Python 字典解包。例如：
#
#     _search_best_scale(
#         current_layer,
#         prev_op=module.input_layernorm,
#         layers=[q_proj, k_proj, v_proj],
#         inp=...,
#         module2inspect=module.self_attn,
#         kwargs=module_kwargs,
#     )
#
# _search_best_scale 会尝试不同缩放比例，比较伪量化后的输出与原始浮点输出之间
# 的 MSE，选择误差最小的 best_scales。其结果通常包含：
#
#     (prev_op_name, layer_names, best_scales)
#
# 后续 apply_scale(...) 再将缩放真正折叠进 prev_op 与下游 Linear 的权重。
# 数学上利用的是等价变换：
#
#     y = x W
#       = (x / s) (diag(s) W)
#
# 即上一算子输出通道除以 s，同时下游权重对应输入通道乘以 s；浮点计算结果
# 理论上不变，但权重各通道的数值分布变得更适合低比特量化。
#
# best_scales 的长度等于该组 Linear 的输入通道数。例如 q/k/v、gate/up 组通常
# 等于 hidden_size；down_proj 组通常等于 intermediate_size，而不是输出维度。
# =============================================================================
