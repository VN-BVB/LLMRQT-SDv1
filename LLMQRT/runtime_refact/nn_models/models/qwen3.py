import tqdm
from typing import List, Tuple
from runtime_refact.core.base import BaseModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer as OldQwen3DecoderLayer,
    Qwen3ForCausalLM as OldQwen3ForCausalLM,
)
from runtime_refact.utils.fused_utils import fuse_qkv
from runtime_refact.nn_models.modules.nonlinear.transformer_layer import QwenFamilyLayer
from runtime_refact.nn_models.modules.nonlinear.model import LlamaLikeModel
from runtime_refact.nn_models.modules.nonlinear.norm import RMSNorm


class Qwen3ModelForCausalLM(BaseModelForCausalLM):
    layer_type = "Qwen3DecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldQwen3ForCausalLM):
        fuser = Qwen3Fuser(model) # 根据csrc kernel的支持情况来确定fusion能力
        fuser.do_fusion()

    @staticmethod
    def get_model_layers(model: OldQwen3ForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldQwen3DecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldQwen3ForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldQwen3DecoderLayer, input_feat, module_kwargs):
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

class Qwen3Fuser:
    def __init__(self, model: OldQwen3ForCausalLM):
        self.model = model

        self.qwen3_layers: List[Tuple[str, OldQwen3DecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "Qwen3DecoderLayer".lower() in module.__class__.__name__.lower()
        ]
        
    def do_fusion(self):
        layers = []
        config = self.model.config
        rope_theta = (
            config.rope_theta
            if hasattr(config, "rope_theta")
            else config.rope_parameters["rope_theta"]
        )
        # rope_theta = self.model.config.rope_theta  # 原代码：只兼容 Transformers 4.x

        module: OldQwen3DecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            
            if isinstance(qkv, tuple) and len(qkv) > 1: # sq and fp8
                q_proj = module.self_attn.q_proj
                k_proj = module.self_attn.k_proj
                v_proj = module.self_attn.v_proj                
            else: # only fuse qkv in awq
                q_proj = None
                k_proj = None
                v_proj = None
                
            norm_1 = RMSNorm(
                module.input_layernorm.weight, module.input_layernorm.variance_epsilon
            )
            norm_2 = RMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon,
            )
            layers.append(
                QwenFamilyLayer(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    q_proj=q_proj,
                    k_proj=k_proj,
                    v_proj=v_proj,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=rope_theta,
                    # rope_theta=self.model.config.rope_theta,  # 原代码：只兼容 Transformers 4.x
                    q_norm=module.self_attn.q_norm,
                    k_norm=module.self_attn.k_norm,
                    head_dim=self.model.config.head_dim,
                )
            )

        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            layers,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        # cursor建议删除，说多余了，但我先试一下留着
        setattr(self.model.model, "layers", self.model.model.layers)

