import tqdm
from typing import List, Tuple
from runtime_refact.core.base import BaseModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer as OldQwen2DecoderLayer,
    Qwen2ForCausalLM as OldQwen2ForCausalLM,
)
from runtime_refact.nn_models.modules.nonlinear.norm import RMSNorm
from runtime_refact.utils.fused_utils import fuse_qkv
from runtime_refact.nn_models.modules.nonlinear.transformer_layer import LlamaFamilyLayer
from runtime_refact.nn_models.modules.nonlinear.model import LlamaLikeModel
class Qwen2ModelForCausalLM(BaseModelForCausalLM):
    layer_type = "Qwen2DecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldQwen2ForCausalLM):
        fuser = Qwen2Fuser(model)
        fuser.do_fusion()

    @staticmethod
    def get_model_layers(model: OldQwen2ForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldQwen2DecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldQwen2ForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)

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
    
class Qwen2Fuser:
    def __init__(self, model: OldQwen2ForCausalLM):
        self.model = model

        self.qwen2_layers: List[Tuple[str, OldQwen2DecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "Qwen2DecoderLayer".lower() in module.__class__.__name__.lower()
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

        module: OldQwen2DecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            # import pdb;pdb.set_trace()
            # Qwen2DecoderLayer(
            #   (self_attn): Qwen2SdpaAttention(
            #     (q_proj): AWQLinear_GEMM(in_features=5120, out_features=5120, bias=True, w_bit=4, group_size=128)
            #     (k_proj): AWQLinear_GEMM(in_features=5120, out_features=1024, bias=True, w_bit=4, group_size=128)
            #     (v_proj): AWQLinear_GEMM(in_features=5120, out_features=1024, bias=True, w_bit=4, group_size=128)
            #     (o_proj): AWQLinear_GEMM(in_features=5120, out_features=5120, bias=False, w_bit=4, group_size=128)
            #     (rotary_emb): Qwen2RotaryEmbedding()
            #   )
            #   (mlp): Qwen2MLP(
            #     (gate_proj): AWQLinear_GEMM(in_features=5120, out_features=13824, bias=False, w_bit=4, group_size=128)
            #     (up_proj): AWQLinear_GEMM(in_features=5120, out_features=13824, bias=False, w_bit=4, group_size=128)
            #     (down_proj): AWQLinear_GEMM(in_features=13824, out_features=5120, bias=False, w_bit=4, group_size=128)
            #     (act_fn): SiLU()
            #   )
            #   (input_layernorm): Qwen2RMSNorm((5120,), eps=1e-06)
            #   (post_attention_layernorm): Qwen2RMSNorm((5120,), eps=1e-06)
            # )
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            if isinstance(qkv, tuple) and len(qkv) > 1: # 1. SQ/FP8 static 返回 3 个投影；2. AWQ 返回融合后的 AWQLinear_GEMM
                q_proj = module.self_attn.q_proj
                k_proj = module.self_attn.k_proj
                v_proj = module.self_attn.v_proj                
            else:
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
                LlamaFamilyLayer(# TODO, a better name might be better
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
                    # 以下是在qwen3.py里会有，qwen2.py没有
                    # q_norm=module.self_attn.q_norm,
                    # k_norm=module.self_attn.k_norm,
                    # head_dim=self.model.config.head_dim,
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
