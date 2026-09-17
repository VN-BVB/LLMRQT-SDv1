import torch
import torch.nn as nn
from typing import List
from runtime_refact.utils import fused_utils
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)
from .transformer_layer import LlamaFamilyLayer

class LlamaLikeModel(nn.Module):
    """
    LlamaLikeModel旨在被Llama, Mistral等相似架构的模型结构复用.
    """

    def __init__(self, vocab_size, layers, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.layers: List[LlamaFamilyLayer] = nn.ModuleList(layers)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @property
    def embed_tokens(self):
        return self.embedding

    # @property
    # def layers(self):
    #     return self.layers

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        # 从transformers 4.35.0之后,input_ids包含了全部上下文的token, 但是在decode阶段, 每次只会传入一个token, 所以需要在这里做处理,提取最后一个last_forward_num_tokens
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape
        # 更新当前fwd的cache start pos
        # 如果是prefill，那么start pos为seqlen，如果是decode，那么每次fwd就start pos+1
        # 如果kv cache容量不够，则需要删除一些
        fused_utils.prepare_cache(self.layers, seqlen)

        h = self.embedding(input_ids) # cuda0
        
        for id, layer in enumerate(self.layers):
            # 虽然layer分发到了不同的device，但是下面一行把hiddenstates to到了其对应的device
            h = h.to(layer.device) # bug0: 第0个iter可以通过，但是第1个iter，h变成了tensor(..., device='meta', size=(1, 61, 5120), dtype=torch.float16)，报错NotImplementedError: Cannot copy out of meta tensor; no data!
            h = layer(h) # bug1: 第0个iter的输出不是nan，但是第1个iter，全是nan, 造成的地方在于attn
        # import pdb
        # pdb.set_trace()
        h = self.norm(h)
        # huggingface中的model.generate()方法，所以这里需要返回BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )

