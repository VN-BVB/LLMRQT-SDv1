import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm

# 以下两个quantize函数已在quantization_utils.py中包含
# @torch.no_grad()
# def real_quantize_weight_per_channel_absmax(w):# fp in int out
#     # w: [out_channel, in_channel]
#     scales = w.abs().max(dim=1)[0] / 127
#     scales = scales.view(-1, 1)
#     if not w.is_cuda:
#         # half rounding is not supported on CPU
#         w = w.float()
#     # use inplace operation to save memory
#     w.div_(scales).round_().clamp_(-128, 127)
#     w_q = w.to(torch.int8)
#     return w_q, scales

# @torch.no_grad()
# def real_quantize_activation_per_token_absmax(t): # fp in int out
#     max_val = t.abs().max(dim=-1, keepdim=True)[0]
#     scales = torch.clamp(max_val, min=1e-8) / 127
#     t_q = t.div_(scales).round_().clamp_(-128, 127).to(torch.int8)#.mul_(max_val) 
#     return t

def get_act_scales(model, tokenizer, calib_data, num_samples=512, seq_len=512, split="validation"):
    # device = next(model.parameters()).device
    device = "cuda:0"
    model.eval().to(device)
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        # [batch * seq_len, hidden_dim]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu() # [hidden_dim]
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max) # 同一个 Linear 会被多个校准样本重复调用。
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)
    # 注册钩子函数到每个linear module
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )
    if isinstance(calib_data, str):
        if calib_data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", revision="main")
        else:
            dataset = load_dataset(calib_data, split=split)
    dataset = dataset.shuffle(seed=42)    
    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    # dataset = dataset.shuffle(seed=42)
    for i in tqdm(range(num_samples), desc="getting input act max to smooth"):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    data,
    num_samples=512,
    seq_len=512,
):
    model.eval()
    device = next(model.parameters()).device
    print("get static decoder layer scales in ",device)
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
            # act_dict[name]["input"] = torch.clamp(x.detach().abs().max(dim=-1, keepdim=True)[0], min=1e-8).cpu() # per channel act不可行
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )              
            # act_dict[name]["input"] = torch.maximum(
            #     act_dict[name]["input"], torch.clamp(x.detach().abs().max(dim=-1, keepdim=True)[0], min=1e-8).cpu()
            # )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
            # act_dict[name]["output"] = torch.clamp(y.detach().abs().max(dim=-1, keepdim=True)[0], min=1e-8).cpu()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )
            # act_dict[name]["output"] = torch.maximum(
            #     act_dict[name]["output"], torch.clamp(y.detach().abs().max(dim=-1, keepdim=True)[0], min=1e-8).cpu()
            # )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", revision="main")
        else:
            dataset = load_dataset(data, split=split)
    dataset = dataset.shuffle(seed=42)    
    # dataset = load_dataset("json", data_files=dataset_path, split="train")

    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()
        
    # import pdb;pdb.set_trace()
    decoder_layer_scales = [] # 包含了48层decoder layer的每个linear的input output scale
    is_opt = model.config.model_type == "opt"

    for idx in range(model.config.num_hidden_layers):
        
        scale_dict = {}
        if is_opt:
            scale_dict["attn_input_scale"] = ( # qkv 3个linear都用这个scale
                act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
                # act_dict[f"model.layers.{idx}.self_attn.q_proj"]["input"] / 127
            )
            scale_dict["q_output_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
                # act_dict[f"model.layers.{idx}.self_attn.q_proj"]["output"] / 127
            )
            scale_dict["k_output_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
                # act_dict[f"model.layers.{idx}.self_attn.k_proj"]["output"] / 127
            )
            scale_dict["v_output_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
                # act_dict[f"model.layers.{idx}.self_attn.v_proj"]["output"] / 127
            )
            scale_dict["out_input_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
                # act_dict[f"model.layers.{idx}.self_attn.o_proj"]["input"] / 127
            )
            scale_dict["fc1_input_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
            )
            scale_dict["fc2_input_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127
            )
        else:
            scale_dict["attn_input_scale"] = ( # qkv 3个linear都用这个scale
                act_dict[f"model.layers.{idx}.self_attn.q_proj"]["input"] / 127
            )
            scale_dict["q_output_scale"] = (
                act_dict[f"model.layers.{idx}.self_attn.q_proj"]["output"] / 127
            )
            scale_dict["k_output_scale"] = (
                act_dict[f"model.layers.{idx}.self_attn.k_proj"]["output"] / 127
            )
            scale_dict["v_output_scale"] = (
                act_dict[f"model.layers.{idx}.self_attn.v_proj"]["output"] / 127
            )
            scale_dict["out_input_scale"] = (
                act_dict[f"model.layers.{idx}.self_attn.o_proj"]["input"] / 127
            )
            scale_dict["gate_input_scale"] = (
                act_dict[f"model.layers.{idx}.mlp.gate_proj"]["input"] / 127
            )
            scale_dict["up_input_scale"] = (
                act_dict[f"model.layers.{idx}.mlp.up_proj"]["input"] / 127
            )
            scale_dict["down_input_scale"] = (
                act_dict[f"model.layers.{idx}.mlp.down_proj"]["input"] / 127
            )
        # 把每一层linear的input scale都存在于decoder_layer_Scales=list中
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict # act_dict表示每个linear的inputmax
