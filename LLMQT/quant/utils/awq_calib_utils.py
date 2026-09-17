import torch
import logging
from typing import List, Union
from datasets import load_dataset


def get_calib_dataset( # awq use
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", revision="main")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42) # 打乱数据集，设置随机种子为42以确保结果可复现

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to max sequence length
    if not samples:
        raise ValueError("No calibration samples were collected. Try reducing max_seq_len or providing shorter texts.")
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")
    if n_split == 0:
        return [cat_samples]
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]


# =============================================================================
# AWQ 校准数据阅读笔记：get_calib_dataset() 的完整处理逻辑
# =============================================================================
#
# 1. 这个函数的目的是什么？
#
# AWQ 属于训练后量化（PTQ）。它不需要用校准数据反向传播或更新模型参数，但需要
# 少量有代表性的文本做前向计算，以观察各个 Linear 的真实输入激活分布。后续会用
# 这些激活搜索：
#
#     - AWQ best scale：怎样缩放权重/激活能让伪量化输出误差更小；
#     - AWQ best clip：权重裁剪到什么阈值时量化误差更小。
#
# 所以本函数只负责把原始文本或 token id 整理成模型可用的校准 token blocks，
# 不负责计算 loss，也没有 label，更不会在这里执行真正的 INT4 权重量化。
#
#
# 2. data 支持哪三种输入？
#
# (1) data="pileval"
#
#     会加载：
#
#         mit-han-lab/pile-val-backup 的 validation split
#
#     这是 AWQ 常用的 Pile 验证集备份。这里固定 revision="main"。
#
# (2) data="某个 Hugging Face 数据集名称"
#
#     会执行：
#
#         load_dataset(data, split=split)
#
#     并从每条数据的 text_column 字段取文本。默认字段名是 "text"。
#
# (3) data 是 Python list
#
#     List[str]：每个元素是一条原始文本，函数先包装成：
#
#         [{"text": "第一条文本"}, {"text": "第二条文本"}, ...]
#
#     List[List[int]]：每个内层 list 已经是 token ids，例如：
#
#         [[1, 234, 56], [1, 789, 10, 11], ...]
#
#     这种输入不再调用 tokenizer.encode()。
#
# 注意：当前实现直接访问 data[0]，因此传入空 list 会触发 IndexError；传入原始文本
# 时 tokenizer 不能为 None，否则执行 tokenizer.encode(line) 时会报错。
#
#
# 3. 为什么要 dataset.shuffle(seed=42)？
#
# 从 Hugging Face 加载的数据集会先随机打乱，避免总是只取数据集开头的相邻样本。
# 固定 seed=42 可以保证每次选到的顺序一致，便于复现实验结果。
#
# 只有 data 是数据集名称时才会 shuffle；调用者直接传入的 Python list 保持原顺序。
#
#
# 4. for 循环怎样筛选一个样本？
#
# 对 dataset 中的每个元素：
#
#     已是 List[int]：直接把它当作 line_encoded；
#     是数据集字典：取 data[text_column]，strip() 后 tokenizer.encode()。
#
# 得到 token ids 后进行两次过滤：
#
#     if len(line_encoded) > max_seq_len:
#         continue
#
#     if sample.numel() == 0:
#         continue
#
# 第一条很重要：超过 max_seq_len 的单条文本会被整个丢弃，而不是在这里截断。
# 空文本编码后若没有 token，也会被丢弃。
#
# 合格 token ids 被转换为：
#
#     sample = torch.tensor([line_encoded])
#
# 外层的 [] 增加 batch 维，所以单条 sample 的 shape 是：
#
#     [1, 当前样本 token 数]
#
# n_run 只统计“通过过滤并加入 samples 的原始样本数”。收集到 n_samples 条合格
# 原始样本后停止遍历；它不代表函数最终返回的 token block 数量。
#
#
# 5. 为什么先拼接，再按 max_seq_len 切块？
#
# 假设筛选后得到三条 token 序列，shape 分别是：
#
#     [1, 100]、[1, 200]、[1, 400]
#
# torch.cat(samples, dim=1) 沿序列维拼接：
#
#     cat_samples.shape = [1, 700]
#
# 原始文本之间的边界在此消失，它们变成一条连续 token 流。这样不用 padding，可以
# 尽量把短文本 token 填满到固定长度的校准块中。
#
# 接着使用整除计算完整块数量：
#
#     n_split = total_token_count // max_seq_len
#
# 若 max_seq_len=512，上例 n_split=700//512=1，最终只返回：
#
#     [cat_samples[:, 0:512]]             # 一个 shape=[1, 512] 的 Tensor
#
# 剩余的 188 个 token 会被舍弃。当前代码不会返回最后一个不足 max_seq_len 的尾块，
# 也不会给尾块 padding。
#
# 唯一例外是总 token 数本身小于 max_seq_len，此时 n_split==0，函数返回：
#
#     [cat_samples]                       # 一个 shape=[1, total_tokens] 的短块
#
# 因此返回值始终是 List[Tensor]，但列表长度通常是：
#
#     floor(合格样本的 token 总数 / max_seq_len)
#
# 而不是 n_samples。n_samples=128 表示最多采用 128 条合格原始文本，不表示一定
# 返回 128 个 `[1, max_seq_len]` Tensor。
#
#
# 6. 返回结果在 AWQQuantizer.init_quant() 中怎样继续流动？
#
# 调用关系是：
#
#     samples = get_calib_dataset(...)
#     samples = torch.cat(samples, dim=0)
#
# 本函数的每个完整块 shape 为 `[1, max_seq_len]`，沿 dim=0 拼接后得到：
#
#     samples.shape = [校准块数量, max_seq_len]
#                   = [batch_size, sequence_length]
#
# 然后 samples 被传给完整模型进行一次校准前向。init_quant() 会暂时用 Catcher
# 包装第 0 个 DecoderLayer，在第 0 层真正计算前截获 hidden_states：
#
#     token ids:          [B, S]
#             ↓ embedding / rotary 等模型前处理
#     layer-0 input act:  [B, S, hidden_size]
#
# 这个 layer-0 input act 被保存到 self.inps。随后 quantize() 逐层运行：当前层输出
# 又成为下一层输入，并通过 hooks 收集 q_proj、k_proj、v_proj、o_proj、gate_proj、
# up_proj、down_proj 等 Linear 的 input_feat，用于 Scale/Clip 搜索。
#
#
# 7. 用一句话概括整个函数
#
#     加载/接收数据
#         → 打乱（仅 Hugging Face 数据集）
#         → 文本 tokenize
#         → 丢弃过长和空样本
#         → 最多保留 n_samples 条合格样本
#         → 沿 token 维拼成一条长序列
#         → 切成若干 max_seq_len 长度的校准块
#         → 交给 AWQ 前向采集激活。
#
# 面试时应特别说明：这是 representative calibration data，不是训练数据；校准阶段
# 不做反向传播。数据质量和覆盖范围会影响激活统计，从而影响 Scale/Clip 搜索质量。
# =============================================================================
