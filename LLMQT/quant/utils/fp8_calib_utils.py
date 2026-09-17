from datasets import load_dataset


# 后续修改：增加 calib_data 参数，使 FP8 静态校准既支持内置文本列表，也支持 Hugging Face 数据集名称。
def prepare_calib_tokens(tokenizer, device, num_samples, max_seq_len, calib_data=None):
    if isinstance(calib_data, (list, tuple)):
        # 后续修改：对文本列表进行重复和截断，精确生成 num_samples 条离线校准样本。
        if not calib_data:
            raise ValueError("FP8 calibration texts cannot be empty.")
        repeats = (num_samples + len(calib_data) - 1) // len(calib_data)
        texts = list(calib_data) * repeats
        texts = texts[:num_samples]
    else:
        # 后续修改：未传文本列表时保留原 UltraChat 默认行为，同时允许指定其他数据集。
        dataset_name = calib_data or "HuggingFaceH4/ultrachat_200k"
        ds = load_dataset(dataset_name, split="train_sft")
        ds = ds.shuffle(seed=42).select(range(num_samples))
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False)
            for messages in ds["messages"]
        ]

    tokenizer.pad_token_id = tokenizer.eos_token_id
    # 后续修改：统一对文本列表或数据集转换出的文本执行分词。
    calibration_tokens = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        add_special_tokens=False,
    ).input_ids.to(device)
    return calibration_tokens
