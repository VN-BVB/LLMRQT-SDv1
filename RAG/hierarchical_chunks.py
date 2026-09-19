"""按 tokenizer token 数构造页内精细 Child Chunk。"""

from __future__ import annotations

from typing import Any


def build_token_window_children(
    documents: list[dict[str, Any]],
    *,
    tokenizer_name: str,
    chunk_tokens: int = 240,
    overlap_tokens: int = 40,
    max_page_chars: int = 30000,
) -> list[dict[str, Any]]:
    """每页独立切分，确保 Child 不跨页，且可由 ``source_id`` 关联 Parent。"""

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens 必须大于 0")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens 必须满足 0 <= overlap < chunk_tokens")
    if max_page_chars <= 0:
        raise ValueError("max_page_chars 必须大于 0")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 transformers，无法按 token 切分") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("240-token Child 需要支持 offset_mapping 的 fast tokenizer")
    stride = chunk_tokens - overlap_tokens
    children: list[dict[str, Any]] = []

    from tqdm.auto import tqdm

    for document in tqdm(documents, desc="构造 240-token Child", unit="page"):
        original_text = str(document.get("text", ""))
        # OHR 中有一页因转换错误把同一段重复到 11 MB。页级 Child 不应让
        # 单个脏样本生成上万重复窗口；30k 字符仍覆盖正常页面和该页完整首轮内容。
        text = original_text[:max_page_chars]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
        if not offsets:
            continue
        child_number = 0
        for token_start in range(0, len(offsets), stride):
            token_end = min(token_start + chunk_tokens, len(offsets))
            char_start = int(offsets[token_start][0])
            char_end = int(offsets[token_end - 1][1])
            chunk_text = text[char_start:char_end].strip()
            if chunk_text:
                child = {
                    key: value
                    for key, value in document.items()
                    if key not in {"text", "headings"}
                }
                child.update(
                    {
                        "id": f"{document['id']}-child-{child_number:04d}",
                        "source_id": document["id"],
                        "parent_id": document["id"],
                        "child_index": child_number,
                        "text": chunk_text,
                        "start": char_start,
                        "end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                        "source_text_truncated": len(original_text) > max_page_chars,
                    }
                )
                children.append(child)
                child_number += 1
            if token_end >= len(offsets):
                break
    return children
