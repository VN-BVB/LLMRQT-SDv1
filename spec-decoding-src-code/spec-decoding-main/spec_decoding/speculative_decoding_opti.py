# SPDX-License-Identifier: Apache-2.0
"""
Optimized single-chain greedy speculative decode (KV-reuse).

与 speculative_decoding.speculative_greedy_decode 数学等价（greedy 无损），但避免
每轮对 target 整段重算：维护一份增量 target KV，verify 后用 crop 把缓存裁剪回
「实际接受的前缀」，只对 纠正 / bonus 这一个 token 再做一次单步前向。
!!!!!!!!!!!!此版本不纳入课程！！！！！！！！！！！！！
为什么 baseline 版要整段重算？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HF DynamicCache 是原地增长的：把同一个 cache 先喂 verify（g 个 draft）再喂
sync，会把含被拒绝 draft 的 KV 也留在缓存里，污染后续步骤。本文件的做法：

  1. verify：把 g 个 draft 追加进 past_t（长度 prefix→prefix+g），读出 logits；
  2. 按首个不匹配位 accept_len 判定；
  3. 裁剪：crop(prefix + accept_len) 丢掉被拒绝的 draft KV；
  4. 只对 1 个纠正 token（或全accept时的 bonus）再走一步，得到下一轮的 next_logit。

这样 target 每轮只前向 g + 1 个新 token（而非整段 L + g），KV 全程复用。

对齐关系（与 baseline 注释一致）：
  - d_0 的真值 = 上一轮留下的 next_logit；
  - d_j (j>=1) 的真值 = 本轮 logits_chunk[j-1]；
  - 全accept 的 bonus = logits_chunk[g-1]。

target / draft 须为 HF CausalLM，词表一致。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .speculative_decoding import SpeculativeConfig, _unwrap_hf


def _kv_seq_len(past: Any) -> int:
    """Current sequence length stored in a KV cache (DynamicCache or legacy tuple)."""
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    # legacy tuple: layer0 key [batch, heads, seq, head_dim]
    return int(past[0][0].shape[-2])


def _crop_kv(past: Any, length: int) -> Any:
    """Truncate a KV cache to length positions (drops rejected draft KV)."""
    if hasattr(past, "crop"):
        past.crop(length)
        return past
    # legacy tuple format: [(k, v), ...] with k/v shape [b, h, seq, d]
    return tuple(
        (k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous())
        for (k, v) in past
    )


def _step(
    model: PreTrainedModel,
    token_id: int,
    past: Any,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Any]:
    """Single-token forward on top of past; returns (last_logits[1,V], new_past)."""
    inp = torch.tensor([[token_id]], device=device, dtype=torch.long)
    total = _kv_seq_len(past) + 1
    out = model(
        inp,
        past_key_values=past,
        use_cache=True,
        attention_mask=torch.ones(1, total, device=device, dtype=torch.long),
    )
    return out.logits[:, -1, :], out.past_key_values


@torch.inference_mode()
def speculative_greedy_decode_opt(
    target: Union[PreTrainedModel, nn.Module],
    draft: Union[PreTrainedModel, nn.Module],
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    config: Optional[SpeculativeConfig] = None,
) -> torch.LongTensor:
    """KV-reuse greedy speculative decode (lossless; same output as the baseline)."""
    if config is None:
        config = SpeculativeConfig()

    target_m = _unwrap_hf(target)
    draft_m = _unwrap_hf(draft)
    device = input_ids.device
    gamma = max(1, int(config.num_draft_tokens))
    eos = config.eos_token_id

    out_ids = input_ids.clone()
    new_count = 0

    # ---------- Prefill：建立 target KV + 第一个待生成位置的 greedy 分布 ----------
    out_t = target_m(
        out_ids,
        attention_mask=torch.ones_like(out_ids, device=device),
        use_cache=True,
    )
    past_t = out_t.past_key_values
    next_logit = out_t.logits[:, -1, :]  # target greedy for the next position

    while new_count < max_new_tokens:
        if eos is not None and int(out_ids[0, -1].item()) == eos:
            break

        prefix_len = int(out_ids.shape[1])  # == current target KV length

        # ---------- Phase A — Draft 链（小模型，本轮独立 prefill + 自回归）----------
        att_full = torch.ones_like(out_ids, device=device)
        pref_d = draft_m(out_ids, attention_mask=att_full, use_cache=True)
        cur_past_d = pref_d.past_key_values
        draft_tokens: List[int] = [int(pref_d.logits[:, -1, :].argmax(dim=-1).item())]
        inp = torch.tensor([[draft_tokens[-1]]], device=device, dtype=torch.long)
        for _ in range(gamma - 1):
            if eos is not None and draft_tokens[-1] == eos:
                break
            step = draft_m(
                inp,
                past_key_values=cur_past_d,
                use_cache=True,
                attention_mask=torch.ones(1, 1, device=device, dtype=torch.long),
            )
            nxt = int(step.logits[:, -1, :].argmax(dim=-1).item())
            draft_tokens.append(nxt)
            cur_past_d = step.past_key_values
            inp = torch.tensor([[nxt]], device=device, dtype=torch.long)
        g = len(draft_tokens)

        # ---------- Phase B — Verify：把 g 个 draft 增量追加到 target KV ----------
        draft_batch = torch.tensor([draft_tokens], device=device, dtype=torch.long)
        out_v = target_m(
            draft_batch,
            past_key_values=past_t,
            use_cache=True,
            attention_mask=torch.ones(1, prefix_len + g, device=device, dtype=torch.long),
        )
        past_t = out_v.past_key_values  # 长度 prefix_len + g（含可能被拒绝的尾部）
        logits_chunk = out_v.logits[0]  # [g, V]

        target_preds = [int(next_logit[0].argmax(dim=-1).item())]
        for j in range(g - 1):
            target_preds.append(int(logits_chunk[j].argmax(dim=-1).item()))

        accept_len = 0
        for j in range(g):
            if target_preds[j] == draft_tokens[j]:
                accept_len += 1
            else:
                break

        # ---------- Phase C — Accept / 纠正 + KV 裁剪复用 ----------
        if accept_len == g:
            # 全中：保留全部 g 个 draft 的 KV，再走一个 bonus。
            bonus = int(logits_chunk[g - 1].argmax(dim=-1).item())
            # 只对 1 个纠正 token（或全 accept 的 bonus）再走一步单步前向，得到下一轮的 next_logit，供下一轮与draft生成的第一个token比对
            next_logit, past_t = _step(target_m, bonus, past_t, device=device)
            append = draft_tokens + [bonus]
        else:
            # 首错位：裁掉被拒绝的 draft KV，仅保留前 accept_len 个，再走纠正 token。
            past_t = _crop_kv(past_t, prefix_len + accept_len)
            corr = target_preds[accept_len]
            # 只对 1 个纠正 token（或全 accept 的 bonus）再走一步单步前向，得到下一轮的 next_logit，供下一轮与draft生成的第一个token比对
            next_logit, past_t = _step(target_m, corr, past_t, device=device)
            append = draft_tokens[:accept_len] + [corr]

        out_ids = torch.cat(
            [out_ids, torch.tensor([append], device=device, dtype=torch.long)], dim=1
        )
        new_count += len(append)

        if eos is not None and any(t == eos for t in append):
            break

    return out_ids


def speculative_greedy_decode_opt_from_strings(
    target: Union[PreTrainedModel, nn.Module],
    draft: Union[PreTrainedModel, nn.Module],
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    config: Optional[SpeculativeConfig] = None,
) -> Tuple[torch.LongTensor, str]:
    """encode → speculative_greedy_decode_opt → decode 便捷封装。"""
    enc = tokenizer(prompt, return_tensors="pt")
    device = next(_unwrap_hf(target).parameters()).device
    input_ids = enc["input_ids"].to(device)
    if config is None:
        config = SpeculativeConfig()
    if config.eos_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        config = replace(config, eos_token_id=tokenizer.eos_token_id)
    out = speculative_greedy_decode_opt(
        target, draft, input_ids, max_new_tokens=max_new_tokens, config=config
    )
    return out, tokenizer.decode(out[0], skip_special_tokens=True)


def _main() -> None:
    """Self-check: opt output must match plain greedy baseline (lossless)."""
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_id = os.environ.get("SPEC_E2E_TARGET", "JackFram/llama-160m")
    draft_id = os.environ.get("SPEC_E2E_DRAFT", "JackFram/llama-68m")
    prompt = os.environ.get("SPEC_E2E_PROMPT", "The capital of France is")
    max_new = int(os.environ.get("SPEC_E2E_MAX_NEW", "32"))
    gamma = int(os.environ.get("SPEC_E2E_GAMMA", "4"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(target_id)
    target = AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_id, torch_dtype=dtype).to(device).eval()

    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    plen = input_ids.shape[1]

    with torch.inference_mode():
        base = target.generate(
            input_ids, max_new_tokens=max_new, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=tok.eos_token_id,
        )
    base_cont = base[0, plen:]

    cfg = SpeculativeConfig(num_draft_tokens=gamma, eos_token_id=tok.eos_token_id)
    out, text = speculative_greedy_decode_opt_from_strings(
        target, draft, tok, prompt, max_new_tokens=max_new, config=cfg
    )
    spec_cont = out[0, plen:]
    m = min(int(base_cont.numel()), int(spec_cont.numel()))
    ok = bool(torch.equal(base_cont[:m].cpu(), spec_cont[:m].cpu()))
    print(f"[opt] text={text!r}")
    print(f"[opt] lossless_vs_greedy={ok} (compared {m} tokens)")
    if not ok:
        print(f"[opt] base={base_cont[:m].tolist()}")
        print(f"[opt] spec={spec_cont[:m].tolist()}")
        raise SystemExit(1)
    print("[opt] OK")


if __name__ == "__main__":
    _main()
