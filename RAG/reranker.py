"""Cross-Encoder Reranker：对 Dense Retriever 的候选 Chunk 重新排序。"""

from __future__ import annotations

import gc
from typing import Any


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    """使用 BGE Cross-Encoder 对 ``(question, chunk)`` 文本对打分。"""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        if max_length <= 0:
            raise ValueError("reranker max_length 必须大于 0")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("缺少 torch/transformers，无法加载 Reranker") from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("reranker-device=cuda，但当前没有可用 CUDA")

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            # 新版 Transformers 已将 torch_dtype 参数弃用，统一使用 dtype。
            model_kwargs["dtype"] = torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **model_kwargs
        ).to(device)
        self.model.eval()

    @staticmethod
    def _passage(item: dict[str, Any]) -> str:
        """使用与 Dense 建库相近的元数据和正文组成 Passage。"""

        parts = [item.get("title", "")]
        if item.get("domain"):
            parts.append(str(item["domain"]))
        if item.get("section_path"):
            parts.append(" > ".join(item["section_path"]))
        parts.append(item["text"])
        return "\n".join(str(part) for part in parts if part).strip()

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int | None = None,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """重排一组候选，保留 Dense 分数并增加 ``rerank_score``。"""

        if not question.strip():
            raise ValueError("不能重排空问题")
        if batch_size <= 0:
            raise ValueError("reranker batch_size 必须大于 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("reranker top_k 必须大于 0 或为 None")
        if not candidates:
            return []

        passages = [self._passage(item) for item in candidates]
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            batch_passages = passages[start : start + batch_size]
            batch_questions = [question] * len(batch_passages)
            tokens = self.tokenizer(
                batch_questions,
                batch_passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            with self.torch.inference_mode():
                logits = self.model(**tokens).logits.reshape(-1)
            scores.extend(logits.float().cpu().tolist())

        results = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["retrieval_rank"] = int(item["rank"])
            item["retrieval_score"] = float(item["score"])
            # Dense 直连时记录 Dense 排名；Hybrid 输入则保留 RRF 中已有的
            # dense_rank，并另外记录 fusion_rank/fusion_score。
            if "rrf_score" in item:
                item["fusion_rank"] = int(item["rank"])
                item["fusion_score"] = float(item["score"])
            else:
                item["dense_rank"] = int(item["rank"])
                item["dense_score"] = float(item["score"])
            item["rerank_score"] = float(score)
            results.append(item)

        results.sort(key=lambda item: item["rerank_score"], reverse=True)
        limit = len(results) if top_k is None else min(top_k, len(results))
        results = results[:limit]
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return results

    def rerank_many(
        self,
        questions: list[str],
        candidate_rows: list[list[dict[str, Any]]],
        *,
        top_k: int | None = None,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """逐题重排候选；批量评测时显示问题级进度。"""

        if len(questions) != len(candidate_rows):
            raise ValueError("questions 与 candidate_rows 数量不一致")

        from tqdm.auto import tqdm

        rows = zip(questions, candidate_rows)
        progress = tqdm(
            rows,
            total=len(questions),
            desc="Reranker 重排",
            unit="query",
            dynamic_ncols=True,
            disable=not show_progress or len(questions) <= 1,
        )
        return [
            self.rerank(question, candidates, top_k=top_k, batch_size=batch_size)
            for question, candidates in progress
        ]

    def unload(self) -> None:
        """批量重排结束后主动释放模型，给生成模型腾出显存。"""

        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
