"""在线阶段：加载 FAISS 检索资料，再用 vLLM HTTP 或本地 AWQ 模型回答。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from generation_backends import DEFAULT_AWQ_MODEL, DEFAULT_LLMQRT_ROOT, create_backend
from bm25_retriever import BM25Retriever
from hybrid_retriever import reciprocal_rank_fusion
from rag_core import FaissRetriever
from reranker import BGEReranker, DEFAULT_RERANKER_MODEL

ROOT = Path(__file__).resolve().parent


def build_source_label(item: dict[str, Any]) -> str:
    """生成供模型原样引用的稳定来源标签。"""

    source_name = (
        item.get("doc_name")
        or item.get("title")
        or item.get("source_id")
        or item.get("id")
        or "未知资料"
    )
    label_parts = [f"资料名：{source_name}"]
    section_path = item.get("section_path") or []
    if section_path:
        label_parts.append(f"章节：{' > '.join(section_path)}")
    page = item.get("page")
    if page is not None:
        label_parts.append(f"页数：{page}")
    return f"[{' | '.join(label_parts)}]"


def build_rag_messages(question: str, retrieved: list[dict[str, Any]]) -> list[dict[str, str]]:
    """把检索结果及可复制的来源标签拼进 Prompt。"""

    context_parts = []
    for item in retrieved:
        context_parts.append(f"{build_source_label(item)}\n{item['text']}")
    context = "\n\n".join(context_parts)

    return [
        {
            "role": "system",
            "content": (
                "你是一个严格依据资料回答问题的助手。"
                "只能使用用户提供的资料，不要使用未在资料中出现的事实。"
                "回答前先判断资料中是否直接包含问题所需事实，并且只能选择以下一种输出："
                "（一）有直接证据时，严格输出两行：第一行“答案：<最短答案>”，"
                "第二行“依据：<原样复制的完整来源标签>”；"
                "（二）没有直接证据时，只输出“根据提供的资料无法确定”，不要附加猜测、计算、解释或来源标签。"
                "绝对禁止先给出一个答案再说无法确定，也禁止把不同概念的数字相加或据此推断答案。"
                "回答要简洁。每个事实结论后必须原样复制其依据资料前的完整来源标签，"
                "格式为[资料名：... | 章节：... | 页数：...]；没有章节时标签不包含章节字段。"
                "禁止写“资料1”“资料2”或[资料1]这类数字代号。"
            ),
        },
        {
            "role": "user",
            "content": f"请根据以下资料回答问题。\n\n{context}\n\n问题：{question}",
        },
    ]


def build_no_rag_messages(question: str) -> list[dict[str, str]]:
    """无 RAG 对照组：相同模型直接回答，用于观察 RAG 是否真的带来提升。"""

    return [
        {"role": "system", "content": "请简洁、准确地回答用户问题。"},
        {"role": "user", "content": question},
    ]


def build_eval_rag_messages(
    question: str, retrieved: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """自动评测专用 Prompt：只输出答案，避免客套话降低 EM/F1。"""

    context_parts = []
    for number, item in enumerate(retrieved, start=1):
        section = " > ".join(item.get("section_path", []))
        section_line = f"章节：{section}\n" if section else ""
        context_parts.append(f"[资料{number}]\n{section_line}{item['text']}")
    context = "\n\n".join(context_parts)
    return [
        {
            "role": "system",
            "content": (
                "你正在参加阅读理解测试。只根据资料回答，可使用与问题相同的语言。"
                "只输出最短答案本身，不要解释、不要复述问题、不要输出资料编号。"
                "如果资料中没有答案，输出：无法确定"
            ),
        },
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}\n答案："},
    ]


def build_eval_no_rag_messages(question: str) -> list[dict[str, str]]:
    """无 RAG 评测也要求相同的简短输出格式，保证对照公平。"""

    return [
        {
            "role": "system",
            "content": "只输出问题的最短答案本身，不要解释、不要复述问题。",
        },
        {"role": "user", "content": f"问题：{question}\n答案："},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "storage/ohr_bench_bge_m3")
    parser.add_argument("--embedding-model", help="默认读取 index_meta.json 中记录的模型")
    parser.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hybrid", action="store_true", help="启用 Dense + BM25 + RRF")
    parser.add_argument("--bm25-index", type=Path, help="默认 INDEX_DIR/bm25/")
    parser.add_argument("--dense-k", type=int, default=50)
    parser.add_argument("--bm25-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker", action="store_true", help="启用 Cross-Encoder 重排")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--candidate-k", type=int, default=50, help="Dense 阶段候选 Chunk 数")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--question", help="不填就进入连续提问模式")
    parser.add_argument(
        "--backend",
        choices=["vllm", "local-awq"],
        default="vllm",
        help="vllm=HTTP 服务；local-awq=直接加载自己的 AWQ checkpoint",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", help="vLLM 的 served model name；默认自动查询")
    parser.add_argument("--awq-model", type=Path, default=DEFAULT_AWQ_MODEL)
    parser.add_argument("--llmqrt-root", type=Path, default=DEFAULT_LLMQRT_ROOT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show-context", action="store_true", help="打印召回的 chunk")
    parser.add_argument("--retrieve-only", action="store_true", help="只看召回，不调用生成模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k 必须大于 0")
    if (args.reranker or args.hybrid) and args.candidate_k < args.top_k:
        raise ValueError("启用 Hybrid/Reranker 时 candidate-k 必须大于等于 top-k")
    for name in ("dense_k", "bm25_k", "rrf_k"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} 必须大于 0")
    if args.reranker and args.reranker_batch_size <= 0:
        raise ValueError("reranker-batch-size 必须大于 0")

    print("正在加载 FAISS 和 Embedding 模型……")
    retriever = FaissRetriever(
        args.index_dir,
        device=args.embedding_device,
        embedding_model=args.embedding_model,
    )
    bm25_retriever = None
    if args.hybrid:
        print("正在加载 BM25 索引……")
        bm25_retriever = BM25Retriever(
            args.index_dir,
            bm25_index=args.bm25_index,
            chunks=retriever.chunks,
        )
    reranker = None
    if args.reranker:
        print(f"正在加载 Reranker：{args.reranker_model} ({args.reranker_device})")
        reranker = BGEReranker(
            model_name=args.reranker_model,
            device=args.reranker_device,
            max_length=args.reranker_max_length,
        )
        if args.backend == "local-awq" and args.reranker_device == "cuda":
            print("警告：Reranker 与本地 AWQ 同驻 GPU，8 GiB 显存可能不足；可改用 --reranker-device cpu")
    client = None
    if not args.retrieve_only:
        client = create_backend(
            args.backend,
            api_base=args.api_base,
            model=args.model,
            awq_model=args.awq_model,
            llmqrt_root=args.llmqrt_root,
        )

    def answer(question: str) -> None:
        if bm25_retriever is not None:
            dense_results = retriever.search(question, top_k=args.dense_k)
            bm25_results = bm25_retriever.search(question, top_k=args.bm25_k)
            candidates = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                rrf_k=args.rrf_k,
                top_k=args.candidate_k,
            )
        else:
            dense_k = args.candidate_k if reranker is not None else args.top_k
            candidates = retriever.search(question, top_k=dense_k)
        retrieved = (
            reranker.rerank(
                question,
                candidates,
                top_k=args.top_k,
                batch_size=args.reranker_batch_size,
            )
            if reranker is not None
            else candidates[: args.top_k]
        )
        if args.show_context or args.retrieve_only:
            print("\n召回结果：")
            for item in retrieved:
                preview = item["text"].replace("\n", " ")[:160]
                if "rerank_score" in item and "rrf_score" in item:
                    score_text = (
                        f"rerank={item['rerank_score']:.4f} "
                        f"rrf_rank={item['fusion_rank']} rrf={item['rrf_score']:.6f}"
                    )
                elif "rerank_score" in item:
                    score_text = (
                        f"rerank={item['rerank_score']:.4f} "
                        f"dense_rank={item['dense_rank']} dense={item['dense_score']:.4f}"
                    )
                elif "rrf_score" in item:
                    score_text = (
                        f"rrf={item['rrf_score']:.6f} "
                        f"dense_rank={item.get('dense_rank', '-')} "
                        f"bm25_rank={item.get('bm25_rank', '-')}"
                    )
                else:
                    score_text = f"dense={item['score']:.4f}"
                print(f"  {item['rank']}. {score_text} {item['id']}  {preview}")
        if client is not None:
            response = client.chat(
                build_rag_messages(question, retrieved),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"\n回答：{response}\n")

    if args.question:
        answer(args.question)
        return

    print("输入问题开始查询；输入 exit 或 quit 结束。")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer(question)


if __name__ == "__main__":
    main()
