# 一个适合入门的标准 RAG：Qwen3-1.7B AWQ + BGE + FAISS

本文中的 BGE 是 **BAAI General Embedding（智源通用嵌入模型）**，BAAI 是
Beijing Academy of Artificial Intelligence（北京智源人工智能研究院）；FAISS 是
Facebook AI Similarity Search（向量相似度搜索库）；AWQ 是 Activation-aware Weight
Quantization（激活感知权重量化）。

这个项目把 RAG 拆成两个容易理解的阶段：

```text
离线阶段（偶尔运行）
文档 JSONL -> 切块 -> BGE 向量 -> index.faiss + chunks.jsonl

在线阶段（每次提问）
问题 -> BGE 问题向量 -> FAISS Top-K -> 拼接 Prompt
                                    ├-> vLLM HTTP API
                                    └-> 直接加载本地 LLMQRT AWQ checkpoint
```

这里的“在线”是指**用户提问时的在线服务阶段**，不是联网搜索。完成首次下载后，
知识库检索和生成都可以在本机离线运行。

如果要直接按照当前实验结论搭建，请先看[第 11 节“严格分阶段消融与最终选型”](#11-严格分阶段消融与最终选型)，
再按[第 10 节“综合权衡后的完整工作流”](#10-综合权衡后的完整工作流)执行；前面的章节保留基础原理、
单模块用法和历史实验过程。

## 当前默认数据集：OHR-Bench

[OHR-Bench](https://github.com/opendatalab/OHR-Bench) 包含论文、财务报告、说明书、法律文件、
教材和新闻等混合长文档。当前代码默认使用已经转换好的 OHR-Bench：

- `data/ohr_bench/documents.jsonl`：8,259 个有效页面；
- `data/ohr_bench/questions.jsonl`：8,456 条可评测问题；
- 每题带一个或多个证据页面，可测 Hit、Recall、MRR 和 nDCG；
- 包含表格、公式、图表与 OCR 噪声场景。

CMRC 2018 相关脚本和数据仍然保留，仅作为旧的中文小规模冒烟对照，不再是默认流程。

## 目录说明

```text
rag/
├── 标准RAG框架.md            # 完整架构、选型、数据格式和端到端流程
├── ingest_documents.py      # PDF/DOCX/MD/TXT -> 原始 documents.jsonl
├── prepare_cmrc2018.py      # 下载/整理公开数据集
├── prepare_ohr_bench.py     # OHR-Bench Parquet/QA -> 页面文档与评测 JSONL
├── OHR-Bench使用.md         # 混合长文档数据集的转换、建库和评测
├── offline_build.py         # 离线建 FAISS 索引
├── online_rag.py            # 在线检索并选择一种生成后端回答
├── reranker.py              # BGE Cross-Encoder 候选重排
├── bm25_retriever.py        # bm25s 关键词检索
├── hybrid_retriever.py      # Dense + BM25 的 RRF 融合
├── build_bm25.py            # 从现有 Chunk 建立 BM25 索引
├── generation_backends.py   # vLLM HTTP / 本地 AWQ 两种生成接口
├── evaluate_retrieval.py    # 不启动 LLM，只测检索
├── evaluate_generation.py   # 比较无 RAG 与有 RAG 的答案
├── evaluate_faithfulness.py # 逐事实忠诚度、幻觉率与综合质量评测
├── evaluate_bge_m3_modes.py # Dense/Sparse/ColBERT/BM25 小语料消融
├── build_hierarchical_index.py       # 生成 Parent/Child 摘要与预设问题并建索引
├── prepare_hierarchical_retrieval.py # 两层检索、RRF、重排并保存固定缓存
├── query_transform.py                # 查询重写、Step-back 回退查询与问题拆解
├── generate_query_variants.py        # 用本地 LLM 批量生成固定查询变体
├── prepare_query_variant_retrieval.py# 查询变体的 Dense/BM25/RRF/Reranker 消融
├── rag_core.py              # 切块、Embedding、FAISS 公共代码
├── requirements.txt
├── data/                    # 运行后生成的文档和问题 JSONL
├── storage/                 # 运行后生成的 FAISS 索引
└── results/                 # 评测结果
```

完整设计说明和自有文件摄取方式见 [`标准RAG框架.md`](标准RAG框架.md)。最短的自有文档流程：

```bash
python ingest_documents.py /path/to/documents --output data/custom/documents.jsonl
python offline_build.py \
  --documents data/custom/documents.jsonl \
  --output-dir storage/custom_bge_m3
python online_rag.py \
  --index-dir storage/custom_bge_m3 \
  --backend local-awq --question "你的问题"
```

## 0. 安装依赖

现有 vLLM 环境已经有 PyTorch、Transformers 和 NumPy；当前 NVIDIA/Linux 环境还会安装
`faiss-gpu`、`pypdf` 和 `python-docx`（纯 CPU 机器需把 requirements 中该项改成
`faiss-cpu`）：

```bash
cd /home/lyc/workspace/rag
/home/lyc/workspace/vllm/.venv/bin/python -m pip install -r requirements.txt
```

默认 Embedding 模型是 `BAAI/bge-m3`，使用 1024 维 dense 向量，查询不添加 instruction
前缀。第一次运行会从 Hugging Face 下载，之后会走本地缓存。Embedding 默认在 CPU 上运行；
只进行离线建库时可以传 `--device cuda` 加速。切换模型后必须重建 FAISS 索引。

## 1. 准备 OHR-Bench

```bash
python prepare_ohr_bench.py
```

转换器读取已经下载的 Parquet 和官方 QA：

```text
data/ohr_bench/raw/OHR-Bench_v2.parquet
data/ohr_bench/raw/qas_v2.json
```

输出：

- `data/ohr_bench/documents.jsonl`：知识库，每行一个可检索页面；
- `data/ohr_bench/questions.jsonl`：问题、答案和正确证据页面 ID；
- `data/ohr_bench/dataset_meta.json`：数量、领域和页码约定。

完整建库前可按每个领域抽取 2 份逻辑文档做冒烟测试：

```bash
python prepare_ohr_bench.py \
  --max-documents-per-domain 2 \
  --output-dir data/ohr_bench_smoke
```

## 2. 离线 RAG：建立 JSONL + FAISS 索引

```bash
python offline_build.py --device cuda --batch-size 16
```

会生成：

```text
storage/ohr_bench_bge_m3/
├── chunks.jsonl     # chunk 原文和来源；方便人查看
├── index.faiss      # 向量索引；供程序快速搜索
├── index_meta.json  # Embedding 模型、切块大小等可复现配置
└── bm25/            # bm25s 关键词索引
```

这里使用 `IndexFlatIP`：因为向量已经 L2 归一化，所以内积等价于余弦相似度。它是精确
搜索，适合小数据集。百万级文档才需要考虑 IVF/HNSW 等近似索引。

先不启动大模型，单独看看检索结果：

```bash
python online_rag.py \
  --retrieve-only --show-context \
  --question "What was the total amount of nonaccrual loans retained as of March 31, 2021?"
```

## 3. 先评测检索是否有效

```bash
python evaluate_retrieval.py --embedding-device cuda --k 1 3 5 10
```

重点看：

- `Hit@K`：前 K 个结果中是否至少有正确文档；
- `Recall@K`：所有正确文档中召回了多少；
- `MRR@K`：第一篇正确文档排得是否靠前；
- `nDCG@K`：整体相关文档排序质量。

Dense 结果默认保存在 `results/ohr_bench_retrieval_dense/`。启用 Reranker 时，同一次运行会同时
报告 Dense 与重排后的指标，并写入 `results/ohr_bench_retrieval_reranker/`：

```bash
python evaluate_retrieval.py \
  --embedding-device cpu \
  --reranker --reranker-device cuda \
  --candidate-k 50 --k 1 3 5 10 \
  --limit 50
```

结果还包含 Dense/Reranker 的每题耗时、指标差值和 chunk 级排名明细。Reranker 可能改善召回，
也可能把正确结果降级，因此应以完整固定测试集的指标为准。

改 chunk size、overlap、Embedding、候选数或 top-k 后，
用同一批问题重跑，这才知道修改是否真正改善了检索。

仓库中原有的数值来自小规模 BGE-small 冒烟测试，只用于证明数据映射和评测链路可运行。
BGE-M3 完整基线应在新索引建立后重新记录，不能沿用旧模型的结果。

### Dense + BM25 + RRF

BM25 复用现有 `chunks.jsonl`，只需建立一次关键词索引：

```bash
python build_bm25.py --index-dir storage/ohr_bench_bge_m3
```

一次评测同时报告 Dense、BM25、RRF 和 RRF+Reranker：

```bash
python evaluate_retrieval.py \
  --hybrid --dense-k 50 --bm25-k 50 --rrf-k 60 \
  --candidate-k 50 \
  --reranker --reranker-device cuda \
  --k 1 3 5 8 10 --limit 100
```

在线使用相同链路时给 `online_rag.py` 或 `evaluate_generation.py` 增加 `--hybrid` 及上述
Dense/BM25/RRF 参数即可。

## 4. 在线生成方式一：vLLM HTTP API

另开一个终端：

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
PATH=/home/lyc/miniconda3/envs/vllm/bin:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
/home/lyc/workspace/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve \
  /home/lyc/workspace/lastdata/eagle3/qwen3_1_7b_awq_eagle3_benchmark/checkpoints/Qwen3-1.7B-W4A16-AWQ \
  --served-model-name qwen3-awq \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.60
```

这里显式使用 `VLLM_USE_V2_MODEL_RUNNER=0`，因为当前 WSL 环境不支持 V2 runner 所需的
UVA；显式 PATH 则保证 vLLM 的首次 JIT 能找到当前环境已经安装的 `ninja`。

如果显存不足，先确保没有其他模型进程占用 GPU，或适当降低
`--gpu-memory-utilization`。RAG 客户端默认让 BGE 使用 CPU，因此不会额外占用显存。

## 5. 通过 vLLM 接口提问

单次提问：

```bash
python online_rag.py \
  --backend vllm \
  --question "What was the total amount of nonaccrual loans retained as of March 31, 2021?" \
  --top-k 3 \
  --show-context
```

连续提问：

```bash
python online_rag.py --top-k 3 --show-context
```

代码会自动访问 `http://127.0.0.1:8000/v1/models` 获取模型名，也可以显式传入
`--model qwen3-awq`。

## 6. 在线生成方式二：直接调用自己的 AWQ 1.7B

这种方式不启动 vLLM 服务。程序直接使用你量化时的 LLMQRT 接口加载 checkpoint：

```bash
python online_rag.py \
  --backend local-awq \
  --awq-model /home/lyc/workspace/lastdata/eagle3/qwen3_1_7b_awq_eagle3_benchmark/checkpoints/Qwen3-1.7B-W4A16-AWQ \
  --question "What was the total amount of nonaccrual loans retained as of March 31, 2021?" \
  --top-k 3 \
  --show-context
```

内部使用的是你 benchmark 已经验证过的：

```python
AutoQuantForCausalLM.from_quantized(...)
```

然后执行带 KV cache 的 greedy decode。它不是把请求转发给 vLLM，因此可以公平对比
“vLLM Marlin 后端”和“自己的 LLMQRT AWQ kernel 后端”的答案及性能。两种模式不能同时
占用同一张 8 GiB GPU；使用 `local-awq` 前应先关闭 vLLM 服务。

## 7. 验证“加 RAG 是否真的更好”

先用 50 条问题做快速对照：

```bash
python evaluate_generation.py --limit 50 --top-k 3 --model qwen3-awq
```

直接加载自己的 AWQ 后端时：

```bash
python evaluate_generation.py \
  --backend local-awq \
  --limit 50 \
  --top-k 3
```

对每一道题，脚本会让**同一个 AWQ 模型**分别执行：

1. 无 RAG：只看问题直接回答；
2. 有 RAG：读取 FAISS 召回的 Top-K chunk 后回答。

最终报告：

- 无 RAG 的 EM/F1；
- 有 RAG 的 EM/F1；
- `RAG - no RAG` 提升量；
- 检索 `Hit@K`。

输出在 `results/ohr_bench_generation/`。EM 对生成模型较严格，语义相同但表达不同也可能无法
完全匹配，所以同时看字符级 F1，并人工抽查
`predictions.jsonl`。

## 建议的第一轮实验

固定模型、Prompt 和数据集，只改一个变量：

| 实验 | chunk size | overlap | top-k |
|---|---:|---:|---:|
| A | 300 | 50 | 3 |
| B | 500 | 80 | 3 |
| C | 800 | 120 | 3 |
| D | 500 | 80 | 1/5/10 |

每次先跑 `evaluate_retrieval.py`。只有检索明显改善后，再跑较慢的
`evaluate_generation.py`。不要同时修改切块、Embedding、Prompt 和模型，否则无法判断收益
来自哪里。

## 如何换成你自己的文档

准备一个 JSONL 文件，每行至少有 `id` 和 `text`：

```json
{"id":"doc-001","title":"产品说明","text":"这里是完整正文……"}
{"id":"doc-002","title":"常见问题","text":"这里是另一篇正文……"}
```

然后执行：

```bash
python offline_build.py \
  --documents /path/to/your_documents.jsonl \
  --output-dir storage/my_documents

python online_rag.py \
  --index-dir storage/my_documents \
  --question "你的问题"
```

如果还要自动评测，就再准备与 `data/ohr_bench/questions.jsonl` 相同格式的问题集，其中
`gold_doc_ids` 必须对应 documents 中的 `id`。

## 8. Parent Summary → Child 的层次化检索

当前进阶方案使用两层 Chunk，并在离线阶段为两层分别生成可能被用户提出的问题：

```text
页面原文
├─ Parent Summary：每页一个摘要 + 3 个预设问题
└─ Child Chunk：240 BGE token，overlap 40 token + 2 个预设问题

Query
  ├─ Parent Summary Dense（稠密向量检索）
  ├─ Parent Question Dense（预设问题向量检索）
  └─ Parent Summary+Question BM25（Best Matching 25，关键词检索）
                   ↓ RRF（Reciprocal Rank Fusion，倒数排名融合）
                   ↓ BAAI/bge-reranker-v2-m3 Cross-Encoder（交叉编码重排器）
              选出 Parent 页面范围
  ├─ 范围内 Child Text Dense
  ├─ 范围内 Child Question Dense
  └─ 范围内 Child Text+Question BM25
                   ↓ RRF → Reranker → Top-K → Qwen3-1.7B AWQ
```

离线生成可断点续跑，`children.partial.jsonl` 已完成的行不会重新生成：

```bash
python build_hierarchical_index.py generate \
  --backend vllm --api-base http://127.0.0.1:8000/v1 --model qwen3-1.7b-awq \
  --parent-question-count 3 --child-question-count 2 \
  --parent-batch-size 16 --parent-workers 16 \
  --child-batch-size 64 --child-workers 32 \
  --child-tokens 240 --child-overlap-tokens 40

python build_hierarchical_index.py index \
  --embedding-device cuda --embedding-batch-size 48 --overwrite
```

最终目录不是把问题拼到正文后只建一个向量，而是保留可解释的三条独立检索路由：

```text
storage/ohr_bench_hierarchical_questions/
├── hierarchy_meta.json
├── parent/
│   ├── chunks.jsonl              # Summary、原页面信息和 generated_questions
│   ├── index.faiss               # Summary 文本向量
│   ├── questions/index.faiss     # Parent 预设问题向量
│   └── bm25_questions/           # Summary + 问题关键词索引
└── child/
    ├── chunks.jsonl              # 240-token 正文和 generated_questions
    ├── index.faiss               # Child 正文向量
    ├── questions/index.faiss     # Child 预设问题向量
    └── bm25_questions/           # Child 正文 + 问题关键词索引
```

### FAISS GPU 口径

当前 Linux/NVIDIA 环境使用官方 `faiss-gpu 1.15.1`。`--embedding-device cuda` 同时表示：

- BGE-M3 在 GPU 上编码 Query；
- 不带过滤条件的 `IndexFlatIP` 全局搜索迁移到 `GpuIndexFlat`；
- 多个 GPU 索引共享 256 MiB 临时区，适配 8 GiB 显存；
- Parent 限定范围内的 Child 搜索使用 CPU 索引副本执行精确 `IDSelector` 过滤。

最后一点不是退回全 CPU：层次化方案同时保留 CPU/GPU 索引，全局粗检索走 GPU，必须按每题
动态 Parent 集合过滤的局部精检索走 CPU，避免不同 FAISS GPU 版本的过滤器兼容问题。

### 两阶段评测口径

检索和答案生成必须分开报告，`MRR` 是 Mean Reciprocal Rank（平均倒数排名），不是 `MMR`：

| 阶段 | 指标 | 含义 |
|---|---|---|
| 检索 | Hit@K | Top-K 是否至少包含一个正确证据页 |
| 检索 | Precision@K | Top-K 中正确证据页比例 |
| 检索 | Recall@K | 标注证据页被召回的比例 |
| 检索 | MRR@K | 第一条正确证据出现得是否靠前 |
| 检索 | nDCG@K | 多个证据在整个排名中的质量 |
| 生成 | EM | 最短答案是否与标准答案完全一致 |
| 生成 | Answer Precision/Recall/F1 | 预测答案与标准答案的字符重叠 |
| 生成 | Faithfulness | 回答中的原子事实有多少可由检索证据支持 |
| 生成 | Hallucination rate | `1 - Faithfulness` |
| 生成 | Answer relevance/correctness | LLM judge 对切题程度和答案正确性的评分 |
| 综合 | F1–Faithfulness H-mean | F1 与忠实度的调和平均；任一项低都会拉低总分 |
| 性能 | mean/P50/P95 latency | 检索、Reranker、生成及完整流水线延迟 |

层次化检索 100 题缓存命令：

```bash
python prepare_hierarchical_retrieval.py \
  --embedding-device cuda --limit 100 \
  --parent-route-k 50 --parent-candidate-k 50 --parent-k 30 \
  --child-route-k 50 --candidate-k 50 --top-k 8 \
  --reranker-device cuda --reranker-batch-size 8 \
  --output-dir results/ohr_hierarchical_retrieval_cache_parent30_k8
```

用固定缓存生成答案，再做逐事实忠实度评测：

```bash
python evaluate_generation.py \
  --backend local-awq --skip-no-rag --limit 100 --top-k 8 --max-tokens 64 \
  --retrieval-cache results/ohr_hierarchical_retrieval_cache_parent30_k8 \
  --output-dir results/ohr_generation_hierarchical_parent30_k8

python evaluate_faithfulness.py \
  --predictions results/ohr_generation_hierarchical_parent30_k8/predictions.jsonl \
  --backend vllm --model qwen3-1.7b-awq --batch-size 16 --workers 16
```

Faithfulness 使用逐事实的本地 LLM-as-a-judge（以大语言模型作为评审）口径。它比只看
答案字面 F1 更接近“有没有依据资料胡乱回答”，但同一个 Qwen 同时作为回答模型和评审模型会有
自评偏差，所以正式报告需要固定 Prompt，并随机人工复核一部分样本。

### 本次离线构建结果

本次 OHR-Bench 离线构建已完整跑完，不再是 partial checkpoint：

| 层级 | Chunk 数 | 预设问题数 | 平均问题数/Chunk | 空问题 Chunk | fallback 问题 |
|---|---:|---:|---:|---:|---:|
| Parent Summary（每页一个） | 8,259 | 24,261 | 2.94 | 38（0.46%） | 14（0.17%） |
| Child（240 token，overlap 40） | 35,007 | 66,786 | 1.91 | 648（1.85%） | 377（1.08%） |

Parent/Child 的文本索引、问题索引均为 1024 维 BGE-M3 向量；Parent 与 Child 各自还有
“文本 + 预设问题”的 BM25 索引。四个 FAISS 全局索引已实际迁移到 `cuda:0` 并完成搜索验证，
不是只在 GPU 上做 Embedding。

### Parent 范围大小消融

固定 100 题、Child Top-8、每层 Text Dense + Question Dense + BM25 + RRF +
`BAAI/bge-reranker-v2-m3`，只改变进入 Child 层的 Parent 数：

| Parent K | Parent Hit@K | 最终 Hit@8 | Precision@8 | Recall@8 | MRR@8 | nDCG@8 | 检索延迟/问 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 74.00% | 71.00% | 11.38% | 67.00% | 62.70% | 61.04% | 1021.5 ms |
| 20 | 88.00% | 82.00% | 13.13% | 80.00% | 65.45% | 66.65% | 950.8 ms |
| 30 | 90.00% | 84.00% | 13.38% | 81.50% | 66.07% | 67.41% | 1181.0 ms |

Parent K=8 明显形成硬路由召回瓶颈；K=30 恢复了基线 Hit@8，并取得本轮最高 MRR/nDCG，
所以端到端实验选 K=30。K=20 延迟稍低，但 Hit/Recall 仍低于 K=30。单次短测的延迟会受
GPU 温度、缓存和调度影响，因此 K=20/30 的延迟差不能被解释成严格的线性规律。

### 历史组合实验：基线与层次化方案

本小节是较早的“所有层次化模块同时开启”组合实验，保留用于追踪历史，不再作为严格消融结论。
由于它同时改变 Parent 路由、预设问题、BM25 和 RRF，无法把变化归因到单个模块。独立控制变量、
统一 vLLM 生成后端后的结果见第 11 节。

实验条件：同一批 OHR-Bench 前 100 题、同一 Qwen3-1.7B W4A16 AWQ、Top-8、候选 50、
最大生成 64 token。基线为 `BGE-M3 Dense → Reranker → LLM`；新方案为两层
`文本 FAISS + 问题 FAISS + 文本/问题 BM25 → RRF → Reranker → LLM`。

Hit、Precision、Recall、MRR、nDCG 属于**检索阶段**，不是生成指标：

| 检索指标 | 基线 | 两层 + 问题 + BM25/RRF | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| Candidate Hit@50 | 92.00% | 90.00% | -2.00 个百分点 | -2.17% |
| Hit@8 | 84.00% | 84.00% | 0.00 个百分点 | 0.00% |
| Precision@8 | 13.13% | 13.38% | +0.25 个百分点 | +1.90% |
| Recall@8 | 81.00% | 81.50% | +0.50 个百分点 | +0.62% |
| MRR@8 | 64.44% | 66.07% | +1.63 个百分点 | +2.52% |
| nDCG@8 | 65.55% | 67.41% | +1.86 个百分点 | +2.84% |

生成与忠诚度结果：

| 生成/事实性指标 | 基线 | 两层 + 问题 + BM25/RRF | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| EM | 14.00% | 15.00% | +1.00 个百分点 | +7.14% |
| Answer Precision | 54.00% | 53.21% | -0.79 个百分点 | -1.46% |
| Answer Recall | 40.35% | 36.39% | -3.96 个百分点 | -9.81% |
| Answer F1 | 40.63% | 36.82% | -3.81 个百分点 | -9.37% |
| Faithfulness | 79.17% | 75.50% | -3.67 个百分点 | -4.63% |
| Hallucination Rate | 20.83% | 24.50% | +3.67 个百分点 | +17.60% |
| Answer Relevance（Judge） | 77.00% | 72.50% | -4.50 个百分点 | -5.84% |
| Answer Correctness（Judge） | 68.50% | 68.00% | -0.50 个百分点 | -0.73% |

这里的 Answer Precision/Recall/F1 是归一化字符重叠；Faithfulness 是逐事实证据支持率。
Judge 指标由同一个本地 Qwen3-1.7B 自动评分，100 条均成功解析，但仍可能存在自评偏差，
不能替代人工抽检或独立更强 Judge。

综合质量和延迟：

| 综合/性能指标 | 基线 | 两层 + 问题 + BM25/RRF | 变化 |
|---|---:|---:|---:|
| 检索命中且 Answer F1 ≥ 0.5 | 32.00% | 31.00% | -1.00 个百分点 |
| F1–Faithfulness H-mean | 53.70% | 49.50% | -4.20 个百分点（-7.82%） |
| 检索平均延迟 | 404.7 ms/问 | 1181.0 ms/问 | +776.3 ms（+191.79%） |
| 生成平均延迟 | 712.5 ms/问 | 947.3 ms/问 | +234.9 ms（+32.97%） |
| 端到端平均延迟 | 1117.2 ms/问 | 2128.3 ms/问 | +1011.1 ms（+90.51%） |
| 新方案端到端估算 P50 / P95 | — | 2080.2 / 2431.8 ms | — |

`F1–Faithfulness H-mean` 是本项目为了避免“答案相似但无证据”或“有证据但答非所问”而定义的
联合诊断值，不是通行的单一行业标准。正式报告不应只报一个综合分，而应同时报告检索、生成、
事实性和延迟，并比较质量—延迟 Pareto 边界。

### 本轮结论与可写入报告的表述

本轮优化使排序指标小幅改善：MRR@8 相对提升 2.52%，nDCG@8 相对提升 2.84%，Recall@8
增加 0.50 个百分点；但 Answer F1 相对下降 9.37%，Faithfulness 下降 3.67 个百分点，端到端
延迟增加 1011.1 ms/问。因此，**当前实现尚不能作为“最终效果提升”的结论，应视为完成了可运行
的层次化检索原型并发现了负收益配置**。

可以在阶段报告中写：

> 完成实验室内网知识查询的本地 RAG 基础框架，并实现按页 Parent Summary 到 240-token
> Child 的两层检索、父子 Chunk 离线预设问题、文本/问题双 FAISS、文本与问题 BM25、RRF
> 融合及 Cross-Encoder 重排。在 OHR-Bench 100 题消融中，MRR@8 与 nDCG@8 分别相对提升
> 2.52% 和 2.84%；但 Qwen3-1.7B 的 Answer F1 下降 9.37%，端到端平均延迟增加
> 1011.1 ms，说明检索排序收益尚未传导到最终回答质量，下一步需减少父层重排开销并优化
> Parent 路由与上下文组织后再决定是否替换基线。

原始结果位于：

- `results/ohr_generation_rag_reranker_k8/summary.json` 与 `quality_summary.json`；
- `results/ohr_hierarchical_retrieval_cache_parent30_k8/retrieval_meta.json`；
- `results/ohr_generation_hierarchical_parent30_k8/summary.json` 与 `quality_summary.json`。

## 9. BGE-M3 三种检索表示与 BM25 小样本消融

BGE-M3 原生提供三种表示：

- Dense Embedding（稠密嵌入）：每个 Chunk 一个 1024 维向量，当前 FAISS 主索引使用这一种；
- Sparse/Lexical Weight（稀疏词项权重）：由 BGE-M3 学习的词项匹配权重，不是 BM25；
- ColBERT Multi-Vector（ColBERT 多向量）：每个 token 一个向量，查询 token 与文档 token 做
  MaxSim 晚交互。

因此，原基础框架并没有使用完整三模态，只使用了 Dense；BM25 是独立的传统关键词路线。

为避免给 122,518 个 Chunk 全量重建 Sparse 和 ColBERT 索引，先进行了固定小语料实验：随机
选择 30 个 OHR-Bench 问题，加入它们的全部标注证据 Chunk，再补充固定随机干扰项，共
1,200 个 Chunk、33 个正确证据页。三组都在同一语料上全库评分，最大长度 256，并使用等权
RRF 融合：

```bash
python -m pip install -r requirements_bge_m3_modes.txt

python evaluate_bge_m3_modes.py \
  --question-count 30 --corpus-size 1200 \
  --device cuda:0 --batch-size 16 --max-length 256 \
  --k 1 3 5 8 10 \
  --output results/bge_m3_modes_small/summary.json
```

### 检索质量

| 方案 | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense + Sparse + ColBERT | 86.67% | 96.67% | 96.67% | 21.33% | 96.67% | 92.22% | 94.16% |
| Dense + BM25 | 93.33% | 93.33% | 100.00% | **22.00%** | **100.00%** | 94.83% | 96.06% |
| Dense + ColBERT | **96.67%** | **100.00%** | **100.00%** | **22.00%** | **100.00%** | **98.33%** | **98.77%** |

到 Top-10 时三组的 Hit 和 Recall 都达到 100%，区别主要在正确证据是否排在最前面。
`Dense + ColBERT` 的排序最好；与 `Dense + BM25` 相比，MRR 高 3.50 个百分点、nDCG 高
2.71 个百分点。等权加入 Sparse 后，MRR/nDCG 反而下降，说明当前小样本中 Sparse 路线给
融合结果带来了噪声；这不表示 Sparse 永远无效，后续可以单独调低其 RRF 权重或改用验证集调权。

### 延迟与离线成本

| 方案 | 在线吞吐等效延迟 |
|---|---:|
| Dense + Sparse + ColBERT | 227.2 ms/问 |
| Dense + BM25 | **4.2 ms/问** |
| Dense + ColBERT | 225.3 ms/问 |

在 1,200 Chunk 小语料上，ColBERT 全库晚交互本身约 222.2 ms/问，而 BM25 打分约
1.1 ms/问。BGE 三模态文档编码用时 5.1 秒，BM25 建库用时 0.5 秒；二者均属于离线成本。
这里的在线延迟是批量总耗时除以问题数，是吞吐等效值，不是并发服务的 P95。

### 是否需要保留 BM25

当前结论是**保留 BM25**：

- 它在这批样本上取得 100% Hit@5，MRR@10=94.83%、nDCG@10=96.06%；
- 相比全库 ColBERT，只损失 3.50 个百分点 MRR 和 2.71 个百分点 nDCG，但在线耗时从
  225.3 ms/问降到 4.2 ms/问，约快 53 倍；
- BM25 对编号、日期、金额、专有名词和 OCR 文档中的精确词匹配仍有价值；
- BGE-M3 Sparse 与 BM25 都是词项路线，但训练目标和评分公式不同，不能认为启用 Sparse 后
  BM25 必然冗余。

仅根据这项小样本，可以把 `Dense + BM25 → RRF` 保留为低延迟候选；若追求更高排序质量，可将
ColBERT 用于少量候选（例如 Top-50）重排，而不是在 122,518 个 Chunk 上逐个做全库晚交互。
第 11 节的完整 100 题复验表明 BM25 收益没有泛化，因此当前生产默认仍为纯 Dense，不应把本节
30 题、1,200 Chunk 的结论直接外推到全库。

原始结果见 `results/bge_m3_modes_small/summary.json`。

## 10. 综合权衡后的完整工作流

### 10.1 最终推荐架构

结合第 11 节补齐的同后端严格消融，当前默认质量版回到最小 Dense 基线。BM25、层级检索、
预设问题和查询变换均保留为可切换实验模块，而不是默认全部打开：

```text
PDF / Word / Markdown / TXT / OHR-Bench
                    ↓
         documents.jsonl（原始页面/文档）
                    ↓
       段落优先 Chunk（500 字符，overlap 80）
                    ↓
BGE-M3 Dense → FAISS Top-50
                    ↓
 BAAI/bge-reranker-v2-m3 Cross-Encoder
                    ↓
                 Top-8
                    ↓
       带来源信息的 Context / Prompt
                    ↓
       Qwen3-1.7B AWQ（以后可换 32B）
                    ↓
        答案 + 文档名/章节/页码引用
```

两个实际部署档位：

| 档位 | 检索链路 | 适用场景 |
|---|---|---|
| 低延迟版 | Dense Top-50 → Top-8 | CPU/GPU 资源有限、并发查询 |
| 质量版（推荐） | Dense Top-50 → Reranker → Top-8 | 离线评测、低并发高质量问答 |
| Hybrid 实验版 | Dense + BM25 → RRF 候选 50 → Reranker → Top-8 | 精确词匹配占主导的自有数据复验 |

暂不进入默认主线的模块：

- BGE-M3 Sparse：小样本等权融合后排序下降，需先在验证集上调权；
- ColBERT 全库搜索：排序最好，但 1,200 Chunk 已需约 222.2 ms/问做晚交互；
- Parent Summary → Child：检索约 1181.0 ms/问，且 Answer F1 相对基线下降 9.37%；
- LLM 预设问题索引：已经实现并保留，当前只作为层次化实验的一部分，不作为生产必选项；
- BM25：小语料表现好，但全库 100 题严格消融未超过纯 Dense 基线，故不再默认启用。

### 10.2 为什么选择候选 50 和 Top-8

下表是较早的筛参记录；最终同后端复验见第 11 节。它仍支持“候选 50 优于候选 100”的选择，
但不再用于宣称 Hybrid 一定优于纯 Dense。

同一批 OHR-Bench 前 100 题的全库实验：

| 方案 | Hit@8 | Recall@8 | MRR@8 | nDCG@8 | 检索延迟 |
|---|---:|---:|---:|---:|---:|
| Dense + BM25 + RRF | 84.00% | 80.00% | 58.47% | 61.69% | 20.7 ms/问 |
| Dense → Reranker，候选 50 | 84.00% | 81.00% | 64.44% | 65.55% | 404.7 ms/问 |
| Dense + BM25 + RRF → Reranker，候选 50 | **87.00%** | **85.00%** | **64.58%** | **67.74%** | 433.0 ms/问 |
| Dense + BM25 + RRF → Reranker，候选 100 | 87.00% | 84.50% | 64.08% | 67.10% | 811.2 ms/问 |
| Parent→Child 多路检索，Parent 30 | 84.00% | 81.50% | **66.07%** | 67.41% | 1181.0 ms/问 |

候选 100 相比候选 50 没有质量收益，延迟却增加约 378.2 ms/问；因此选择候选 50。Top-8
相对 Top-5 能增加证据召回，同时尚未像 Top-10 那样继续扩大生成上下文，适合作为当前折中点。
层次化方案的 MRR 略高，但端到端质量和延迟均不占优，所以不作为默认链路。

### 10.3 第一步：安装基础依赖

```bash
cd /home/lyc/workspace/rag
/home/lyc/workspace/vllm/.venv/bin/python -m pip install -r requirements.txt
```

`requirements_bge_m3_modes.txt` 只用于 Sparse/ColBERT 消融，建议装在独立虚拟环境，不是基础
RAG 的必需依赖。

### 10.4 第二步：准备数据

使用 OHR-Bench：

```bash
python prepare_ohr_bench.py
```

使用自有 PDF、Word、Markdown、TXT：

```bash
python ingest_documents.py /path/to/documents \
  --output data/custom/documents.jsonl
```

统一输出 `documents.jsonl`，每行至少包含 `id`、`text`，推荐同时保存 `title`、`domain`、
`page`、`section_path` 和 `source_path`，以便最终回答显示可追溯引用。

### 10.5 第三步：建立 Dense FAISS 与 BM25 索引

OHR-Bench 默认索引：

```bash
python offline_build.py \
  --documents data/ohr_bench/documents.jsonl \
  --output-dir storage/ohr_bench_bge_m3 \
  --device cuda --batch-size 48 \
  --chunk-size 500 --overlap 80 \
  --embedding-max-length 1024

python build_bm25.py \
  --index-dir storage/ohr_bench_bge_m3 \
  --overwrite
```

离线输出及对应关系：

```text
storage/ohr_bench_bge_m3/
├── chunks.jsonl       # 第 i 行是第 i 个 Chunk 的正文与元数据
├── index.faiss        # 第 i 个向量与 chunks.jsonl 第 i 行严格对应
├── index_meta.json    # 模型、维度、切块参数
└── bm25/              # 与同一份 chunks.jsonl 对应的关键词索引
```

`--device cuda` 会让 BGE 编码和无过滤 FAISS Flat 搜索使用 GPU；磁盘上的 `index.faiss` 仍是
可移植的 CPU 格式，加载时再迁移为 `GpuIndexFlat`。

### 10.6 第四步：先做检索消融

先测 Dense 基线：

```bash
python evaluate_retrieval.py \
  --embedding-device cuda \
  --k 1 3 5 8 10 --limit 100 \
  --output-dir results/ohr_retrieval_dense_100
```

再测 Hybrid 候选；是否采用以第 11 节所示的全链路结果为准：

```bash
python evaluate_retrieval.py \
  --embedding-device cpu \
  --hybrid --dense-k 50 --bm25-k 50 --rrf-k 60 \
  --reranker --reranker-device cuda \
  --candidate-k 50 --reranker-batch-size 8 --reranker-max-length 512 \
  --k 1 3 5 8 10 --limit 100 \
  --output-dir results/ohr_retrieval_hybrid_reranker_k8
```

这里将 Embedding 放在 CPU、Reranker 放在 GPU，是为了适配单张 8 GiB GPU。只测检索且显存
足够时可把 `--embedding-device` 改成 `cuda`，此时 Query Embedding 和全局 FAISS 都走 GPU。
判断是否进入生成实验，至少同时看 Hit/Recall、MRR/nDCG 和检索延迟；不要只看余弦分数。

### 10.7 第五步：端到端生成评测

使用自己的 `AutoQuantForCausalLM.from_quantized` 加载 Qwen3-1.7B AWQ；默认质量版只启用
Dense + Reranker：

```bash
python evaluate_generation.py \
  --backend local-awq \
  --embedding-device cpu \
  --reranker --reranker-device cuda \
  --candidate-k 50 --reranker-batch-size 8 --reranker-max-length 512 \
  --top-k 8 --limit 100 --max-tokens 64 \
  --output-dir results/ohr_generation_dense_reranker_k8
```

`evaluate_generation.py` 会先完成全部检索与重排并释放 Reranker，再加载本地 AWQ，避免两个模型
同时占用 8 GiB GPU。第一次完整对照不要加 `--skip-no-rag`，这样能同时得到无 RAG 与有 RAG
的 EM/F1；以后只比较不同检索策略时可加该参数节省生成时间。

这一阶段必须报告：

- 检索：Hit@8、Precision@8、Recall@8、MRR@8、nDCG@8；
- 生成：EM、Answer Precision/Recall/F1；
- 联合：检索命中且 Answer F1 ≥ 0.5 的比例；
- 性能：Dense、BM25、RRF、Reranker、生成和端到端 mean/P50/P95 延迟。

对应的严格端到端结果已在第 11 节补齐。仍不能用检索 nDCG 的提升代替 Answer F1 的提升结论，
两类指标必须分别报告。

### 10.8 第六步：评测忠诚度和幻觉

先启动本地 vLLM Judge，再运行：

```bash
python evaluate_faithfulness.py \
  --predictions results/ohr_generation_dense_reranker_k8/predictions.jsonl \
  --backend vllm --model qwen3-1.7b-awq \
  --batch-size 16 --workers 16 --max-tokens 512 \
  --output-dir results/ohr_generation_dense_reranker_k8
```

读取 `quality_summary.json` 中的 Faithfulness、Hallucination Rate、Answer Relevance、Answer
Correctness 和 F1–Faithfulness H-mean。当前 Judge 与回答模型同为 Qwen3-1.7B，正式结论需要
抽样人工核查，未来可换独立的 Qwen3-32B Judge。

### 10.9 第七步：实际在线提问

单张 8 GiB GPU 同时运行 vLLM、BGE-M3 和 Reranker 容易显存不足。建议把生成模型放在 vLLM
服务中，Embedding 和 Reranker 先用 CPU；如果有第二张 GPU，再把 Reranker 单独放到 GPU：

```bash
python online_rag.py \
  --backend vllm --api-base http://127.0.0.1:8000/v1 \
  --embedding-device cpu \
  --reranker --reranker-device cpu \
  --candidate-k 50 --top-k 8 \
  --question "你的问题" --show-context
```

如果当前更重视延迟，可关闭 Reranker：

```bash
python online_rag.py \
  --backend vllm --embedding-device cpu \
  --candidate-k 50 --top-k 8 \
  --question "你的问题"
```

### 10.10 后续升级顺序

后续每次只引入一个变量，并使用相同问题 ID 比较：

1. 用独立 Qwen3-32B Judge 复核第 11 节的 Faithfulness 与查询变换结论；
2. 在 RRF 后的 Top-50 候选上加入 ColBERT，而不是进行全库 ColBERT；
3. 在验证集上调 Dense/BM25/ColBERT 融合权重，再用独立测试集确认；
4. 若换 Qwen3-32B，只替换生成端，保持检索结果固定，单独测模型收益；
5. 只有当层次化方案同时提升 Answer F1/Faithfulness 且延迟可接受时，才替换扁平 Hybrid 主线。

一个改动只有在固定测试集上同时给出“检索变化、答案变化、忠诚度变化、延迟变化”，才应被认定
为有效优化。

## 11. 严格分阶段消融与最终选型

### 11.1 基线、控制变量与实验顺序

最小 RAG 基线定义为：

```text
Query → BGE-M3 Dense Embedding → FAISS → Retrieval
      → BAAI/bge-reranker-v2-m3 → Qwen3-1.7B AWQ
```

不能把所有优化一次性打开后只比较两个总分。本轮按下列顺序推进；每一步只与其直接前驱比较，
若前一步被淘汰，后续模块回到当前最优路径继续消融：

| 编号 | 只改变的变量 | 目的 |
|---|---|---|
| B0 | 最小 Dense FAISS + Reranker 基线 | 建立可运行下限 |
| A1 | BGE-M3 Dense/Sparse/ColBERT 与 BM25 的检索组合 | 选择第一阶段召回表示 |
| E1 | 选中的平铺 Dense + BM25 + RRF + Reranker | 作为后续严格对照 |
| E2 | E1 改为 Parent Summary → Child，两层都只检索文本 | 单测层级路由 |
| E3 | 在 E2 的 Parent/Child 增加 LLM 预设问题索引 | 单测预设问题对层级方案的增量 |
| E4a | 在 E1 上加入 Query Rewrite（查询重写） | 单测重写 |
| E4b | 在 E1 上加入 Step-back Query（退一步/回退查询） | 单测抽象回退查询 |
| E4c | 在 E1 上加入 Query Decomposition（问题拆解） | 单测子问题检索 |
| E4d | 在 E1 上同时使用三种查询变体 | 测组合路径 |

所有主表都使用 OHR-Bench 前 100 条相同问题、候选 50、最终 Top-8、同一
`BAAI/bge-reranker-v2-m3`、同一 Qwen3-1.7B W4A16 AWQ、最大生成 64 token。生成阶段统一走
同一个 vLLM 服务；查询变换也由同一个模型以 temperature=0 生成。LLM Judge 800 条均解析成功。

### 11.2 A1：BGE-M3 表示层消融

这一阶段已经完成，详细语料构造、指标和命令见第 9 节。固定 30 题、1,200 Chunk 小语料的
结论是：

| 召回组合 | Hit@5 | MRR@10 | nDCG@10 | 延迟/问 | 决策 |
|---|---:|---:|---:|---:|---|
| Dense + Sparse + ColBERT | 96.67% | 92.22% | 94.16% | 227.2 ms | 等权 Sparse 引入噪声，暂不选 |
| Dense + BM25 | 100.00% | 94.83% | 96.06% | **4.2 ms** | **选作可扩展主召回** |
| Dense + ColBERT | 100.00% | **98.33%** | **98.77%** | 225.3 ms | 质量最好，保留作候选重排实验 |

小样本阶段因此把 `Dense + BM25 → RRF → Reranker` 选为 E1 候选；但小样本选择必须接受全库
复验。完整 100 题结果显示 E1 未超过 B0，所以最终默认路径仍是纯 Dense。完整 122,518 Chunk
上也不做全库 ColBERT 晚交互。

### 11.3 B0/E1/E2/E3：基线、BM25、层级与预设问题

E2 明确关闭生成问题路由，只使用 Parent Summary 文本与 Child 文本；E3 才开启父子层预设问题。
这样可以把“层级路由”和“预设问题”分开归因。

| 路径 | Hit@8 | Precision@8 | Recall@8 | MRR@8 | nDCG@8 | EM | Answer F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Dense FAISS+Reranker | **86.00%** | **13.38%** | **83.00%** | 64.69% | 66.18% | 14.00% | **40.47%** |
| E1 平铺 Dense+BM25+RRF+Reranker | 85.00% | 13.25% | 81.50% | 63.85% | 65.23% | 13.00% | 39.61% |
| E2 Parent Summary→Child，仅文本 | 79.00% | 12.75% | 76.50% | 62.67% | 63.70% | 14.00% | 33.97% |
| E3 E2 + 父子预设问题 | 84.00% | **13.38%** | **81.50%** | **66.07%** | **67.41%** | **15.00%** | 36.82% |

全库复验中 E1 相对 B0 的 Hit、Recall、nDCG 和 F1 分别下降 1.00、1.50、0.95 和 0.86 个
百分点，说明小语料上的 BM25 收益没有泛化到该 100 题集合。BM25 因此降为可选路由。
E2 相对 E1 的 Hit、Recall 和 F1 分别下降 6.00、5.00 和 5.65 个百分点，说明 Parent 硬路由
会漏掉正确页面。E3 相对 E2 恢复了 5.00 个百分点 Hit、5.00 个百分点 Recall 和 2.85 个百分点
F1，证明父子预设问题对层级检索本身有效；但 E3 的 F1 仍比 E1 低 2.80 个百分点。

事实性与延迟：

| 路径 | Faithfulness | 幻觉率 | Judge 相关性 | Judge 正确性 | F1–Faithfulness H-mean | 检索 | 生成 | 端到端 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | **79.67%** | **20.33%** | **77.00%** | 68.00% | **53.68%** | 370.8 ms | 232.4 ms | 603.3 ms |
| E1 | 76.50% | 23.50% | 75.50% | 67.00% | 52.20% | **338.2 ms** | **203.5 ms** | **541.7 ms** |
| E2 | 75.50% | 24.50% | 72.50% | 67.50% | 46.86% | 776.8 ms | 296.2 ms | 1073.1 ms |
| E3 | 75.50% | 24.50% | 72.50% | **69.00%** | 49.50% | 1181.0 ms | 293.4 ms | 1474.4 ms |

E1 的本次延迟低于 B0，主要来自候选文本长度和运行波动，而不是 BM25 在结构上减少了步骤；
因此不能把 61.5 ms 差值解释为稳定加速。E3 的检索排序优于 E1，但收益没有传导为更高答案
F1 或忠实度，并使端到端延迟相对 B0 增加 871.1 ms/问。因此 Parent→Child 与预设问题保留
为实验功能，不进入默认生产路径。

### 11.4 E4：查询重写、Step-back 回退查询与拆解

离线固定查询变体保存在 `data/ohr_bench/query_variants.jsonl`。每个检索实验始终保留原始 Query，
再添加当前被测变体；各 Query 分别走 Dense 和 BM25，使用等权 RRF 合并，最后仍以原始 Query
执行 Reranker。这样不会用改写后的措辞替代用户原意。生成 100 组变体无 fallback，吞吐等效
耗时为 122.4 ms/query，该耗时已计入 E4 检索与端到端时间。

| 路径 | 平均 Query 数 | Hit@8 | Recall@8 | MRR@8 | nDCG@8 | EM | Answer F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 原始 Query | 1.00 | **85.00%** | **81.50%** | 63.85% | 65.23% | **13.00%** | 39.61% |
| E4a 原始 + Rewrite | 1.31 | **85.00%** | **81.50%** | 63.70% | 65.12% | 12.00% | 39.00% |
| E4b 原始 + Step-back | 1.80 | 83.00% | 79.50% | 63.10% | 64.16% | 12.00% | **40.53%** |
| E4c 原始 + Decomposition | 3.03 | 84.00% | 81.00% | **64.47%** | **65.49%** | 12.00% | 38.86% |
| E4d 原始 + 全部变体 | 3.57 | 84.00% | 81.00% | **64.47%** | **65.49%** | **13.00%** | 39.55% |

| 路径 | Faithfulness | 幻觉率 | Judge 相关性 | Judge 正确性 | H-mean | 检索 | 端到端 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 76.50% | 23.50% | 75.50% | 67.00% | 52.20% | **338.2 ms** | **541.7 ms** |
| E4a Rewrite | 75.50% | 24.50% | 74.00% | 64.50% | 51.44% | 467.5 ms | 673.6 ms |
| E4b Step-back | 76.50% | 23.50% | 75.00% | 67.50% | 52.99% | 469.7 ms | 676.6 ms |
| E4c Decomposition | 78.00% | 22.00% | 77.50% | 69.50% | 51.87% | 476.4 ms | 678.1 ms |
| E4d 全部变体 | **82.00%** | **18.00%** | **81.50%** | **72.50%** | **53.36%** | 478.9 ms | 678.1 ms |

E4b 的 Answer F1 比 E1 高 0.91 个百分点，但检索 Hit/Recall/nDCG 均下降，且端到端增加
134.9 ms/问；100 题上不足以证明稳定提升。E4d 的 F1 与 E1 基本持平（-0.07 个百分点），
但 Judge Faithfulness 增加 5.50 个百分点、幻觉率下降 5.50 个百分点、H-mean 增加 1.16 个
百分点，代价是端到端增加 136.4 ms/问。由于回答模型和 Judge 都是同一个 1.7B 模型，这个
事实性收益必须用独立 32B Judge 或人工抽检复验后才能作为正式结论。

与最终 B0 而不是 E1 比较时，E4d 的 F1 低 0.92 个百分点，Faithfulness 高 2.33 个百分点，
H-mean 低 0.32 个百分点，端到端增加 74.8 ms/问；因此它也未整体支配 B0。

### 11.5 最终选择

当前默认路径选择 B0：

```text
Query → BGE-M3 Dense FAISS Top-50
      → BGE Reranker → Top-8 → Qwen3-1.7B/32B
```

理由是它在完整 100 题上取得本轮最高 Answer F1 和 H-mean，并且检索 Hit/Recall/nDCG 均高于
加入 BM25 的 E1。E1 可作为实测较低延迟的候选，但必须在多轮延迟基准中确认。对于“宁可慢
一些，也优先减少无依据陈述”的场景，可把 E4d 作为高忠实度候选路径，但需先由独立 Judge
复验。E2/E3 不进入默认路径；它们适合未来在超大知识库中重新测试，因为当前 8,259 页规模下
Parent 路由的降搜索空间收益不足以抵消两次 Reranker 和硬路由漏召回。

因此现阶段报告应写“完成并否决了若干负收益配置”，不能写成所有模块都带来正收益：

> 在固定 OHR-Bench 100 题上完成 BGE-M3 表示、层级检索、父子预设问题及查询变换的分阶段
> 消融。BGE-M3 Dense FAISS+Reranker 基线取得 Hit@8 86.00%、Recall@8 83.00%、Answer
> F1 40.47%、Faithfulness 79.67%，作为当前默认方案。Dense+BM25 在小语料表现较好，但全库
> 100 题的检索和生成指标均未超过纯 Dense 基线，故保留为可选路由。Parent→Child 加预设问题虽将层级方案的
> nDCG@8 提升到 67.41%，但 Answer F1 降至 36.82% 且端到端增至 1474.4 ms/问，未被选用。
> 全查询变体方案将 LLM Judge 忠实度由 76.50% 提至 82.00%、幻觉率由 23.50% 降至
> 18.00%，F1 基本持平，端到端增加 136.4 ms/问；该结果待独立 32B Judge 复验后再决定是否
> 用于高可靠场景。

### 11.6 复现实验

纯层级文本 E2 与带问题 E3：

```bash
# E2：关闭 Parent/Child generated-question 路由
/home/lyc/workspace/vllm/.venv/bin/python prepare_hierarchical_retrieval.py \
  --embedding-device cuda --limit 100 \
  --parent-route-k 50 --parent-candidate-k 50 --parent-k 30 \
  --child-route-k 50 --candidate-k 50 --top-k 8 --rrf-k 60 \
  --no-generated-question-routes \
  --reranker-device cuda --reranker-batch-size 8 --reranker-max-length 512 \
  --output-dir results/ohr_ablation_e2_summary_child_parent30_k8

# E3：默认开启 Parent/Child question FAISS 与文本+问题 BM25
/home/lyc/workspace/vllm/.venv/bin/python prepare_hierarchical_retrieval.py \
  --embedding-device cuda --limit 100 \
  --parent-route-k 50 --parent-candidate-k 50 --parent-k 30 \
  --child-route-k 50 --candidate-k 50 --top-k 8 --rrf-k 60 \
  --reranker-device cuda --reranker-batch-size 8 --reranker-max-length 512 \
  --output-dir results/ohr_hierarchical_retrieval_cache_parent30_k8
```

查询变体生成与 B0/E1/E4 检索：

```bash
python generate_query_variants.py \
  --backend vllm --model qwen3-1.7b-awq \
  --limit 100 --batch-size 16 --workers 16 --max-tokens 256 \
  --output data/ohr_bench/query_variants.jsonl

# B0：严格 Dense 基线
/home/lyc/workspace/vllm/.venv/bin/python prepare_query_variant_retrieval.py \
  --variant-mode original --retrieval-routes dense \
  --limit 100 --embedding-device cuda --dense-k 50 \
  --candidate-k 50 --top-k 8 --rrf-k 60 \
  --reranker-device cuda --reranker-batch-size 8 --reranker-max-length 512 \
  --output-dir results/ohr_ablation_b0_dense_k8

# E1：不做查询变换的 Dense+BM25 对照
/home/lyc/workspace/vllm/.venv/bin/python prepare_query_variant_retrieval.py \
  --variant-mode original --retrieval-routes dense-bm25 \
  --limit 100 --embedding-device cuda \
  --dense-k 50 --bm25-k 50 --candidate-k 50 --top-k 8 --rrf-k 60 \
  --reranker-device cuda --reranker-batch-size 8 --reranker-max-length 512 \
  --output-dir results/ohr_ablation_e0_original_k8

# E4：每个查询变体都与原始 Query 一同参与 RRF
for mode in rewrite step_back decompose all; do
  /home/lyc/workspace/vllm/.venv/bin/python prepare_query_variant_retrieval.py \
    --variant-mode "$mode" --retrieval-routes dense-bm25 \
    --limit 100 --embedding-device cuda \
    --dense-k 50 --bm25-k 50 --candidate-k 50 --top-k 8 --rrf-k 60 \
    --reranker-device cuda --reranker-batch-size 8 --reranker-max-length 512 \
    --output-dir "results/ohr_ablation_e4_${mode}_k8"
done
```

每个检索缓存再统一运行 `evaluate_generation.py --retrieval-cache ...`，最后对其
`predictions.jsonl` 运行 `evaluate_faithfulness.py`。原始结果分别位于
`results/ohr_ablation_b0_*`、`ohr_ablation_e0_*`、`ohr_ablation_e2_*`、`ohr_ablation_e3_*` 与
`ohr_ablation_e4_*`，表格数值均直接来自这些 JSON 文件。
