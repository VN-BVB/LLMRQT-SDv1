# RAG（Retrieval-Augmented Generation，检索增强生成）探索与实验记录

> 更新日期：2026-09-18  
> 项目目录：`/home/lyc/workspace/rag`  
> 说明：本文不是聊天逐字稿，而是对本次对话中所有问题、排查过程、代码演进、实验结果和结论的压缩记录，方便后续复核与复现实验。
> 术语规则：专业名词第一次出现时给出英文全称、中文名称或用途说明；后文直接使用简称，避免重复。

## 1. 最初目标

本项目希望建立一套可以离线建库、在线检索、调用本地量化 Qwen（通义千问系列大语言模型）生成答案，并能量化比较优化效果的标准 RAG 框架。最初提出的需求包括：

1. 支持 PDF（Portable Document Format，便携式文档格式）、Word、Markdown（轻量级标记语言）、TXT（Plain Text，纯文本）文档读取；
2. 初期采用简单段落分块，后续支持章节和父子块；
3. 选择中文/多语言 Embedding（文本向量嵌入模型），并兼容 Qwen3-1.7B（约 17 亿参数）和后续 32B（约 320 亿参数）生成模型；
4. 离线保存原始文档 JSONL（JSON Lines，以行为单位保存 JSON；JSON 全称 JavaScript Object Notation，中文为 JavaScript 对象表示法）、Chunk（文本块）JSONL 和 FAISS（Facebook AI Similarity Search，向量相似度检索库）索引；
5. 引入公开数据集测试 RAG；
6. 使用自定义 `AutoQuantForCausalLM.from_quantized` 加载 Qwen3-1.7B AWQ（Activation-aware Weight Quantization，激活感知权重量化），跑通完整 RAG；
7. 建立检索、生成、事实性和延迟指标，证明 RAG 优化是否真正有效。

## 2. 当前项目结构与职责

核心文件：

| 文件 | 职责 |
|---|---|
| `ingest_documents.py` | 将 PDF、Word、Markdown、TXT 等自有资料转换为文档 JSONL |
| `prepare_ohr_bench.py` | 将 OHR-Bench（OCR Hinders RAG Benchmark，OCR 对 RAG 级联影响评测基准）的 Parquet（Apache 列式存储格式）转换为项目使用的 documents/questions JSONL |
| `offline_build.py` | 文档分块、Embedding、构建并保存 FAISS |
| `rag_core.py` | 分块、BGE（BAAI General Embedding，智源通用文本嵌入；BAAI 全称 Beijing Academy of Artificial Intelligence，中文为北京智源人工智能研究院）模型、FAISS 读取和检索等核心逻辑 |
| `reranker.py` | Cross-Encoder（交叉编码器）候选重排 |
| `online_rag.py` | 单问题/交互式 Dense Retrieval（稠密向量检索）或 Dense+Reranker（重排序器）检索及 LLM（Large Language Model，大语言模型）回答 |
| `generation_backends.py` | vLLM（高吞吐量大模型推理引擎）HTTP（Hypertext Transfer Protocol，超文本传输协议）与本地 AWQ 两种生成后端 |
| `evaluate_retrieval.py` | 检索指标评测，不调用 LLM |
| `evaluate_generation.py` | 端到端回答、EM（Exact Match，精确匹配）、F1（精确率与召回率的调和平均）、nDCG（Normalized Discounted Cumulative Gain，归一化折损累计增益）和延迟评测 |
| `标准RAG框架.md` | 框架原理、数据格式和使用说明 |
| `进阶.md` | 后续检索优化方向记录 |

主要数据目录：

```text
data/ohr_bench/
├── documents.jsonl
├── questions.jsonl
└── dataset_meta.json

storage/ohr_bench_bge_m3/
├── chunks.jsonl
├── index.faiss
└── index_meta.json
```

## 3. 数据集选择历程

### 3.1 CMRC2018

最初使用 CMRC2018（Chinese Machine Reading Comprehension 2018，2018 中文机器阅读理解数据集）验证最小 RAG 流程。它适合中文阅读理解冒烟测试，但文档规模、文档类型和章节结构较简单，不足以测试论文、说明书、财报和 OCR（Optical Character Recognition，光学字符识别）等混合场景。

### 3.2 OHR-Bench

随后选择 OHR-Bench 作为主要公开测试集，原因是它包含结构化 PDF 页面、表格、章节、财报等复杂内容，并带有问题、答案和证据页面标注，适合评估真实 RAG。

当前转换结果约为：

```text
文档/页面数：8259
Chunk 数：122518
```

OHR 问题记录中保留的关键字段包括：

```json
{
  "id": "问题 ID（Identifier，标识符）",
  "question": "问题",
  "answers": ["标准答案"],
  "gold_doc_ids": ["证据页面 source_id"],
  "evidence_context": "人工证据文本",
  "doc_name": "文档名",
  "domain": "领域",
  "evidence_pages": [25]
}
```

需要特别注意：OHR 当前主要提供页面级 `gold_doc_ids`，没有直接提供构建索引后生成的 gold chunk ID。因此页面命中不一定代表命中了包含答案的准确 chunk。

## 4. 分块与离线索引

当前索引参数来自 `storage/ohr_bench_bge_m3/index_meta.json`：

```text
chunk_size = 500 字符
overlap = 80 字符
chunk_strategy = heading-aware paragraph + oversized paragraph window
embedding_model = BAAI/bge-m3（BAAI 为 Beijing Academy of Artificial Intelligence，即北京智源人工智能研究院；该模型是 BGE-M3 多语言向量模型）
embedding_dimension = 1024
embedding_max_length = 1024
index_type = IndexFlatIP（基于 Inner Product，即内积的精确平面索引）
normalized = true
```

分块策略为：优先按标题和自然段切分，只对超长自然段使用带 overlap 的窗口。Chunk 会保存标题、领域、章节路径、页码、原文起止位置等元数据。

### 4.1 FAISS 与 Chunk 如何对应

`offline_build.py` 按顺序生成：

```python
chunks.append(chunk)
embedding_texts.append(text)
```

向量随后按相同顺序写入 FAISS。因此：

```text
FAISS vector id 0 ↔ chunks.jsonl 第 0 行
FAISS vector id 1 ↔ chunks.jsonl 第 1 行
...
```

检索返回的 FAISS 向量编号直接用于索引内存中的 `chunks` 列表。`index.faiss` 只保存向量和索引结构，不保存正文、标题、页码等内容；这些内容位于 `chunks.jsonl`。

## 5. Embedding 模型演进

早期尝试过较小的中文 BGE 模型，随后统一更换为：

```text
BAAI/bge-m3
```

选择原因：支持中英文和多语言，适合 OHR-Bench 混合资料，向量维度 1024，可作为后续优化的稳定 Dense 基线。

### 5.1 Qwen3-Embedding 输入讨论

讨论过 Qwen3-Embedding（通义千问第三代文本向量模型）官方模型的输入要求。其查询侧通常需要 instruction-aware（指令感知）输入，并采用对应模型规定的 pooling（池化）。不能仅把 `--embedding-model` 字符串替换为 Qwen3-Embedding 就假定行为正确，需要实现其官方输入模板、tokenization（分词）和 pooling。因此当前稳定基线继续使用 BGE-M3（M3 表示 Multi-Functionality、Multi-Linguality、Multi-Granularity，即多功能、多语言、多粒度）。

### 5.2 Hugging Face（开源模型与数据集托管平台）看似重复下载

加载 BGE-M3 时曾看到 `pytorch_model.bin` 与 `model.safetensors` 的下载/重建进度，看起来像下载两次。原因通常是模型仓库包含多种权重格式、缓存重建或不同加载路径触发文件解析，并不代表每次运行都会永久重复下载。模型完整进入 Hugging Face 缓存后，后续一般直接复用。

### 5.3 Embedding 进度条

针对 122518 个 chunk 编码时“加载权重后像卡住”的问题，Embedding 批处理增加了 tqdm（Python 命令行进度条库）进度显示，以区分模型加载、文本编码和索引写入阶段。

## 6. 标准 RAG 在线流程

当前标准流程：

```text
问题
  → BGE-M3 Query Embedding
  → FAISS Dense 检索
  → 可选 Cross-Encoder Reranker
  → 取最终 Top-K（按得分选择前 K 个）Chunk
  → 构造包含来源标签的 Prompt
  → Qwen3-1.7B AWQ / vLLM
  → 答案
```

生成后端支持：

1. `vllm`：调用 OpenAI 兼容 HTTP API（Application Programming Interface，应用程序编程接口）；
2. `local-awq`：通过项目中的自定义量化加载逻辑直接加载 Qwen3-1.7B W4A16（4 位权重、16 位激活）AWQ。

## 7. 来源标签与回答格式问题

早期 Prompt 使用 `[资料1]`、`[资料2]`，模型也会在回答中输出“根据资料1”。后来改为向模型提供可直接复制的稳定来源标签：

```text
[资料名：finance/JPMORGAN_2021Q1_10Q | 章节：13 -2624428 | 页数：25]
```

Prompt 要求模型只能选择以下两种输出之一：

```text
答案：<最短答案>
依据：<完整来源标签>
```

或者：

```text
根据提供的资料无法确定
```

这样做是为了解决以下问题：

- 模型只写“资料1”，无法对应真实文档；
- 先给出一个猜测答案，随后又写“无法确定”；
- 把不同年份、不同概念的数值相加；
- 引用信息缺少文档名、章节和页码。

自动 EM/F1 评测使用专门的简短回答 Prompt，不要求引用，以免引用文本降低字面匹配分数。

## 8. 评测指标演进

最初只有向量余弦相似度，但相似度不能说明最终回答是否正确或是否胡编。因此指标分为三层。

### 8.1 候选召回层

- Candidate Hit@N（候选集前 N 项是否至少命中一个正确页面）
- Candidate Recall@N（候选集前 N 项覆盖了多少正确页面）

用于回答：正确页面是否进入 Dense 候选集。如果未进入，Reranker 无法补救。

### 8.2 最终排序层

- Hit@K
- Recall@K
- MRR@K（Mean Reciprocal Rank@K，前 K 项平均倒数排名）
- nDCG@K（前 K 项归一化折损累计增益）

含义：

- Hit@K：前 K 个结果是否至少包含一个正确页面；
- Recall@K：所有 gold 页面中召回了多少；
- MRR@K：第一个正确页面是否靠前；
- nDCG@K：综合相关结果位置的排序质量。

### 8.3 端到端回答层

- Exact Match（EM，精确匹配）
- 字符级 F1
- RAG 与无 RAG 的差值
- 检索命中时/未命中时的回答质量
- 后续可加入 Faithfulness（答案忠实度）、Answer Relevance（答案相关性）、引用准确率和人工错误分类。

F1 同时考虑 Precision 与 Recall，比 EM 更能容忍回答与标准答案字面不完全一致。但生成式答案可能包含解释或等价表述，所以仍需抽查 `predictions.jsonl`。

### 8.4 延迟

`evaluate_generation.py` 当前记录：

- Dense 检索总耗时与 ms/query；
- Reranker 总耗时与 ms/query；
- 无 RAG 生成耗时（未跳过时）；
- RAG 生成耗时；
- Dense + Reranker + RAG 生成的完整流水线平均耗时。

模型首次下载与模型加载时间不计入流水线延迟。

## 9. Reranker 的加入与完善

Reranker 基线模型：

```text
BAAI/bge-reranker-v2-m3（BAAI 为 Beijing Academy of Artificial Intelligence，即北京智源人工智能研究院；该模型是第二代 M3 多语言交叉编码重排序模型）
```

位置：FAISS 候选召回之后、最终 Top-K 和 Prompt 构造之前。

```text
BGE-M3 + FAISS Top-50
  → bge-reranker-v2-m3
  → Top-5/8/10
  → Qwen3
```

完成的代码能力：

- `--reranker` 开关；
- `--reranker-model`；
- `--reranker-device cpu/cuda`；
- `--candidate-k`；
- `--reranker-batch-size`；
- `--reranker-max-length`；
- `(question, passage)` Cross-Encoder 打分；
- passage 包含标题、领域、章节和正文；
- 保存 `dense_rank`、`dense_score`、`rerank_score`；
- 批量重排 tqdm；
- 重排完成后主动卸载模型、清理 CUDA（Compute Unified Device Architecture，统一计算设备架构）cache（缓存）；
- Dense 与 Reranker 输出使用不同结果目录；
- 同次检索评测报告 Dense、Reranker 和指标差值；
- 保存 chunk 级排名明细，支持失败分析。

### 9.1 GPU（Graphics Processing Unit，图形处理器）使用策略

端到端 `local-awq` 推荐：

```text
Embedding：CPU（Central Processing Unit，中央处理器）
Reranker：CUDA
Reranker 完成后释放显存
Qwen AWQ：CUDA
```

这样避免 BGE-M3、Reranker 和 Qwen 同时占用有限显存。

只跑 `evaluate_retrieval.py` 时可以同时使用：

```text
--embedding-device cuda
--reranker-device cuda
```

但 BGE-M3 和 Reranker 会同时驻留 GPU，显存不足时应把 Embedding 改回 CPU，或把 Reranker batch 从 8 降到 4。

FAISS 当前使用 `faiss-cpu`，神经网络编码和重排可用 CUDA，但 FAISS 相似度搜索仍在 CPU。

## 10. 关键困难案例：842、174 和 317

问题：

```text
What was the total amount of nonaccrual loans retained as of March 31, 2021?
```

OHR 标准记录明确给出：

```text
标准答案：842
证据页面：ohr-doc-e72b45cb02e5b151-page-0024
证据文本：Nonaccrual loans retained ... 842 ... 689 ... 22%
```

准确证据 chunk：

```text
ohr-doc-e72b45cb02e5b151-page-0024-chunk-001
```

正文包含：

```text
Nonaccrual loans retained ... $842 ... $689 ... 22%
```

Dense Top-20 的关键排名：

```text
Rank 1   page-0024-chunk-004  score=0.679291
Rank 14  page-0024-chunk-001  score=0.643531  ← 准确答案证据
```

Dense Rank 1 的 chunk 内容是：

```text
Allowance for loan losses of $174 million and $317 million were held against
these nonaccrual loans at March 31, 2021 and 2020, respectively.
```

它描述的是两个年份的贷款损失准备，而不是非应计贷款总额。模型曾错误计算：

```text
174 + 317 = 491
```

真实 Reranker 测试中，准确 `842` chunk 从 Dense Rank 14 提升到 Reranker Rank 4，因此进入 Top-5，但没有进入 Top-3。错误的 174/317 chunk 仍可能排名很高，所以 Reranker 改善了排序但没有完全解决语义混淆。

该案例还暴露了页面级 gold 的局限：错误 chunk 和准确 chunk 都来自同一页，因此页面级 Hit@1 可能显示命中，但 LLM 实际并未得到准确证据。

## 11. `evaluate_generation.py` 当前行为

### 11.1 跳过无 RAG

新增：

```text
--skip-no-rag
```

启用后：

- 完全不调用无 RAG 生成；
- `no_rag_answer` 保存为 `null`；
- 不计算 `no_rag_em/no_rag_f1`；
- `rag_minus_no_rag` 为 `null`；
- 继续计算 RAG EM、F1、Hit、nDCG 和延迟。

这一开关用于只比较：

```text
RAG
vs.
RAG + Reranker
```

### 11.2 nDCG 口径

生成评测中的 `K` 严格对应实际送给 LLM 的前 K 个 chunk。由于 OHR 只有页面级标注，使用 `source_id` 判断相关性；同一页面重复出现时只将第一次计为相关，避免重复 chunk 人为抬高 nDCG。

`evaluate_retrieval.py` 当前是按排名去重页面后计算 K 个页面，而 `evaluate_generation.py` 是实际 K 个 chunk，因此两个程序的 Hit@K 可能有轻微差异。例如同一组实验出现过检索评测 Hit@5=0.82，而生成评测 Hit@5=0.80。

## 12. 已完成的 100 题端到端实验

### 12.1 普通 RAG，Top-5

```text
无 RAG EM              0.0100
无 RAG F1              0.2774
RAG EM                 0.1100
RAG F1                 0.3473
RAG - 无 RAG EM       +0.1000
RAG - 无 RAG F1       +0.0700
Dense Candidate Hit@5  0.6400
最终 Hit@5             0.6400
Dense nDCG@5           0.4862
最终 nDCG@5            0.4862
Dense 延迟              63.2 ms/query
RAG 生成延迟           771.7 ms/query
完整流水线延迟         834.8 ms/query
```

### 12.2 RAG + Reranker，Candidate-50 → Top-5

```text
无 RAG EM               0.0100
无 RAG F1               0.2774
RAG EM                  0.1300
RAG F1                  0.4124
RAG - 无 RAG EM        +0.1200
RAG - 无 RAG F1        +0.1350
Dense Candidate Hit@50  0.9200
最终 Hit@5              0.8000
Dense nDCG@5            0.4862
最终 nDCG@5             0.6338
Reranker nDCG 差值     +0.1476
Dense 延迟               76.7 ms/query
Reranker 延迟           798.0 ms/query
RAG 生成延迟            527.5 ms/query
完整流水线延迟         1402.2 ms/query
```

RAG + Reranker 相对普通 RAG 的直接变化：

```text
EM       0.1100 → 0.1300，绝对 +0.0200
F1       0.3473 → 0.4124，绝对 +0.0650，约相对 +18.7%
Hit@5    0.6400 → 0.8000，绝对 +0.1600
nDCG@5   0.4862 → 0.6338，绝对 +0.1476
总延迟   834.8  → 1402.2 ms/query，约增加 68%
```

因此 Reranker 对当前 100 题有明显质量收益，但延迟代价也明显。

### 12.3 命中和失败拆解

普通 RAG：

```text
Hit 64 题：EM=0.1563，F1=0.4240
Miss 36 题：EM=0.0278，F1=0.2110
```

RAG + Reranker：

```text
Hit 80 题：EM=0.1500，F1=0.4508
Miss 20 题：EM=0.0500，F1=0.2587
```

Reranker Top-5 中首个 gold 页面位置：

```text
Rank 1：54 题
Rank 2：13 题
Rank 3： 4 题
Rank 4： 5 题
Rank 5： 4 题
未命中：20 题
```

20 个 Top-5 miss 中：

```text
12 题：gold 已进入 Dense Top-50，但未被重排进 Top-5
 8 题：gold 未进入 Dense Top-50，增加最终 Top-K 无法解决
```

100 题逐题比较 F1：

```text
Reranker 后提升：36 题
保持不变：        42 题
下降：            22 题
平均 F1 差值：   +0.0650
```

这说明不能假定 Reranker 对每道题都更好，必须报告整体指标并分析下降案例。

## 13. Reranker K 值检索实验

100 题、Dense Candidate Top-50 的重排结果：

| K | Hit@K | Recall@K | MRR@K | nDCG@K |
|---:|---:|---:|---:|---:|
| 1 | 0.5400 | 0.4100 | 0.5400 | 0.5400 |
| 3 | 0.7200 | 0.6750 | 0.6217 | 0.6112 |
| 5 | 0.8200 | 0.7900 | 0.6447 | 0.6602 |
| 8 | 0.8600 | 0.8350 | 0.6500 | 0.6762 |
| 10 | 0.8900 | 0.8700 | 0.6532 | 0.6869 |

边际收益：

```text
K=5 → 8： Hit +0.0400，Recall +0.0450，nDCG +0.0160
K=8 →10： Hit +0.0300，Recall +0.0350，nDCG +0.0107
K=5 →10： Hit +0.0700，Recall +0.0800，nDCG +0.0267
```

当前判断：

- 只看检索覆盖率，K=10 最好，Hit@10 已接近 Candidate Hit@50 的上限；
- 考虑 Qwen3-1.7B 的抗干扰能力和输入长度，K=8 是下一步最合理的端到端实验；
- 如果 K=8 只提升 Hit/nDCG，却使回答 F1 下降，应继续使用 K=5；
- 如果 K=8 的 F1 明显提升，再测试 K=10。

## 14. 当前推荐命令

### 14.1 普通 RAG Top-5，不跑无 RAG

```bash
cd ~/workspace/rag

python evaluate_generation.py \
  --backend local-awq \
  --embedding-device cpu \
  --top-k 5 \
  --skip-no-rag \
  --limit 100 \
  --max-tokens 64 \
  --output-dir results/ohr_generation_rag_k5_skip_no_rag
```

### 14.2 RAG + Reranker Top-5，不跑无 RAG

```bash
python evaluate_generation.py \
  --backend local-awq \
  --embedding-device cpu \
  --top-k 5 \
  --reranker \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --reranker-device cuda \
  --candidate-k 50 \
  --reranker-batch-size 8 \
  --reranker-max-length 512 \
  --skip-no-rag \
  --limit 100 \
  --max-tokens 64 \
  --output-dir results/ohr_generation_rag_reranker_k5_skip_no_rag
```

### 14.3 下一步：RAG + Reranker Top-8

```bash
python evaluate_generation.py \
  --backend local-awq \
  --embedding-device cpu \
  --top-k 8 \
  --reranker \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --reranker-device cuda \
  --candidate-k 50 \
  --reranker-batch-size 8 \
  --reranker-max-length 512 \
  --skip-no-rag \
  --limit 100 \
  --max-tokens 64 \
  --output-dir results/ohr_generation_rag_reranker_k8
```

### 14.4 运行全部问题

```text
--limit 0
```

含义：不截断 `questions.jsonl`，评测其中全部问题。

## 15. 如何判定一次优化有效

每次只修改一个变量，并固定：

- 问题集合及顺序；
- 数据与 Chunk；
- Embedding 和索引；
- Candidate-K 和最终 Top-K（除非它本身就是实验变量）；
- 生成模型；
- Prompt；
- temperature；
- max tokens；
- 评分代码。

建议报告表：

| 配置 | Candidate Recall | Hit@K | MRR@K | nDCG@K | EM | F1 | 检索延迟 | 重排延迟 | 总延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG | | | | | | | | 0 | |
| Dense + Reranker | | | | | | | | | |

有效性的优先顺序：

1. Candidate Recall 足够高，否则后级无解；
2. 最终 Hit/MRR/nDCG 提升；
3. 端到端 EM/F1 提升；
4. 人工检查事实性、引用和失败类型；
5. 质量提升足以抵消新增延迟与显存成本。

不能只用余弦相似度、单个成功案例或模型主观回答判断优化有效。

## 16. 已讨论但尚未实现/完成的方向

`进阶.md` 中记录的候选方向：

1. Dense + BM25（Best Matching 25，基于词项匹配的概率相关性排序算法），通过 RRF（Reciprocal Rank Fusion，倒数排名融合）融合；
2. Cross-Encoder Reranker（已经实现基线）；
3. RSE（Relevant Segment Extraction，相关片段提取）/滑动窗口和命中块邻居扩展；
4. Parent-Child Chunk；
5. Summary → Fine Chunk 两阶段检索；
6. 为每个 chunk 预生成问题；
7. Query Rewrite / Multi-Query；
8. 无答案阈值；
9. 引用正确性校验；
10. Faithfulness/幻觉指标；
11. OCR 噪声鲁棒性对比；
12. 使用更大的 Qwen 32B 生成模型。

建议顺序仍是：先完成 K=8 端到端实验和失败分析，再加入 BM25/RRF；不要同时修改多个组件，否则无法确定收益来源。

## 17. 当前已知限制和核查重点

1. **页面级 gold 不等于准确证据 chunk**：同页错误段落可能被算作 Hit；
2. **检索评测与生成评测 K 口径略有差异**：前者去重页面，后者使用实际 chunk；
3. **EM 对生成答案过严**：需结合 F1 和人工抽查；
4. **字符 F1 可能高估包含大量多余文本的答案**：未来可增加数值答案专用评分；
5. **Reranker 不是必然提升**：当前仍有 22/100 问题 F1 下降；
6. **候选召回仍是瓶颈**：100 题中有 8 题未进入 Dense Top-50；
7. **延迟明显增加**：Reranker 当前约增加 798 ms/query；
8. **Top-K 增大可能干扰 1.7B 模型**：必须以端到端 F1/EM 决策；
9. **模型加载时间未计入延迟**：当前延迟适合比较稳态推理，不代表冷启动；
10. **已有结果中的无 RAG 输出**：是在添加 `--skip-no-rag` 前产生，或运行时未携带该参数，并非开关失效。

## 18. 下一步最小实验计划

1. 使用完全相同的前 100 题运行 Reranker Top-8，携带 `--skip-no-rag`；
2. 对比 Top-5 与 Top-8 的 EM、F1、Hit、nDCG、生成延迟和总延迟；
3. 筛选以下四组问题人工查看：
   - K=5 miss、K=8 hit 且 F1 上升；
   - K=5 miss、K=8 hit 但 F1 下降；
   - Candidate Top-50 hit，但重排后仍 miss；
   - Candidate Top-50 miss；
4. 如果 K=8 F1 提升，再测试 K=10；否则保留 K=5；
5. 固定最终 K 后，再加入 BM25 + RRF，避免同时改变多个变量。

## 19. BM25 + RRF 混合检索实现与结果

已安装并接入 `bm25s 0.3.11` 与 `jieba 0.42.1`。BM25 复用现有 122518 个 Chunk，索引保存在
`storage/ohr_bench_bge_m3/bm25/`，大小约 49 MiB。当前完整检索链路为：

```text
Dense Top-50 ─┐
              ├→ RRF → Candidate-50 → Reranker → Top-K
BM25 Top-50 ──┘
```

100 题、Candidate-50、K=5 结果：

| 阶段 | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 64.00% | 60.50% | 49.53% | 50.34% |
| BM25 | 76.00% | 69.00% | 59.48% | 58.62% |
| RRF | 81.00% | 77.50% | 58.02% | 60.79% |
| RRF + Reranker | 83.00% | 80.00% | 64.02% | 66.00% |

Candidate-50 延迟：Dense 22.39 ms/query、BM25 1.17 ms/query、RRF 0.09 ms/query、Reranker
409.38 ms/query，完整检索链路 433.04 ms/query。RRF 相对 Dense 的 Hit@5 增加 17 个百分点，
相对提升 26.56%；RRF+Reranker 相对 Dense 增加 19 个百分点，相对提升 29.69%。

RRF 保留 100 个候选时 Candidate Hit 达到 98%，但把 100 个候选全部送入 Reranker 后，Top-5
Hit 为 82%、nDCG 为 65%，低于 Candidate-50 的 83% 和 66%，而延迟升至 811.20 ms/query。
因此当前推荐 Candidate-50，而不是 Candidate-100。
