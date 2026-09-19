# 标准、可落地的本地 RAG 框架

本文对应本目录里的可执行代码，目标是先完成一个容易检查、容易评测的 RAG 基线，再逐项增加
混合检索、reranker、父子块等高级能力。生成模型先使用本机的 Qwen3-1.7B W4A16 AWQ，后续可直接
换成 Qwen3-32B，而不需要重建检索框架。

## 1. 总体架构

```text
离线阶段
PDF / DOCX / MD / TXT
        │
        ▼
统一解析、清洗、保留来源元数据
        │
        ├── data/.../documents.jsonl       原始、可读、可追溯
        ▼
按自然段切 Chunk（超长段落才按句末继续切）
        │
        ├── storage/.../chunks.jsonl        Chunk 文本与来源
        ▼
BGE 文本向量化 → L2 归一化 → FAISS IndexFlatIP
        │
        ├── storage/.../index.faiss         向量索引
        └── storage/.../index_meta.json     可复现配置

在线阶段
用户问题 → 同一个 BGE 编码问题 → FAISS Top-K → 拼接受约束的 Qwen3 Chat Prompt
                                                │
                                                ▼
                           AutoQuantForCausalLM.from_quantized
                                                │
                                                ▼
                                      答案（可附资料编号）
```

RAG 中有两种完全不同的模型：

- **Embedding 模型**负责“找资料”，输出向量，不生成答案；
- **LLM** 负责“阅读召回资料并回答”，这里是 Qwen3-1.7B，未来可以换 Qwen3-32B。

因此，生成模型从 1.7B 换成 32B 时不必换 Embedding，也不必重建索引；只有更换
Embedding 模型时才必须重新生成全部文档向量与 FAISS。

## 2. 文档读取与统一格式

入口脚本是 `ingest_documents.py`，支持：

| 输入 | 解析库 | 当前行为 |
|---|---|---|
| `.pdf` | `pypdf` | 每页一条原始记录，保留 `page`；适合文字型 PDF |
| `.docx` | `python-docx` | 读取 Word paragraph，并以空行保留自然段边界 |
| `.md` / `.markdown` | Python 标准库 | 保留正文与段落边界；第一版不丢弃标题或代码块 |
| `.txt` | Python 标准库 | 依次尝试 UTF-8、UTF-8-SIG、GB18030 |

`.doc` 是旧二进制格式，先用 LibreOffice 转成 `.docx`。扫描版 PDF 没有文本层，`pypdf`
不能完成 OCR；生产阶段应在摄取之前接 PaddleOCR 等 OCR，并保存页码与置信度。

```bash
cd /home/lyc/workspace/rag
python ingest_documents.py /path/to/files_or_directory \
  --output data/custom/documents.jsonl
```

统一后的每行 JSON 示例：

```json
{"id":"doc-abc123","title":"产品手册","source_path":"/docs/manual.pdf","format":"pdf","page":3,"text":"正文……"}
```

`id` 是由绝对路径与页码产生的稳定 ID；PDF 的页码、原路径、格式会继续传到 Chunk，便于引用
与问题定位。若文件内容更新，应重新运行摄取和建库。

## 3. Chunk：第一版按自然段

基线规则如下：

1. 用一个或多个空行识别自然段；
2. 一个正常自然段就是一个 Chunk，不把无关段落强行拼在一起；
3. 只有单段超过 `chunk-size`（默认 500 个字符）时，才按 500 字窗口继续切；
4. 超长段落尽量在 `。！？；` 处结束，并保留默认 80 字重叠；
5. 每个 Chunk 保存 `source_id`、`paragraph_index`、`start/end`、路径和页码。

第一版用字符数而非 token 数，是因为中文没有稳定的空格分词，而且逻辑直观。上线前可改为所选
Embedding tokenizer 的 token 数，避免模型输入上限造成静默截断。

## 4. Embedding 选型

### 当前推荐：`BAAI/bge-m3`

当前代码使用 Transformers 直接加载，无需 LangChain 或 sentence-transformers。BGE-M3 输出
1024 维 dense 向量，支持中文、英文及多语言检索。当前 dense 检索模式下，问题与文档都直接
输入模型，不添加 instruction 前缀：

```text
Query: <用户问题>
Document: <标题、领域、章节和正文>
```

这里的 `Query:` 和 `Document:` 只是角色说明，不是实际拼入模型的字符串。查询和文档向量都做
L2 归一化，随后使用 FAISS `IndexFlatIP`；对归一化向量，
内积排序等价于余弦相似度。模型信息见
[BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3)，FAISS 的距离说明见
[FAISS 官方文档](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)。

选择建议：

| 阶段 | 推荐 | 理由 |
|---|---|---|
| 当前统一基线 | `BAAI/bge-m3` | 1024 维、多语言、适合 CMRC 与 OHR-Bench 共用 |
| 低资源速度对照 | `bge-small-zh-v1.5` | 512 维、中文、小而快，适合作为消融基线 |
| 中文质量对照 | `bge-large-zh-v1.5` | 1024 维中文模型，可与 BGE-M3 做 A/B 测试 |
| 新模型对比 | `Qwen3-Embedding-0.6B` | 32K、最高 1024 维，但资源开销更高且需实现 last-token pooling |

Qwen3-Embedding 的官方模型卡明确采用 instruction-aware 输入和不同的 last-token pooling，见
[Qwen3-Embedding-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。因此**不要在当前命令里
只把 `--embedding-model` 改成 Qwen3-Embedding**；应先增加对应适配器，再用同一评测集比较。

## 5. 离线存储与建库

安装依赖：

```bash
/home/lyc/workspace/vllm/.venv/bin/python -m pip install -r requirements.txt
```

对自有文档建库：

```bash
/home/lyc/workspace/vllm/.venv/bin/python offline_build.py \
  --documents data/custom/documents.jsonl \
  --output-dir storage/custom_bge_m3 \
  --device cpu \
  --chunk-size 500 \
  --overlap 80
```

三类离线产物的职责不同：

- `documents.jsonl`：解析后的原始数据，方便迁移、审计和重新切块；
- `chunks.jsonl`：实际送入 Embedding 的检索单元及来源映射；
- `index.faiss`：只存向量与顺序 ID；第 N 个向量严格对应 `chunks.jsonl` 第 N 行；
- `index_meta.json`：Embedding、维度、切块参数、条数、创建时间。

当前用精确的 `IndexFlatIP`，适合当前几百到几万 Chunk 的基线。数据达到百万级后，再用实际
延迟和 Recall 指标评估 HNSW 或 IVF；不要在基线阶段先引入近似检索误差。

## 6. 公开数据集与评测设计

### 旧的轻量对照：CMRC 2018 trial

CMRC 2018 是中文抽取式阅读理解数据集，官方仓库见
[ymcui/cmrc2018](https://github.com/ymcui/cmrc2018)。本项目把每个 context 当成知识库文档，把问题、
标准答案和 context ID 转成 `questions.jsonl`，因此既能测检索，也能测最终答案。

```bash
python prepare_cmrc2018.py
```

当前本地已有：

```text
data/raw/cmrc2018_trial.json        官方原始 JSON
data/cmrc2018/documents.jsonl       256 篇知识库文档
data/cmrc2018/questions.jsonl       1002 个问题与标准答案
```

注意：CMRC 原本是“给定文章做阅读理解”，转换成检索集合后适合冒烟测试，但不等于真实开放域
RAG。项目后续默认使用 OHR-Bench，建议按下面的层级评测：

| 层级 | 数据 | 目的 |
|---|---|---|
| 1 | OHR-Bench 每领域 2 份 PDF | 快速验证转换、Chunk、索引与指标链路 |
| 2 | 完整 OHR-Bench | 调整 Chunk、Embedding、Top-K 与重排策略 |
| 3 | OHR-Bench OCR 噪声文本 | 评估解析和检索鲁棒性 |
| 4 | 真实业务盲测集 | 评测“资料不存在”、时效性、权限、引用正确性 |

评测集必须与建库/调参数据分开，至少报告：

- 检索：Hit@K、Recall@K、MRR@K、nDCG@K；
- 生成：EM、字级 F1，以及人工检查“答案是否被资料支持”；
- 效率：建库耗时、单问题检索时延、首 token 时延、吞吐、显存。

## 7. CMRC 旧版冒烟流程（非默认）

### 7.1 建段落索引

```bash
/home/lyc/workspace/vllm/.venv/bin/python offline_build.py \
  --documents data/cmrc2018/documents.jsonl \
  --output-dir storage/cmrc2018_bge_m3 \
  --device cpu
```

### 7.2 先测检索

```bash
/home/lyc/workspace/vllm/.venv/bin/python evaluate_retrieval.py \
  --index-dir storage/cmrc2018_bge_m3 \
  --k 1 3 5 10
```

也可以人工看 Top-K：

```bash
/home/lyc/workspace/vllm/.venv/bin/python online_rag.py \
  --index-dir storage/cmrc2018_bge_m3 \
  --retrieve-only --show-context \
  --question "若游戏中离，则多少分钟内不得进行配对？"
```

### 7.3 用自写 AWQ loader 生成

```bash
/home/lyc/workspace/vllm/.venv/bin/python online_rag.py \
  --backend local-awq \
  --index-dir storage/cmrc2018_bge_m3 \
  --awq-model /home/lyc/workspace/lastdata/eagle3/qwen3_1_7b_awq_eagle3_benchmark/checkpoints/Qwen3-1.7B-W4A16-AWQ \
  --llmqrt-root /home/lyc/workspace/week78/LLMQRT \
  --question "若游戏中离，则多少分钟内不得进行配对？" \
  --top-k 3 --show-context
```

代码实际执行：

```python
wrapper = AutoQuantForCausalLM.from_quantized(
    model_path,
    max_seq_len=4096,
    torch_dtype=torch.float16,
    fuse_layers=False,
    device_map="cuda",
)
model = wrapper.model.eval()
```

### 7.4 Qwen3 的正确输入格式

不要手拼 `<|im_start|>`。代码把结构化 messages 交给该 checkpoint 自带 tokenizer：

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

RAG system message要求只使用提供资料、资料不足时明确拒答；user message包含带编号的 Top-K 资料
和问题。`enable_thinking=False` 可避免 64/128 个生成 token 被 Qwen3 的思考段占满。自动评测的
Prompt 只要求输出最短答案，不附解释或资料编号。

批量比较无 RAG / 有 RAG：

```bash
/home/lyc/workspace/vllm/.venv/bin/python evaluate_generation.py \
  --backend local-awq --limit 50 --top-k 3
```

### 7.5 本机实测结果（2026-09-18）

本目录当前产物已经实际运行过，不只是代码示例：

| 测试 | 结果 |
|---|---|
| 完整 1002 题检索 | Hit@1 `0.9182`、Hit@3 `0.9561`、Hit@10 `0.9800` |
| 10 题 AWQ 生成冒烟测试 | 无 RAG F1 `0.2013`，有 RAG F1 `0.5211` |
| 10 题 AWQ 生成冒烟测试 | EM `0.0000 → 0.1000`，Hit@3 `0.8000` |

生成测试确实走了本机 `runtime_refact.core.api.AutoQuantForCausalLM.from_quantized` 和
Qwen3-1.7B-W4A16-AWQ。详细结果位于 `results/generation_10/`；10 题只证明端到端链路和趋势，
不能当成稳定的模型质量结论。正式对比应至少跑完整 trial，并人工抽查错误类型。

## 8. OHR-Bench：混合长文档 RAG

CMRC 适合验证中文纯文本链路，但文档较短、结构简单。第二阶段使用 OHR-Bench 测试论文、
说明书、教材、财报、法律、报纸和行政文档中的章节、页面、表格、公式与图表。

### 8.1 本地原始文件与转换

当前原始文件：

```text
data/ohr_bench/raw/OHR-Bench_v2.parquet  # 8,561 个页面及 OCR/noise 文本
data/ohr_bench/raw/qas_v2.json           # 8,498 条官方 QA
data/ohr_bench/raw/retrieval.zip         # GT/OCR/噪声分文件语料，不包含 QA
```

`retrieval.zip` 不包含 `qas_v2.json`。`prepare_ohr_bench.py` 检测到 QA 不存在时，会从
OHR-Bench 官方 GitHub 下载，然后执行：

```bash
cd /home/lyc/workspace/rag
/home/lyc/workspace/vllm/.venv/bin/python prepare_ohr_bench.py
```

已生成：

```text
data/ohr_bench/documents.jsonl    # 8,259 个有文本页面，约 39 MB
data/ohr_bench/questions.jsonl    # 8,456 条可做文本检索评测的问题
data/ohr_bench/dataset_meta.json  # 来源、领域数量和页码约定
```

原始 Parquet 中一部分纯图表/公式页面的 `gt_text` 为空，不能建立文本向量，因此对应页面不写入
`documents.jsonl`。与这些空页面关联的 42 道 QA 同样被过滤，避免保留永远无法命中的
`gold_doc_ids`，导致 Recall 指标失真。

### 8.2 页面文档 JSONL

OHR-Bench 以页面作为检索指标的判断单元，以整份 PDF 作为逻辑父文档：

```json
{
  "id": "ohr-doc-e72b45cb02e5b151-page-0024",
  "document_id": "ohr-doc-e72b45cb02e5b151",
  "doc_name": "finance/JPMORGAN_2021Q1_10Q",
  "title": "JPMORGAN_2021Q1_10Q",
  "domain": "finance",
  "format": "structured_pdf_page",
  "source": "OHR-Bench",
  "page_index": 24,
  "page": 25,
  "section_path": ["Selected metrics"],
  "headings": [],
  "text": "..."
}
```

字段约定：

- `id`：页级 ID，也是问题中 `gold_doc_ids` 的目标；
- `document_id`：整份 PDF 的稳定父 ID；
- `page_index`：与官方 `evidence_page_no` 一致，从 0 开始；
- `page`：供界面和引用展示，从 1 开始；
- `section_path`：页面开始时从前页继承的章节路径；
- `headings`：本页识别出的 Markdown、Chapter/Section 或数字编号标题。

### 8.3 评测问题 JSONL

```json
{
  "id": "00073cc2-c801-467c-9039-fca63c78c6a9",
  "question": "What was the total amount of nonaccrual loans retained...?",
  "answers": ["842"],
  "gold_doc_ids": ["ohr-doc-e72b45cb02e5b151-page-0024"],
  "gold_document_ids": ["ohr-doc-e72b45cb02e5b151"],
  "doc_name": "finance/JPMORGAN_2021Q1_10Q",
  "domain": "finance",
  "answer_form": "Numeric",
  "evidence_source": "table",
  "evidence_page_indices": [24],
  "evidence_pages": [25]
}
```

一道问题可以包含多个 `gold_doc_ids`，对应跨页证据。`answers` 保持 list 格式，与 CMRC 评测器
兼容；额外字段可用于按领域、答案形式和证据类型分别统计结果。

### 8.4 章节感知 Chunk 格式

当前层级关系是：

```text
整份 PDF document_id
└── 页面 source_id / gold_doc_id
    └── 章节 section_path
        └── 段落 Chunk
```

`offline_build.py` 生成的 OHR Chunk 示例：

```json
{
  "id": "ohr-doc-...-page-0024-chunk-003",
  "source_id": "ohr-doc-...-page-0024",
  "document_id": "ohr-doc-...",
  "doc_name": "finance/JPMORGAN_2021Q1_10Q",
  "domain": "finance",
  "page_index": 24,
  "page": 25,
  "section_path": ["Selected metrics"],
  "paragraph_index": 3,
  "start": 1200,
  "end": 1638,
  "text": "..."
}
```

切块器会识别：

- Markdown `#` 到 `######`；
- `Chapter 1`、`Section 2`；
- `1 Introduction`、`2.1 Configuration` 等数字标题；
- 跨页继承的父章节。

实际送入 Embedding 的文本为：

```text
标题
领域
父章节 > 子章节
Chunk 正文
```

FAISS 只保存向量，`chunks.jsonl` 第 N 行与 FAISS 向量 ID N 一一对应。Chunk 数量或顺序发生
变化后必须重建 FAISS。

### 8.5 页级检索指标

一个页面通常被切成多个 Chunk。`evaluate_retrieval.py` 会先多召回候选 Chunk，再按
`source_id` 去重，然后计算 Hit@K、Recall@K、MRR 和 nDCG。因此 Hit@10 表示前 10 个不同
页面，而不是同一页的 10 个段落。

默认候选数量为：

```text
max(K) × candidate_multiplier
```

其中 `candidate_multiplier` 默认是 20，可通过命令行调整。

### 8.6 均衡冒烟测试

先从每个领域选择 2 份 PDF：

```bash
python prepare_ohr_bench.py \
  --max-documents-per-domain 2 \
  --output-dir data/ohr_bench_smoke

python offline_build.py \
  --documents data/ohr_bench_smoke/documents.jsonl \
  --output-dir storage/ohr_bench_smoke_bge_small_zh \
  --device cuda \
  --batch-size 64

python evaluate_retrieval.py \
  --questions data/ohr_bench_smoke/questions.jsonl \
  --index-dir storage/ohr_bench_smoke_bge_small_zh \
  --embedding-device cuda \
  --output-dir results/ohr_bench_smoke \
  --k 1 3 5 10
```

本机实测：

| 项目 | 数量/结果 |
|---|---:|
| 逻辑 PDF | 14 |
| 有效页面 | 477 |
| 问题 | 502 |
| Chunk | 5,638 |
| Hit@1 | 0.2171 |
| Hit@10 | 0.4203 |

该测试使用中文专用 `bge-small-zh-v1.5`，主要用于证明数据映射、Chunk 和指标链路跑通。
OHR-Bench 以英文为主，并包含大量表格、公式和图表，不能把这个数值当成最终 Embedding 质量。

### 8.7 完整建库与 Embedding 选择

完整数据按当前参数预计约产生 122,518 个 Chunk。BGE-M3 使用 1024 维 float32，Flat 索引仅
向量部分约 479 MiB。当前默认数据集和模型已经统一为 OHR-Bench + `BAAI/bge-m3`：

```bash
python offline_build.py \
  --documents data/ohr_bench/documents.jsonl \
  --output-dir storage/ohr_bench_bge_m3 \
  --embedding-model BAAI/bge-m3 \
  --query-prefix "" \
  --device cuda \
  --batch-size 16

python evaluate_retrieval.py \
  --questions data/ohr_bench/questions.jsonl \
  --index-dir storage/ohr_bench_bge_m3 \
  --embedding-device cuda \
  --output-dir results/ohr_bench_retrieval \
  --k 1 3 5 10
```

旧 BGE-small 索引不能复用，必须重新生成全部向量和 FAISS。Qwen3-Embedding 使用 instruction-aware Query
和 last-token pooling，不能只修改 `--embedding-model`，需要先实现对应 Embedder。

### 8.8 OCR 鲁棒性实验

转换器默认使用人工核验的 `gt_text`。也可以从同一个 Parquet 选择带 OCR 错误的字段：

```bash
python prepare_ohr_bench.py \
  --text-column semantic_noise_MinerU_moderate \
  --output-dir data/ohr_bench_mineru_moderate
```

对 GT 与噪声版本固定 QA、Embedding、Chunk、Top-K 和生成 Prompt，比较指标差值，即可量化 OCR
错误对检索和回答的影响。

### 8.9 Cross-Encoder Reranker 基线

当前支持 `BAAI/bge-reranker-v2-m3`，位于 FAISS 候选召回之后、构造 LLM Prompt 之前。它是
查询时组件，不需要重建现有 FAISS：

```text
BGE-M3 + FAISS Top-50 → BGE Reranker → Top-5 → Qwen3
```

先用固定问题集同时比较 Dense 与重排结果：

```bash
python evaluate_retrieval.py \
  --embedding-device cpu \
  --reranker --reranker-device cuda \
  --candidate-k 50 --k 1 3 5 10 \
  --limit 50
```

单问题检查：

```bash
python online_rag.py \
  --reranker --reranker-device cpu \
  --candidate-k 50 --top-k 5 \
  --retrieve-only --show-context \
  --question "What was the total amount of nonaccrual loans retained as of March 31, 2021?"
```

报告至少包含 Candidate Recall@50，以及重排前后的 Recall@K、MRR@K、nDCG@K 和每题耗时。
`predictions.jsonl` 同时保存 Dense/Reranker 的 chunk 排名与两种分数，便于检查具体升降原因。
Reranker 不是必然提升：只有完整固定测试集上的指标差值为正，才能认定该配置有效；不能用单个
成功例子下结论。端到端生成评测还应保持相同的最终 Top-K、Qwen、Prompt 和 Token 预算。

### 8.10 Dense + BM25 + RRF 混合检索

当前已实现 `bm25s + jieba` 关键词索引，以及按 Chunk ID 去重的 Reciprocal Rank Fusion：

```text
FAISS Dense Top-50 ─┐
                    ├→ RRF Top-50 → Reranker → Top-K → Qwen
BM25 Top-50 ────────┘
```

RRF 只融合排名，不直接相加不可比的余弦分数和 BM25 分数：

```text
RRF(d) = Σ 1 / (60 + rank_m(d))
```

先运行 `python build_bm25.py`，再给评测或在线程序增加 `--hybrid`。评测输出同时保留
`dense_rank`、`bm25_rank`、`rrf_score` 和 `rerank_score`，并报告每个阶段的指标与延迟。

## 9. 从基线到生产的升级顺序

建议一次只改一个变量并保留评测结果：

1. 段落 Chunk 的 `chunk-size` 与 Top-K；
2. 已实现：dense + BM25，用 RRF 融合；
3. 已实现：Top-N 召回后用 cross-encoder reranker 排到 Top-3/5；
4. 命中 Chunk 自动补相邻段，或建立父文档/子 Chunk；
5. 查询改写与多查询检索；
6. 无答案阈值、引用校验、失败样本集；
7. 再尝试摘要层、HyDE/预生成问题、Self-RAG。

这套顺序让每个复杂组件都能用固定测试集证明收益，避免同时修改 Embedding、Chunk、Prompt 和
生成模型后无法知道提升来自哪里。
