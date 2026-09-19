# OHR-Bench 转换、建库与评测

## 1. 本地原始文件

当前转换需要两个源文件：

```text
data/ohr_bench/raw/OHR-Bench_v2.parquet  # 8,561 个页面及不同 OCR/noise 文本
data/ohr_bench/raw/qas_v2.json           # 8,498 条官方 QA
```

`retrieval.zip` 包含 GT、MinerU、Qwen2.5-VL 和噪声版本的分文件语料，但不包含 QA。
`prepare_ohr_bench.py` 在 `qas_v2.json` 不存在时会从 OHR-Bench 官方 GitHub 自动下载。

## 2. 转换完整数据

```bash
cd /home/lyc/workspace/rag
/home/lyc/workspace/vllm/.venv/bin/python prepare_ohr_bench.py
```

当前实测输出：

```text
data/ohr_bench/documents.jsonl   # 8,259 个有文本的页面，约 39 MB
data/ohr_bench/questions.jsonl   # 8,456 条可做文本检索评测的问题，约 6.2 MB
data/ohr_bench/dataset_meta.json # 领域统计、来源和页码约定
```

原始 Parquet 有 8,561 行，其中一部分页面的 `gt_text` 为空（常见于纯图表页面），无法建立
文本向量，因此不写入 documents。相应的 42 条 QA 也被过滤；不能把不存在的证据页留在
`gold_doc_ids` 中，否则 Recall 指标没有意义。

## 3. JSONL 格式

OHR-Bench 以“页面”作为检索评测单元，以整份 PDF 作为逻辑父文档：

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

- `page_index` 与官方 `evidence_page_no` 一致，从 0 开始；
- `page` 用于展示，从 1 开始；
- `id` 是页级 ID，供 `gold_doc_ids` 和检索指标使用；
- `document_id` 是整份 PDF 的父 ID；
- `section_path` 是从前页继承的章节上下文；
- `headings` 是本页识别到的 Markdown、Chapter/Section 或数字编号标题。

问题格式：

```json
{
  "id": "00073cc2-c801-467c-9039-fca63c78c6a9",
  "question": "What was the total amount of nonaccrual loans retained...?",
  "answers": ["842"],
  "gold_doc_ids": ["ohr-doc-e72b45cb02e5b151-page-0024"],
  "gold_document_ids": ["ohr-doc-e72b45cb02e5b151"],
  "domain": "finance",
  "answer_form": "Numeric",
  "evidence_source": "table",
  "evidence_page_indices": [24],
  "evidence_pages": [25]
}
```

## 4. 新 Chunk 格式

`offline_build.py` 现在会生成兼容 CMRC 和 OHR-Bench 的 Chunk：

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

Embedding 的实际文本由以下内容拼成：

```text
标题
领域
父章节 > 子章节
Chunk 正文
```

FAISS 仍然只保存向量；`chunks.jsonl` 第 N 行与 FAISS 向量 ID N 一一对应。

## 5. 先跑均衡冒烟集

每个领域选择 2 份 PDF：

```bash
python prepare_ohr_bench.py \
  --max-documents-per-domain 2 \
  --output-dir data/ohr_bench_smoke
```

然后建库：

```bash
python offline_build.py \
  --documents data/ohr_bench_smoke/documents.jsonl \
  --output-dir storage/ohr_bench_smoke_bge_m3 \
  --device cuda \
  --batch-size 16
```

评测页级检索：

```bash
python evaluate_retrieval.py \
  --questions data/ohr_bench_smoke/questions.jsonl \
  --index-dir storage/ohr_bench_smoke_bge_m3 \
  --embedding-device cuda \
  --output-dir results/ohr_bench_smoke \
  --k 1 3 5 10
```

`evaluate_retrieval.py` 会先多召回 Chunk，再按 `source_id` 去重，保证 Hit@10 中的 10 表示
10 个不同证据页，而不是同一页的 10 个段落。

本机此前用每类 2 份 PDF 得到 477 页、502 个问题、5,638 个 Chunk。旧的中文专用
`bge-small-zh-v1.5` 测得 Hit@1=0.2171、Hit@10=0.4203。这个结果主要用于证明数据映射和
评测代码跑通，不能作为当前 BGE-M3 的基线；新命令会把结果写入独立的 BGE-M3 目录。

## 6. 完整建库

完整数据按当前参数约产生 122,518 个 Chunk，BGE-M3 的 1024 维 float32 Flat 索引仅向量
部分约 479 MiB。建议先完成冒烟集，再运行：

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

旧 Embedding 索引不能复用，必须重新建 FAISS。Qwen3-Embedding 还需要 last-token pooling 适配，不能只改
模型名。

## 7. OCR 鲁棒性实验

转换器默认读取人工核验的 `gt_text`。也可以选择 Parquet 中的噪声字段，例如：

```bash
python prepare_ohr_bench.py \
  --text-column semantic_noise_MinerU_moderate \
  --output-dir data/ohr_bench_mineru_moderate
```

对 GT 和噪声版本使用相同的 QA、Embedding、Chunk 和 Top-K，比较指标差值，即可量化 OCR
错误对检索的影响。
