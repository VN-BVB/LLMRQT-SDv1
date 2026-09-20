# 一个适合入门的标准 RAG：Qwen3-1.7B AWQ + BGE + FAISS

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
RAG/
├── 标准RAG框架.md            # 完整架构、选型、数据格式和端到端流程
├── ingest_documents.py      # PDF/DOCX/MD/TXT -> 原始 documents.jsonl
├── prepare_cmrc2018.py      # 下载/整理公开数据集
├── prepare_ohr_bench.py     # OHR-Bench Parquet/QA -> 页面文档与评测 JSONL
├── OHR-Bench使用.md         # 混合长文档数据集的转换、建库和评测
├── offline_build.py         # 离线建 FAISS 索引
├── online_rag.py            # 在线检索并选择一种生成后端回答
├── build_hierarchical_index.py # 构建父子层次化索引
├── online_hierarchical_rag.py  # 父级路由后检索细粒度 Chunk
├── build_question_index.py  # 为 Chunk 预生成问题建立索引
├── question_retriever.py    # 原文/问题/BM25 多路召回
├── reranker.py              # BGE Cross-Encoder 候选重排
├── bm25_retriever.py        # bm25s 关键词检索
├── hybrid_retriever.py      # Dense + BM25 的 RRF 融合
├── build_bm25.py            # 从现有 Chunk 建立 BM25 索引
├── generation_backends.py   # vLLM HTTP / 本地 AWQ 两种生成接口
├── evaluate_retrieval.py    # 不启动 LLM，只测检索
├── evaluate_generation.py   # 比较无 RAG 与有 RAG 的答案
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

现有 vLLM 环境已经有 PyTorch、Transformers 和 NumPy；还会安装 `faiss-cpu`、`pypdf`
和 `python-docx`：

```bash
# 从仓库根目录进入
cd RAG
python -m pip install -r requirements.txt
```

如果使用已有的 vLLM 或其他虚拟环境，请先激活该环境，再执行上面的安装命令。

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
