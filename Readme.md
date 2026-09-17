<div align="center">

# ⚡ LLM Inference Optimization Lab

### Quantization · CUDA Runtime · Speculative Decoding

从 **模型量化**、**低比特算子** 到 **EAGLE / SSD 推测解码** 的端到端大模型推理优化实验仓库。

<p>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/vLLM-0.26-5B5BD6?style=for-the-badge" alt="vLLM">
  <img src="https://img.shields.io/badge/Platform-NVIDIA_GPU-111111?style=for-the-badge&logo=nvidia" alt="NVIDIA GPU">
</p>

<p>
  <a href="#-项目概览">项目概览</a> ·
  <a href="#-核心能力">核心能力</a> ·
  <a href="#-性能亮点">性能亮点</a> ·
  <a href="#-快速开始">快速开始</a> ·
  <a href="#-目录结构">目录结构</a>
</p>

</div>

---

<table align="center">
  <tr>
    <td align="center"><b>🚀 2.46×</b><br><sub>吞吐提升</sub><br><sub>66.29 → 163.10 tok/s</sub></td>
    <td align="center"><b>💾 −52.1%</b><br><sub>引擎显存增量</sub><br><sub>4159 → 1994 MiB</sub></td>
    <td align="center"><b>✅ 1152 / 1152</b><br><sub>Token 严格一致</sub><br><sub>确定性验证路径</sub></td>
    <td align="center"><b>🧩 3 Stages</b><br><sub>量化 → Runtime → 推测解码</sub><br><sub>端到端优化闭环</sub></td>
  </tr>
</table>

> [!NOTE]
> 上述吞吐与显存数据来自 RTX 4060 Laptop、Qwen3-1.7B、batch=1、固定生成 128 tokens 的吞吐优先配置。严格 Token 一致性来自另一组 batch-invariant + eager 配置；两组结果不可混为同一测试设置。详见[性能亮点](#-性能亮点)。

## 🌌 项目概览

本仓库围绕 LLM 推理效率构建三层优化栈：

1. **LLMQT**：将 FP16/BF16 模型转换为 AWQ W4A16、SmoothQuant INT8 或 FP8 模型。
2. **LLMQRT**：使用 CUDA、CUTLASS 与 Triton 实现低比特 GEMM/GEMV 和量化模型推理。
3. **Speculative Decoding**：实现 EAGLE-1/2/3、EAGLE3Pro 与 SSD，减少 Target Model 的串行解码轮数。

```mermaid
flowchart LR
    A["🤗 FP16 / BF16<br/>Hugging Face Model"] --> B{"⚖️ LLMQT<br/>Offline Quantization"}
    B -->|AWQ W4A16| C1["INT4 Checkpoint"]
    B -->|SmoothQuant| C2["INT8 Checkpoint"]
    B -->|Static / Dynamic FP8| C3["FP8 Checkpoint"]

    C1 --> D["🔥 LLMQRT<br/>CUDA · CUTLASS · Triton"]
    C2 --> D
    C3 --> D

    D --> E["Target Model"]
    F["Draft Model"] --> G{"🌲 Speculative Decoding<br/>Draft · Verify · Accept"}
    E --> G
    G --> H["⚡ Faster Token Generation"]

    classDef quant fill:#6C5CE7,color:#fff,stroke:#A29BFE
    classDef runtime fill:#E17055,color:#fff,stroke:#FAB1A0
    classDef spec fill:#0984E3,color:#fff,stroke:#74B9FF
    classDef output fill:#00B894,color:#fff,stroke:#55EFC4
    class B,C1,C2,C3 quant
    class D,E runtime
    class F,G spec
    class H output
```

## ✨ 核心能力

| 模块 | 能力 | 关键技术 | 入口 |
|---|---|---|---|
| **LLMQT** | 离线权重量化与校准 | AWQ、SmoothQuant、Static/Dynamic FP8 | [`workspace/week78/LLMQT`](workspace/week78/LLMQT) |
| **LLMQRT** | 量化模型加载与推理 | CUDA Extension、CUTLASS、Triton、低比特 GEMM/GEMV | [`workspace/week78/LLMQRT`](workspace/week78/LLMQRT) |
| **EAGLE-1** | 静态 Top-K 候选树 | Feature-conditioned Draft、Tree Verify | [`spec_decoding/eagle`](workspace/week91011/spec-decoding-src-code/spec-decoding-main/spec_decoding/eagle) |
| **EAGLE-2** | 动态候选树 | 累计概率 Beam、全局剪枝 | [`spec_decoding/eagle2`](workspace/week91011/spec-decoding-src-code/spec-decoding-main/spec_decoding/eagle2) |
| **EAGLE-3** | 多层特征 Draft | Multi-layer Features、Draft Vocabulary、KV Cache | [`spec_decoding/eagle3_sgl`](workspace/week91011/spec-decoding-src-code/spec-decoding-main/spec_decoding/eagle3_sgl) |
| **EAGLE3Pro** | 面向部署的优化实现 | Packed Verify、Flash KV、CUDA Graph、KV Compact | [`spec_decoding/eagle3pro`](workspace/week91011/spec-decoding-src-code/spec-decoding-main/spec_decoding/eagle3pro) |
| **SSD** | Draft / Verify 并行探索 | Async Speculation、Speculation Cache、独立 CUDA Stream | [`eagle_ssd`](workspace/week91011/spec-decoding-src-code/eagle_ssd) |

### 量化能力

| 方法 | 权重 / 激活 | 适合场景 | Runtime 路径 |
|---|---|---|---|
| **AWQ** | W4A16 | 显存敏感、单请求低延迟 | INT4 GEMM / GEMV、Marlin/CUTLASS |
| **SmoothQuant** | W8A8 | 权重与激活统一 INT8 | INT8 GEMM |
| **Static FP8** | FP8 + 离线 Scale | 稳定部署、固定校准分布 | Tensor-wise / Row-wise FP8 |
| **Dynamic FP8** | FP8 + 在线 Scale | 输入分布变化较大 | Dynamic Quant + FP8 GEMM |

代码中包含 Llama、Qwen2/Qwen3、Qwen3-MoE、DeepSeek 与 OPT 等模型适配；不同组合的完整验证状态以对应示例脚本为准。

### 推测解码演进

```mermaid
flowchart LR
    E1["EAGLE-1<br/>静态 Top-K 树"] --> E2["EAGLE-2<br/>累计概率动态剪枝"]
    E2 --> E3["EAGLE-3<br/>多层特征 + AR Draft"]
    E3 --> P["EAGLE3Pro<br/>Packed Verify + Flash KV"]
    P --> CG["CUDA Graph<br/>固定 Shape + Static Cache"]
    P -. 并行调度探索 .-> SSD["SSD<br/>Async Draft / Verify"]

    classDef base fill:#2D3436,color:#fff,stroke:#636E72
    classDef eagle fill:#6C5CE7,color:#fff,stroke:#A29BFE
    classDef pro fill:#00B894,color:#fff,stroke:#55EFC4
    classDef explore fill:#FDCB6E,color:#2D3436,stroke:#E17055
    class E1,E2,E3 eagle
    class P,CG pro
    class SSD explore
```

## 📊 性能亮点

### 吞吐优先：AWQ Marlin + CUDA Graph

测试环境：**RTX 4060 Laptop 8 GiB · Qwen3-1.7B · batch=1 · 128 output tokens · vLLM 0.26.0**。

```mermaid
xychart-beta
    title "Qwen3-1.7B Decode Throughput"
    x-axis ["FP16 Target", "AWQ Target", "AWQ + EAGLE3"]
    y-axis "tokens / second" 0 --> 180
    bar [66.29, 155.32, 163.10]
```

| 配置 | 吞吐 | 相比 FP16 | 引擎显存增量 | TPOT |
|---|---:|---:|---:|---:|
| FP16 Target | 66.29 tok/s | 1.000× | 4159 MiB | 15.03 ms |
| AWQ Marlin Target | 155.32 tok/s | 2.343× | 1763 MiB | 6.35 ms |
| **AWQ Marlin + EAGLE3 (`k=1`)** | **163.10 tok/s** | **2.460×** | **1994 MiB** | **5.97 ms** |

关键结论：

- AWQ + Marlin 是主要收益来源：相较 FP16 Target 达到 **2.343×**。
- EAGLE3 在 AWQ Target 基线上进一步贡献 **5.01%** 吞吐提升。
- 完整方案相较 FP16 Target 的引擎显存增量降低 **52.06%**。
- 当前短文本、单并发负载下 `k=1` 最优；Speculative Tokens 并非越多越快。

<details>
<summary><b>查看 Speculative Token 扫描结果</b></summary>

| Speculative Tokens | 吞吐 | 相比 AWQ Target | 平均每轮推进 Token |
|---:|---:|---:|---:|
| `k=1` | **163.10 tok/s** | **+5.01%** | 1.534 |
| `k=2` | 158.11 tok/s | +1.80% | 1.733 |
| `k=3` | 155.14 tok/s | −0.12% | 1.837 |
| `k=5` | 137.29 tok/s | −11.61% | 1.887 |

随着 `k` 增大，接受长度上升，但更多 Draft Forward 和更宽的 Target Verify 会抵消收益。

</details>

### 严格一致：Batch-Invariant + Eager

为验证算法等价性，另一组实验禁用 CUDA Graph，并使用确定性 AWQ 反量化 + MatMul 路径：

| 配置 | 吞吐 | TPOT | E2E / 请求 | Token 一致性 |
|---|---:|---:|---:|:---:|
| AWQ INT4 Target | 22.17 tok/s | 45.03 ms | 5772.84 ms | Oracle |
| **AWQ INT4 + EAGLE3** | **29.03 tok/s** | **33.90 ms** | **4409.27 ms** | **1152 / 1152** |

- 吞吐提升 **30.92%**，TPOT 降低 **24.72%**。
- 三条固定 Prompt 重复三轮，完整 Token ID **1152 / 1152 严格一致**。
- 默认 Marlin + CUDA Graph 适合吞吐优先，但临界 Argmax 可能因数值路径差异产生 Token 分叉，不能直接宣称严格无损。

## 🗂️ 目录结构

```text
.
├── README.md
└── workspace
    ├── week78
    │   ├── LLMQT                       # 模型量化工具
    │   │   └── quant
    │   │       ├── core                # 配置与量化 API
    │   │       ├── quantization        # AWQ / SQ / FP8
    │   │       ├── nn_models           # 模型与量化层适配
    │   │       ├── examples            # 量化与精度示例
    │   │       └── utils               # 校准、Scale、Packing 工具
    │   └── LLMQRT                      # 低比特推理 Runtime
    │       └── runtime_refact
    │           ├── core                # Runtime API
    │           ├── csrc                # CUDA / C++ Kernels
    │           ├── triton_kernels      # Triton Kernels
    │           ├── nn_models           # Runtime 模型实现
    │           └── example             # AWQ / SQ / FP8 示例
    └── week91011
        └── spec-decoding-src-code
            ├── spec-decoding-main
            │   └── spec_decoding
            │       ├── eagle           # EAGLE-1
            │       ├── eagle2          # EAGLE-2
            │       ├── eagle3_sgl      # EAGLE-3
            │       ├── eagle3_sgl_profile_opti
            │       └── eagle3pro       # 部署优化版本
            └── eagle_ssd               # Speculative Speculative Decoding
```

## 🚀 快速开始

### 1. 推荐环境

| 组件 | 推荐配置 |
|---|---|
| OS | Linux / WSL2 |
| GPU | NVIDIA GPU，Compute Capability 8.0+ |
| Python | 3.11 / 3.12 |
| CUDA | 12.x |
| PyTorch | 2.7 系列 |
| 容器 | `nvcr.io/nvidia/pytorch:25.04-py3` |

> [!IMPORTANT]
> 模型权重不包含在本仓库中。运行端到端示例前，请自行准备 Hugging Face 权重路径，并确认对应模型许可证与访问权限。

### 2. LLMQT：生成量化模型

```bash
cd workspace/week78/LLMQT

python -m pip install -r requirements.txt
python setup_quant.py install

# 根据本机模型路径修改示例配置后运行
python quant/examples/awq_quantize.py
# 或
python quant/examples/fp8_static_quantize.py
```

主要示例：

```text
quant/examples/awq_quantize.py          AWQ 量化
quant/examples/awq_quantize_and_run.py  量化并运行
quant/examples/sq_quantize.py           SmoothQuant
quant/examples/fp8_static_quantize.py   Static FP8
quant/examples/fp8_dyn_quantize.py      Dynamic FP8
```

### 3. LLMQRT：编译低比特 CUDA Runtime

```bash
cd workspace/week78/LLMQRT

python -m pip install -r requirements.txt
python setup_runtime.py install

# 示例：运行 Qwen3 AWQ
python runtime_refact/example/awq_run_qwen3.py
```

编译脚本会根据当前 GPU 读取 Compute Capability，并构建 PyTorch CUDA Extension。请确保 `gcc`、`g++`、`nvcc` 与 PyTorch ABI 兼容。

### 4. EAGLE3Pro：运行推测解码

```bash
cd workspace/week91011/spec-decoding-src-code/spec-decoding-main

python -m pip install -e spec_decoding/eagle3pro

# CPU 冒烟测试：不需要真实大模型
PYTHONPATH=. python -m \
  spec_decoding.eagle3pro.example.run_smoke_eagle3_sgl \
  --device cpu

# Qwen3 端到端示例
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m \
  spec_decoding.eagle3pro.example.run_eagle3_sgl_qwen3 \
  --topk 1 \
  --num-steps 1 \
  --max-tree-nodes 1 \
  --max-new-tokens 128
```

### 5. SSD：Draft / Verify 并行实验

```bash
cd workspace/week91011/spec-decoding-src-code/eagle_ssd

uv sync
source .venv/bin/activate

export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
export SSD_CUDA_ARCH=8.9   # RTX 40 系；H100 使用 9.0

cd bench
python -O bench.py --llama --size 70 --gpus 5 --spec --async \
  --k 7 --f 3 --b 1 --numseqs 128 --output_len 512 --all
```

## 🧪 验证方法

性能数字只有在测试口径一致时才有意义。本项目重点记录以下指标：

| 类别 | 指标 |
|---|---|
| 正确性 | Token ID 严格一致、PPL、跨轮重复稳定性 |
| 延迟 | TTFT、TPOT、E2E Latency |
| 吞吐 | Tokens/s、Acceptance Length、Rounds |
| 显存 | Checkpoint Size、引擎驻留显存增量、KV Cache |
| Kernel | CUDA Event、Torch Profiler、Chrome Trace |

推荐遵循以下基准原则：

- Target-only 与 Speculative Decoding 使用相同 Prompt、生成长度和随机设置。
- 每组配置使用独立进程完成加载、预热和计时。
- 将“吞吐优先”和“严格确定性”作为两种配置分别报告。
- 不把 AWQ、Marlin、CUDA Graph 与 EAGLE3 的收益重复归因。
- 不将 batch=1、短上下文结果直接外推至高并发或长上下文服务。

## 🧭 Roadmap

- [x] AWQ / SmoothQuant / FP8 量化流程
- [x] CUDA / CUTLASS / Triton 低比特 Kernel
- [x] EAGLE-1 / EAGLE-2 / EAGLE-3 教学实现
- [x] EAGLE3Pro + vLLM + AWQ Marlin 集成
- [x] CUDA Graph、Static KV Cache 与 Packed Verify
- [x] SSD 异步调度与 Speculation Cache 探索
- [ ] Multi Lora训练，调度
- [ ] 多 Batch / Continuous Batching 系统化基准
- [ ] 长上下文与多并发参数扫描
- [ ] OpenAI-compatible Serving API
- [ ] 更完整的 CI、跨 GPU 架构测试与可复现实验脚本

## ⚠️ 使用说明

本仓库以研究、学习与性能实验为目的：

- 部分脚本中的模型路径、设备编号和缓存路径需要按本机环境修改。
- 不同 GPU、CUDA、PyTorch、Transformers 与 vLLM 版本可能产生性能或数值差异。
- 仓库包含 CUTLASS 等第三方源码；公开发布前请保留其原始版权声明并检查各依赖许可证。
- 当前仓库尚未声明统一开源许可证。若计划接受外部贡献，建议在发布前补充根目录 `LICENSE`。

## 🤝 Contributing
欢迎提交 Issue 或 Pull Request。若贡献新的 Kernel 或性能结果，请同时提供：

1. GPU、CUDA、PyTorch 与依赖版本；
2. 模型、精度、Batch、Prompt / Output Length；
3. 完整启动参数与预热方式；
4. 正确性验证结果；
5. Target-only 与优化方案的同口径对照。

---

<div align="center">

**Build smaller. Run faster. Verify every token.**

<sub>如果这个项目对你有帮助，欢迎点一个 ⭐</sub>

</div>