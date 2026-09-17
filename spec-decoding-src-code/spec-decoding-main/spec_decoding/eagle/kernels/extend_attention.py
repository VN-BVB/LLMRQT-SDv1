"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl

_is_cuda = torch.cuda.is_available()
CUDA_CAPABILITY = (0, 0)
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        Lq: Query head dimension
        Lv: Value head dimension

    Returns:
        tuple: (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    # Determine BLOCK_DMODEL and BLOCK_DPE based on head dimension
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0

    BLOCK_DV = triton.next_power_of_2(Lv)

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware (CUDA)
    if _is_cuda and CUDA_CAPABILITY[0] == 12:
        # sm120 workstation Blackwell architecture (RTX Pro 6000) has a much smaller shared memory size (100K)
        if Lq <= 128:
            BLOCK_M, BLOCK_N = (64, 128)
        elif Lq <= 256:
            BLOCK_M, BLOCK_N = (64, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 32)
    elif _is_cuda and CUDA_CAPABILITY[0] >= 9:
        # Hopper architecture (H100, etc.)
        if Lq <= 256:
            BLOCK_M, BLOCK_N = (128, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
        # Ampere architecture (A100, etc.)
        # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
        if CUDA_CAPABILITY[1] == 9 or CUDA_CAPABILITY[1] == 6:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (64, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 32)
        else:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
    else:
        # Older architectures / CPU fallback sizing
        BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

    num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


# Triton kernel：把前缀 KV indices与extend KV indices并行拷贝拼成统一索引数组。
@triton.jit
def _copy_unified_indices_kernel(
    # Input buffers
    prefix_kv_indptr,
    prefix_kv_indices,
    extend_start_loc,
    extend_seq_lens,
    extend_kv_indices,
    unified_kv_indptr,
    # Output buffer
    unified_kv_indices,
    # Size
    bs,
):
    """
    Triton kernel to copy indices to unified buffer (parallel per sequence).
    Each thread block processes one sequence with vectorized loads/stores.
    """
    pid = tl.program_id(0)

    if pid >= bs:
        return

    # Load sequence info
    prefix_start = tl.load(prefix_kv_indptr + pid)
    prefix_end = tl.load(prefix_kv_indptr + pid + 1)
    extend_start = tl.load(extend_start_loc + pid)
    extend_len = tl.load(extend_seq_lens + pid)

    prefix_len = prefix_end - prefix_start
    unified_start = tl.load(unified_kv_indptr + pid)

    # Copy indices in vectorized chunks
    BLOCK_SIZE: tl.constexpr = 128

    # Process prefix indices
    for block_start in range(0, prefix_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < prefix_len

        src_idx = prefix_start + offs
        dst_idx = unified_start + offs

        vals = tl.load(prefix_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)

    # Process extend indices
    for block_start in range(0, extend_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < extend_len

        src_idx = extend_start + offs
        dst_idx = unified_start + prefix_len + offs

        vals = tl.load(extend_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)


def build_unified_kv_indices(
    prefix_kv_indptr: torch.Tensor,
    prefix_kv_indices: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_kv_indices: torch.Tensor,
    bs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build unified KV indices efficiently:
    - Use PyTorch's optimized cumsum (NVIDIA CUB) for indptr
    - Use Triton kernel for parallel index copying

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    device = prefix_kv_indptr.device

    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]

    # Create unified_kv_indptr avoiding direct assignment (for CUDA graph compatibility)
    unified_lens = prefix_lens + extend_seq_lens[:bs]
    unified_kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(unified_lens, dim=0),
        ]
    )

    max_unified_len = len(prefix_kv_indices) + len(extend_kv_indices)

    unified_kv_indices = torch.empty(max_unified_len, dtype=torch.int64, device=device)

    # Launch Triton kernel for parallel index copying
    _copy_unified_indices_kernel[(bs,)](
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        extend_kv_indices,
        unified_kv_indptr,
        unified_kv_indices,
        bs,
    )

    return unified_kv_indptr, unified_kv_indices, prefix_lens


# Triton kernel：extend（prefill + KV cache）注意力前向主体，支持 custom_mask（树掩码）。
#
# 输入张量的逻辑形状：
#   Q_Extend: [所有请求的 extend token 总数, query_head_num, Lq]
#   K_Extend: [所有请求的 extend token 总数, kv_head_num,    Lq]
#   V_Extend: [所有请求的 extend token 总数, kv_head_num,    Lv]
#   O_Extend: [所有请求的 extend token 总数, query_head_num, Lv]
#   K_Buffer: [KV cache 槽位数, kv_head_num, Lq]
#   V_Buffer: [KV cache 槽位数, kv_head_num, Lv]
#
# 固定一个请求和一个 head 后，完整注意力矩阵的逻辑分区如下。E 表示该请求的
# extend_len，P 表示 prefix_len；kernel 不会真的生成这张完整矩阵，而是分块计算：
#
#                       KV 列：P 个 prefix       KV 列：E 个 extend
#                    ┌────────────────────────┬────────────────────────┐
#                    │                        │                        │
#   Query：E 个      │ stage 1 分数 [E, P]   │ stage 2 分数 [E, E]   │
#   extend token     │ Q_Extend @ K_Buffer^T │ Q_Extend @ K_Extend^T │
#                    │                        │                        │
#                    └────────────────────────┴────────────────────────┘
#                                      │
#                          对每个 Query 行做 softmax @ V
#                                      ▼
#                              O_Extend [E, Lv]
#
# 一个 Triton program 只计算：
#   “一个请求 + 一个 query head + BLOCK_M 个 extend query”的输出。
# 它分两个阶段遍历这个请求可见的 KV：
#   stage 1：从 K/V Buffer 读取已经缓存的 prefix KV；
#   stage 2：从 K/V Extend 读取本轮新增 token 的 KV。
# 两阶段共享 (e_max, deno, acc)，通过 online softmax 合并，数学结果等价于
# 对 [prefix KV, extend KV] 一次性做完整 softmax，但无需把完整注意力矩阵写回显存。
#
# MLA 边界说明：BLOCK_DMODEL + BLOCK_DPE 的拆分可以承载 MLA 常见的
# [q_nope/qk_nope, q_rope/k_rope] 维度布局；但本函数不执行 MLA 的低秩投影、
# 矩阵吸收或 latent KV 压缩。它收到的是已经准备好的显式 Q、K、V，主体仍是
# scaled QK -> mask -> softmax -> PV 的通用 MHA/GQA attention kernel。
@triton.jit
def _fwd_kernel(
    # 指向本轮新增 token 的 Q；逻辑形状 [total_extend, q_head_num, Lq]。
    Q_Extend,
    # 指向本轮新增 token 的 K；逻辑形状 [total_extend, kv_head_num, Lq]。
    K_Extend,
    # 指向本轮新增 token 的 V；逻辑形状 [total_extend, kv_head_num, Lv]。
    V_Extend,
    # 输出地址；逻辑形状 [total_extend, q_head_num, Lv]。
    O_Extend,
    # prefix K cache；逻辑形状 [cache_slot_num, kv_head_num, Lq]。
    K_Buffer,
    # prefix V cache；逻辑形状 [cache_slot_num, kv_head_num, Lv]。
    V_Buffer,
    # extend token 的 CSR indptr；qo_indptr[s:s+2] 给出请求 s 的起止位置。
    qo_indptr,
    # prefix 索引的 CSR indptr；kv_indptr[s:s+2] 给出请求 s 的索引区间。
    kv_indptr,
    # 逻辑 prefix token -> K/V Buffer cache slot 的映射表。
    kv_indices,
    # 展平存储的自定义 bool mask；在 EAGLE 验证中通常是树形可见性 mask。
    mask_ptr,
    # 每个请求的 custom mask 在 mask_ptr 中的起始偏移。
    mask_indptr,
    # 可选 attention sink；每个 query head 对应一个 sink logit。
    sink_ptr,
    # 每个请求的 sliding-window KV 列偏移；仅 custom mask + SWA 时读取。
    window_kv_offset_ptr,
    # QK 分数缩放系数，通常是 1 / sqrt(Lq)。
    sm_scale,
    # 每个 KV head 服务的 Q head 数；MHA=1，GQA/MQA>1。
    kv_group_num,
    # Q_Extend 沿 token 维移动一行需要跨过的元素数。
    stride_qbs,
    # Q_Extend 沿 query-head 维移动一格需要跨过的元素数。
    stride_qh,
    # K_Extend 沿 token 维的元素 stride。
    stride_kbs,
    # K_Extend 沿 KV-head 维的元素 stride。
    stride_kh,
    # V_Extend 沿 token 维的元素 stride。
    stride_vbs,
    # V_Extend 沿 KV-head 维的元素 stride。
    stride_vh,
    # O_Extend 沿 token 维的元素 stride。
    stride_obs,
    # O_Extend 沿 query-head 维的元素 stride。
    stride_oh,
    # K_Buffer 沿 cache-slot 维的元素 stride。
    stride_buf_kbs,
    # K_Buffer 沿 KV-head 维的元素 stride。
    stride_buf_kh,
    # V_Buffer 沿 cache-slot 维的元素 stride。
    stride_buf_vbs,
    # V_Buffer 沿 KV-head 维的元素 stride。
    stride_buf_vh,
    # 编译期常量：滑动窗口大小；<=0 表示关闭 SWA。
    SLIDING_WINDOW_SIZE: tl.constexpr,
    # 编译期常量：logits soft-cap 上限；<=0 表示关闭 soft cap。
    logit_cap: tl.constexpr,
    # 编译期常量：xAI 长上下文温度修正阈值；<=0 表示关闭。
    xai_temperature_len: tl.constexpr,
    # 编译期常量：Q/K 最后一维的真实长度。
    Lq: tl.constexpr,
    # 编译期常量：V/O 最后一维的真实长度。
    Lv: tl.constexpr,
    # 编译期 tile：一次主 QK 点积处理的特征维数。
    BLOCK_DMODEL: tl.constexpr,
    # 编译期 tile：主特征块之后单独处理的额外 PE 维数；0 表示没有。
    BLOCK_DPE: tl.constexpr,
    # 编译期 tile：向上填充后的 V/O 特征维数。
    BLOCK_DV: tl.constexpr,
    # 编译期 tile：一个 program 同时处理的 query 行数。
    BLOCK_M: tl.constexpr,
    # 编译期 tile：每轮循环同时处理的 KV 列数。
    BLOCK_N: tl.constexpr,
    # 编译期开关：是否提供了 custom mask。
    USE_CUSTOM_MASK: tl.constexpr,
    # 编译期开关：无 custom mask 时是否应用普通因果下三角 mask。
    IS_CAUSAL: tl.constexpr,
    # 编译期开关：prefix 是否默认全部可见，从而跳过 prefix mask 访存。
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    # 编译期开关：是否用转置 tile 布局执行输出 store。
    STORE_TRANSPOSE: tl.constexpr,
    # 编译期开关：是否提供 attention sink。
    HAS_SINK: tl.constexpr,
):
    # 启动网格是 (batch_size, query_head_num, ceil(max_extend_len/BLOCK_M))。
    # 读取第 0 维 program id：当前处理 batch 中的第几个请求。
    cur_seq = tl.program_id(0)
    # 读取第 1 维 program id：当前处理第几个 query head。
    cur_head = tl.program_id(1)
    # 读取第 2 维 program id：当前处理该请求的第几个 BLOCK_M query 行块。
    cur_block_m = tl.program_id(2)

    # GQA/MQA 映射。例如 32 个 Q head、8 个 KV head 时 kv_group_num=4，
    # Q head 0~3 共用 KV head 0，Q head 4~7 共用 KV head 1，以此类推。
    # 整数除法把当前 query-head id 映射成实际读取的 KV-head id。
    cur_kv_head = cur_head // kv_group_num

    # 从 qo_indptr[s] 读取当前请求在拼接 Q/K/V_Extend 中的首 token 行号。
    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    # qo_indptr[s+1]-qo_indptr[s] 得到当前请求本轮新增的 token 数。
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx

    # 从 kv_indptr[s] 读取当前请求在 kv_indices 中的首元素位置。
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    # kv_indptr[s+1]-kv_indptr[s] 得到该请求已有的 prefix token 数。
    # 这是逻辑长度；各 token 的实际 cache slot 仍需查询 kv_indices。
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx

    # 一个 custom-mask 行覆盖的完整逻辑 KV 长度：prefix 列在前，extend 列在后。
    # 设 P=cur_seq_len_prefix，E=cur_seq_len_extend，则 mask 的二维视图是：
    #
    #                              P 列                         E 列
    #                    ┌────────────────────────┬────────────────────────┐
    #              Q0    │ prefix 可见性          │ extend/树可见性        │
    #              Q1    │                        │                        │
    #   E 个 Query 行    │   mask[:, 0:P]         │   mask[:, P:P+E]       │
    #              ...   │                        │                        │
    #           Q(E-1)   │                        │                        │
    #                    └────────────────────────┴────────────────────────┘
    #                    <--------------- 每行宽度 P+E ------------------>
    #
    # mask_ptr 是上面矩阵的行优先展平存储，所以定位第 r 行时必须跨过
    # r*(P+E) 个元素；cur_seq_len 正是这里使用的行宽 P+E。
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        # 只有启用 custom mask 时才读取；得到当前请求 mask 的展平首地址偏移。
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # 默认没有 sliding-window 引入的 mask 列偏移。
    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        # custom mask 与 SWA 同时启用时，加载当前请求的 KV 列偏移；后续计算
        # mask 行宽和列地址时都要加上它，使地址与调用方准备的 mask 布局一致。
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    # 生成主 Q/K 特征块的局部列号 [0, BLOCK_DMODEL)。
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # 生成 V/O 特征块的局部列号 [0, BLOCK_DV)。
    offs_dv = tl.arange(0, BLOCK_DV)
    # 生成当前 query tile 的局部行号 [0, BLOCK_M)。
    offs_m = tl.arange(0, BLOCK_M)
    # 当前 tile 的全局局部行号 = block 起点 + tile 内行号；小于真实 extend
    # 长度的行才有效。最后一个 BLOCK_M tile 通常需要这个边界 mask。
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    # BLOCK_DMODEL 可能是向上补齐值；只允许读取 Q/K 的真实 Lq 范围。
    mask_d = offs_d < Lq
    # BLOCK_DV 可能是向上补齐到 2 的幂；只允许读取/写入真实 Lv 范围。
    mask_dv = offs_dv < Lv
    # q * k = qk,q shape =[bs, head num, qseqlen, head dim]； k shape =[bs, head num, kv seqlen, head dim]
    # softmax(qk)=P [bs, head num, qseqlen, kvseqlen]
    #P*V=o  v shape [bs, head num, vseqlen, head dim]  o[bs, head num, seqlen, head dim]
    if xai_temperature_len > 0:
        # 把 extend 局部 query 行号换成包含 prefix 的绝对序列位置。
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        # 预先计算 1/log2(阈值)，避免每个 query 行执行一次除法。
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        # 阈值内的行保持倍率 1；阈值外的行使用
        # log2(绝对位置)/log2(阈值) 放大 attention logits。
        xai_temperature_reg = tl.where(
            # tl.where 的条件：该 query 是否已经超过温度修正起点。
            offs_qidx > xai_temperature_len,
            # 条件为真时选择随上下文长度对数增长的倍率。
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            # 条件为假时不改变 logits。
            1.0,
        )

    # 下面通过广播构造 [BLOCK_M, BLOCK_DMODEL] 的 Q 元素偏移矩阵。
    offs_q = (
        # 先把“请求首 token + query block 首行 + tile 内行”变成拼接后的 token id。
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        # token id 乘 token stride，得到每个 query 行的基础地址。
        * stride_qbs # head num, qseqlen, head dim
        # 再移到当前 query head 的起始地址。
        + cur_head * stride_qh
        # 最后加主特征维列号；None 触发二维广播。
        + offs_d[None, :]
    )
    # 从 Q_Extend 取出当前 query tile；无效 query 行和补齐特征列填 0。
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        # 生成额外 PE 维的真实列号。例如 Lq=576 时，这里是 [512,576)。
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        # 与 offs_q 相同地构造 [BLOCK_M, BLOCK_DPE] 地址，只是特征列换成 dpe。
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        # 加载额外 PE query；这里只需屏蔽越界 query 行，因为这些特殊维度组合
        # 保证 BLOCK_DMODEL+BLOCK_DPE 恰好等于 Lq。
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # ------------------------------------------------------------------
    # stage 1：Q_Extend × K_Buffer(prefix)，读取历史 prefix KV
    # ------------------------------------------------------------------
    # 当前 program 每轮实际处理的矩阵 tile（M=BLOCK_M，N=BLOCK_N，D=主特征维）：
    #
    #          q [M,D]                    k [D,N]                   qk [M,N]
    #   ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
    #   │ q00 ... q0(D-1)  │       │ k00 ... k0(N-1)  │       │ s00 ... s0(N-1)  │
    #   │ q10 ... q1(D-1)  │   @   │  :         :     │   =   │ s10 ... s1(N-1)  │
    #   │  :         :     │       │ k(D-1) ...       │       │  :         :     │
    #   │ q(M-1) ...       │       └──────────────────┘       │ s(M-1) ...       │
    #   └──────────────────┘                                  └──────────────────┘
    #
    # prefix 的 K/V 可能散落在 cache 中；矩阵形状仍是 [D,N]/[N,DV]，只是每个
    # N 列对应的物理 slot 需要先通过 kv_indices 间接寻址。
    # 生成一个 KV tile 内的局部列号 [0, BLOCK_N)。
    offs_n = tl.arange(0, BLOCK_N)

    # Online softmax 不保存完整 qk，而只为 M 个 Query 行维护以下状态：
    #
    #        e_max [M]          deno [M]                  acc [M,DV]
    #        ┌───────┐          ┌───────┐          ┌────────────────────┐
    #   Q0   │ max_0 │          │ sum_0 │          │ acc_00 ... acc_0d │
    #   Q1   │ max_1 │          │ sum_1 │          │ acc_10 ... acc_1d │
    #   ...  │  ...  │          │  ...  │          │  ...         ...  │
    # Q(M-1) │ max_m │          │ sum_m │          │ acc_m0 ... acc_md │
    #        └───────┘          └───────┘          └────────────────────┘
    #
    # 初始化 online-softmax 输出分子；每一行对应一个 query，每一列对应一个 V 维度。
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    # 初始化 online-softmax 分母；每个 query 行独立维护一个标量。
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    # 初始化每个 query 行已经遍历过的最大 logit；尚未看到 KV，所以是 -inf。
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # start_n 是当前 prefix KV tile 的逻辑首列；每轮向右移动 BLOCK_N 列。
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        # 向编译器声明 start_n 是 BLOCK_N 的整数倍，帮助生成对齐访存代码。
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # 当前 KV tile 的逻辑列号不能超过真实 prefix 长度。
        mask_n = (start_n + offs_n) < cur_seq_len_prefix

        # 外积得到二维边界 mask：[BLOCK_M query 行, BLOCK_N KV 列]。
        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            # 当前 stage 从完整 mask 的左半区取一个 [M,N] prefix tile：
            #
            #              prefix 区 [0,P)                        extend 区 [P,P+E)
            #   ┌───────────────┬──────────────────┬──────────┬────────────────────┐
            #   │ 已处理的列    │ 本轮 [M,N] tile │ 后续列   │ stage 2 才会读取   │
            #   │ [0,start_n)   │                │          │                    │
            #   └───────────────┴──────────────────┴──────────┴────────────────────┘
            #                    ▲
            #                    start_n + offs_n
            #
            # 读取当前二维 tile 对应的 custom mask：
            #   请求 mask 首址
            # + query 局部行号 * mask 行宽
            # + sliding-window 列偏移
            # + 当前 prefix tile 首列
            # + tile 内 KV 列号。
            custom_mask = tl.load(
                # 当前请求 custom mask 的全局首地址。
                mask_ptr
                + cur_seq_mask_start_idx
                # 移动到当前 BLOCK_M 个 query 分别对应的 mask 行。
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                # 移动到 SWA mask 中实际保存的第一列。
                + window_kv_offset
                # 移动到本轮 prefix KV tile 的第一列。
                + start_n
                # 加上 tile 内的 KV 列号，形成 [BLOCK_M,BLOCK_N] 地址。
                + offs_n[None, :],
                # 只读取没有越过 query/KV 边界的 mask 元素。
                mask=(mask_m[:, None] & mask_n[None, :]),
                # 越界位置当成 False，即不可见。
                other=0,
            )
            # 一个位置必须同时满足边界 mask 和调用方提供的 custom mask。
            final_mask &= custom_mask
        if SLIDING_WINDOW_SIZE > 0:
            # 计算每个 query 行在完整 [prefix,extend] 序列中的绝对位置。
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            # 只保留 q_id-kv_id <= window_size 的 KV；过旧 prefix 被屏蔽。
            ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            # custom/边界/滑动窗口条件必须同时成立。
            final_mask &= window_mask

        # 默认认为当前 tile 需要计算。
        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            # 先沿 KV 列、再沿 query 行做 max；结果为 0 表示整块没有任何 True。
            # 整块不可见时跳过 K/V 访存和矩阵乘，节省计算。
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # 用当前请求在 kv_indices 中的首位置 + tile 首列 + tile 内列号，
            # 找到这 BLOCK_N 个逻辑 prefix token 对应的索引表元素。
            #
            #   prefix 逻辑位置       kv_indices 查表          Buffer 物理槽位
            #   ┌──────────────┐     ┌──────────────┐         ┌──────────────┐
            #   │ n0 n1 ... nN │ ──> │ s0 s1 ... sN │ ──────> │ slot s0 ...  │
            #   └──────────────┘     └──────────────┘         └──────────────┘
            #
            # 中间的 s0/s1/... 可以不连续，因此不能直接用逻辑 n 访问 K/V Buffer。
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                # 最后一个 KV tile 可能不满，越过 prefix 尾部的索引不能读取。
                mask=mask_n,
                # 无效索引填 0；后续 K/V load 还会被 mask_n 屏蔽，不会参与结果。
                other=0,
            )

            # 构造 K_Buffer 的 [BLOCK_DMODEL, BLOCK_N] 地址矩阵。
            offs_buf_k = (
                # 每个实际 cache slot 定位一列 K token。
                offs_kv_loc[None, :] * stride_buf_kbs
                # 定位 GQA/MQA 映射后的当前 KV head。
                + cur_kv_head * stride_buf_kh
                # 加上 K 的主特征维行号；形成转置读取布局 [D,N]。
                + offs_d[:, None]
            )
            # 加载 prefix K tile；KV 尾列及补齐的特征维都以 0 填充。
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )

            # [M,D]@[D,N] 得到当前 query tile 对 prefix tile 的原始 QK 分数 [M,N]。
            # q 转为 k.dtype，使 tl.dot 使用输入张量的数据类型执行高效矩阵乘。
            qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                # 构造 K_Buffer 中额外 PE 部分的 [BLOCK_DPE,BLOCK_N] 地址。
                offs_kpe = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                # 加载 prefix K 的额外 PE 特征；只需屏蔽越界 KV 列。
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                # 把 PE 子空间的点积加入主 QK 分数：qk=q·k+qpe·kpe。
                qk += tl.dot(qpe.to(kpe.dtype), kpe)

            # 执行标准 scaled dot-product attention 的缩放。
            qk *= sm_scale

            if logit_cap > 0:
                # 把极端 logits 平滑压缩进 (-logit_cap,logit_cap)。
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                # 每个 query 行乘自己的长上下文温度倍率；[:,None] 广播到所有 KV 列。
                qk *= xai_temperature_reg[:, None]

            # 不可见/越界位置在 softmax 前置为 -inf，exp 后权重恰好为 0。
            qk = tl.where(final_mask, qk, float("-inf"))

            # 沿 KV 列取 max，得到当前 tile 每个 query 行的最大 logit。
            row_max = tl.max(qk, 1)
            # 若某行在此 tile 全不可见，row_max=-inf；换成有限负数防止 inf-inf=NaN。
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)

            # 一个 tile 的 online-softmax 合并可看成下面的状态变换；所有量都逐行独立：
            #
            #   ┌────────────────────── 旧状态 ──────────────────────┐
            #   │ e_max_old [M] │ deno_old [M] │ acc_old [M,DV]     │
            #   └────────────────────────────────────────────────────┘
            #                              +
            #   ┌──────────────────── 当前 KV tile ──────────────────┐
            #   │ qk [M,N]                         │ V [N,DV]        │
            #   └────────────────────────────────────────────────────┘
            #                              │
            #                              ▼
            #   ┌────────────────────── 新状态 ──────────────────────┐
            #   │ m_new = max(e_max_old, row_max(qk))                │
            #   │ r     = exp(e_max_old - m_new)                     │
            #   │ deno  = deno_old*r + sum(exp(qk-m_new), axis=N)    │
            #   │ acc   = acc_old*r  + exp(qk-m_new) @ V             │
            #   └────────────────────────────────────────────────────┘
            #
            # 新运行最大值取“历史最大值”和“当前 tile 最大值”的逐行较大者。
            n_e_max = tl.maximum(row_max_fixed, e_max)

            # 把历史 acc/deno 从 exp(score-e_max) 基准换到 exp(score-n_e_max) 基准。
            re_scale = tl.exp(e_max - n_e_max)
            # 计算当前 tile 在新基准下的未归一化 softmax 权重。
            p = tl.exp(qk - n_e_max[:, None])
            # 重标定旧分母后，加上当前 tile 每行的权重和。
            deno = deno * re_scale + tl.sum(p, 1)

            # 构造 V_Buffer 的 [BLOCK_N,BLOCK_DV] 地址矩阵。
            offs_buf_v = (
                # 由实际 cache slot 定位每个 prefix V token 的首地址。
                offs_kv_loc[:, None] * stride_buf_vbs
                # 定位当前 KV head。
                + cur_kv_head * stride_buf_vh
                # 加上 value 特征列号，得到 [N,DV] 地址。
                + offs_dv[None, :]
            )
            # 加载 prefix V；越界 KV 行或补齐 value 维以 0 填充。
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            # 将权重转成 V 的 dtype，让后续 tl.dot 使用匹配的低精度输入类型。
            p = p.to(v.dtype)
            # 重标定历史分子，再累加当前 tile 的 P@V：[M,N]@[N,DV] -> [M,DV]。
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            # 保存新的运行最大值，供下一个 prefix tile 或 stage 2 使用。
            e_max = n_e_max

    # ------------------------------------------------------------------
    # stage 2：Q_Extend × K_Extend，读取本轮新增 token 的 KV
    # ------------------------------------------------------------------
    # 与 stage 1 的矩阵乘形状相同，但 K/V 来源和 mask 区域不同：
    #
    #         q [M,D]             k_extend [D,N]            qk_extend [M,N]
    #   ┌────────────────┐      ┌────────────────┐        ┌────────────────┐
    #   │                │  @   │                │   =    │                │
    #   │ extend Query   │      │ extend K tile  │        │ extend scores  │
    #   │                │      │                │        │                │
    #   └────────────────┘      └────────────────┘        └────────────────┘
    #
    # K_Extend/V_Extend 按 token 连续存放，因此这里不再经过 kv_indices。
    # 先计算 stage 2 需要扫描到的 extend KV 右边界。
    cur_block_m_end = (
        # 非因果模式：当前 query tile 可能看到本请求的全部 extend K。
        cur_seq_len_extend
        if not IS_CAUSAL
        # 因果模式：最多扫描到当前 query block 末尾；更右侧的 K 必然不可见。
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    # 以 BLOCK_N 为步长遍历 [0,cur_block_m_end) 内的 extend KV。
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        # 告诉编译器当前 extend KV tile 起点按 BLOCK_N 对齐。
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # 屏蔽当前扫描右边界之外的 KV 列。
        mask_n = (start_n + offs_n) < cur_block_m_end

        # 先建立 query 行、extend KV 列都合法的二维边界 mask。
        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            # 当前 stage 从完整 mask 的右半区取 [M,N] extend tile：
            #
            #               prefix 区 [0,P)                     extend 区 [P,P+E)
            #   ┌──────────────────────────────┬──────────────────────────────────┐
            #   │ stage 1 读取                 │ stage 2：树形可见性              │
            #   │                              │  ┌────────────────────────────┐  │
            #   │                              │  │ 当前 start_n 的 [M,N] tile │  │
            #   │                              │  └────────────────────────────┘  │
            #   └──────────────────────────────┴──────────────────────────────────┘
            #                                  ▲
            #                        列地址先加 cur_seq_len_prefix
            #
            # custom mask 的地址仍然按“请求首址 + query 行 + KV 列”计算，
            # 但 extend KV 列位于 prefix 列之后，所以列地址额外加 prefix_len。
            custom_mask = tl.load(
                # 当前请求 custom mask 的首地址。
                mask_ptr
                + cur_seq_mask_start_idx
                # 移动到当前 query 所在 mask 行；行宽覆盖 prefix+extend。
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                # 加上 SWA mask 的起始列偏移。
                + window_kv_offset
                # 跳过 mask 行最前面的全部 prefix 列。
                + cur_seq_len_prefix
                # 移动到当前 extend KV tile 的第一列。
                + start_n
                # 加 tile 内列号，形成 [BLOCK_M,BLOCK_N] mask 地址。
                + offs_n[None, :],
                # query 行或 KV 列越界时禁止读取。
                mask=(mask_m[:, None] & mask_n[None, :]),
                # 越界位置按 False 处理。
                other=0,
            )
            # 再次并入行列边界，保证 custom mask 的 padding 不可能变成有效位置。
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            # 树验证时，这一步只保留“自身及祖先”，屏蔽兄弟/其他分支。
            # custom mask 已完整表达可见性，因此该分支不会再叠加普通 causal mask。
            # 以抽象树 root -> {child_0, child_1} 为例，右半区线框矩阵是：
            #
            #                         KV：root   child_0   child_1
            #                              ┌────────┬────────┬────────┐
            #   Query：root                │   1    │   0    │   0    │
            #          child_0             ├────────┼────────┼────────┤
            #                              │   1    │   1    │   0    │
            #          child_1             ├────────┼────────┼────────┤
            #                              │   1    │   0    │   1    │
            #                              └────────┴────────┴────────┘
            # 每个孩子能看 root 和自己，但两个兄弟不能互看。
            final_mask &= custom_mask
        elif IS_CAUSAL:
            # 普通链式因果 mask 是下三角矩阵（1=可见，0=未来 token）：
            #
            #                         KV 局部列
            #                         0    1    2    3
            #                      ┌────┬────┬────┬────┐
            #   Query 局部行  0    │ 1  │ 0  │ 0  │ 0  │
            #                 1    ├────┼────┼────┼────┤
            #                      │ 1  │ 1  │ 0  │ 0  │
            #                 2    ├────┼────┼────┼────┤
            #                      │ 1  │ 1  │ 1  │ 0  │
            #                 3    ├────┼────┼────┼────┤
            #                      │ 1  │ 1  │ 1  │ 1  │
            #                      └────┴────┴────┴────┘
            #
            # 计算普通下三角条件：query 的 extend 局部行号 >= KV 的 extend 局部列号。
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            # 因果条件还必须与真实 query/KV 边界同时成立。
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            # 把因果下三角条件合入最终 mask。
            final_mask &= mask_causual
        else:
            # 无 custom 且非因果时，唯一限制就是 query/KV 不能越界。
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            # 合入非因果边界 mask；此处与初始 final_mask 等价，保留结构统一。
            final_mask &= mask_non_causal

        if SLIDING_WINDOW_SIZE > 0:
            # extend 区内两边都用局部下标，计算 q_local-kv_local <= window_size。
            window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            # 在已有边界/树形/因果可见性上再限制滑动窗口。
            final_mask &= window_mask

        # 默认计算当前 extend KV tile。
        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            # 整个 tile 没有任何可见元素时，跳过后续 K/V 加载和矩阵乘。
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # 构造 K_Extend 的 [BLOCK_DMODEL,BLOCK_N] 地址矩阵。
            offs_k = (
                # 当前请求首 token + tile 首列 + tile 内列号，得到拼接 token id。
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                # 定位当前 GQA/MQA KV head。
                + cur_kv_head * stride_kh
                # 加主特征维行号，以 [D,N] 转置布局读取。
                + offs_d[:, None]
            )
            # 加载当前 extend K tile；越界 KV 列及补齐特征维以 0 填充。
            k = tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
            )

            # 计算 extend 部分的主特征 QK 分数 [M,D]@[D,N] -> [M,N]，
            # 并指定累加/输出为 float32 以提高数值稳定性。
            qk = tl.dot(q, k, out_dtype=tl.float32)
            if BLOCK_DPE > 0:
                # 构造 K_Extend 额外 PE 维的 [BLOCK_DPE,BLOCK_N] 地址。
                offs_kpe = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[:, None]
                )
                # 加载 extend K 的额外 PE 特征；越过扫描右边界的列以 0 填充。
                kpe = tl.load(
                    K_Extend + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                # 累加额外 PE 子空间的 QK 点积。
                qk += tl.dot(qpe, kpe)

            # 应用 scaled dot-product attention 缩放。
            qk *= sm_scale

            if logit_cap > 0:
                # 可选 soft cap，限制过大的正负 logits。
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                # 对每个 query 行应用它自己的长上下文温度倍率。
                qk *= xai_temperature_reg[:, None]

            # 所有不可见或越界的位置在 softmax 前变成 -inf。
            qk = tl.where(final_mask, qk, float("-inf"))

            # 当前 extend tile 中，每个 query 行沿 KV 列取最大 logit。
            row_max = tl.max(qk, 1)
            # 全 mask 行用有限负数替代 -inf，避免后续 -inf-(-inf) 产生 NaN。
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            # 与 stage 1/此前 extend tiles 的运行最大值合并。
            n_e_max = tl.maximum(row_max_fixed, e_max)

            # 把历史 online-softmax 状态换算到新的指数基准。
            re_scale = tl.exp(e_max - n_e_max)
            # 计算当前 extend KV tile 的未归一化权重。
            p = tl.exp(qk - n_e_max[:, None])
            # 重标定历史分母并累加当前 tile 权重和。
            deno = deno * re_scale + tl.sum(p, 1)

            # 构造 V_Extend 的 [BLOCK_N,BLOCK_DV] 地址矩阵。
            offs_v = (
                # 由拼接 token id 定位每个 extend V token。
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                # 定位当前 KV head。
                + cur_kv_head * stride_vh
                # 加 value 特征列号。
                + offs_dv[None, :]
            )
            # 加载 extend V；越界 KV 行和补齐 value 维以 0 填充。
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            # 将 float32 指数权重转换为 V dtype，以供高效 tl.dot 使用。
            p = p.to(v.dtype)
            # 重标定历史输出分子，并累加本 tile 的 P@V。
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            # 将合并后的最大值留给下一个 extend KV tile。
            e_max = n_e_max

    if HAS_SINK:
        # 读取当前 query head 的 sink logit；同一 head 的所有 query 行共享该值。
        cur_sink = tl.load(sink_ptr + cur_head)
        # sink 只有 softmax 概率质量、没有对应 V，故只更新分母不更新 acc；
        # 数学上等价于额外加入一个 value=0、logit=cur_sink 的位置。
        deno += tl.exp(cur_sink - e_max)

    # 所有 prefix/extend KV tiles 处理完后，最后一个逐行归一化的线框关系为：
    #
    #           acc [M,DV]                 deno[:,None] [M,1]
    #   ┌────────────────────────┐        ┌───────────────┐
    #   │ acc_00 ... acc_0d     │        │ deno_0        │
    #   │ acc_10 ... acc_1d     │   ÷    │ deno_1        │  （按行广播）
    #   │  ...          ...      │        │  ...          │
    #   │ acc_m0 ... acc_md     │        │ deno_m        │
    #   └────────────────────────┘        └───────────────┘
    #                    │
    #                    ▼
    #           ┌────────────────────────┐
    #           │ O_Extend tile [M,DV]   │
    #           └────────────────────────┘
    #
    # 构造 O_Extend 的 [BLOCK_M,BLOCK_DV] 输出地址矩阵。
    offs_o = (
        # 由当前请求首 token、query block 起点和 tile 内行号定位输出 token 行。
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        # 定位当前 query head；输出 head 数跟 Q head 数一致。
        + cur_head * stride_oh
        # 加 value/output 特征列号；输出最后一维是真实 Lv，而不是 Lq。
        + offs_dv[None, :]
    )
    if STORE_TRANSPOSE:
        # 先做逐 query 行归一化 O=acc/deno，再同时转置值、地址和 mask 写回。
        # 三者一起转置不会改变逻辑输出地址，只改变 Triton 使用的 tile 布局。
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            # query 尾行和补齐的 output 维都禁止写回。
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        # 当前封装走此分支：逐行用 softmax 分母归一化输出分子并写回。
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            # 仅写真实 query 行和真实 Lv 特征，防止最后一个 tile 越界。
            mask=mask_m[:, None] & mask_dv[None, :],
        )


def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    # Get block sizes and configuration
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    HAS_SINK = sinks is not None

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        window_kv_offsets,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        HAS_SINK=HAS_SINK,
        STORE_TRANSPOSE=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
