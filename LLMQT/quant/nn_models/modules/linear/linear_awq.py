import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from quant.utils.common_utils import get_best_device
from quant.utils.packing_utils import dequantize_gemm
from .linear_base import LinearBase


class AWQLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
    ):
        # 这个 autograd.Function 只实现了 forward，没有实现 backward。
        # 当前 AWQLinear_GEMM 主要用于推理，并且外层调用时也包了 torch.no_grad()。
        #
        # 参数含义：
        #   x: 当前 Linear 的输入 activation，最后一维必须是 in_features。
        #      常见 shape：
        #        prefill: [batch, seq_len, in_features]
        #        decode 或某些测试: [tokens, in_features]
        #   qweight: pack 后的 int4 权重，shape [in_features, out_features / 8]。
        #   qzeros:  pack 后的 int4 zero point，shape [in_features / group_size, out_features / 8]。
        #   scales:  fp16 scale，shape [in_features / group_size, out_features]。
        #   bias:    原始 Linear 的 bias，shape [out_features]，没有 bias 时为 None。
        #
        # ctx 里保存张量是 autograd.Function 的常规写法；这里虽然不做 backward，
        # 但保留这些信息方便以后扩展或调试。
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        # 输出 shape 只替换最后一维：
        #   x[..., in_features] -> out[..., out_features]
        # 例如：
        #   [batch, seq_len, 5120] -> [batch, seq_len, 5120]  (q_proj/o_proj)
        #   [batch, seq_len, 5120] -> [batch, seq_len, 13824] (up_proj/gate_proj)
        #   [batch, seq_len, 13824] -> [batch, seq_len, 5120] (down_proj)
        out_shape = x.shape[:-1] + (out_features,)

        # 当前 dequantize_gemm/matmul 路径使用 fp16 权重，所以这里把输入转成 fp16。
        # 外层 AWQLinear_GEMM.forward() 会在最后按需要转回原始 input_dtype。
        x = x.to(torch.float16)

        # 空 batch/token 的保护分支，避免后面的 matmul 在空输入上产生异常。
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # dequantize_gemm 做三件事：
        #   1. unpack qweight/qzeros: int32 -> 8 个 int4
        #   2. reverse AWQ order: 恢复成正常输出通道顺序
        #   3. 反量化: fp_weight = (int_weight - zero) * scale
        #
        # 返回的 out 实际是反量化后的浮点权重矩阵，shape 为：
        #   [in_features, out_features]
        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)

        # Linear 的数学形式：
        #   y = x @ W
        # 这里 W 已经是 [in_features, out_features]，所以可以直接 matmul。
        # x 的最后一维是 in_features，结果最后一维变成 out_features。
        out = torch.matmul(x, out)

        # bias 按最后一维 out_features 广播相加。
        out = out + bias if bias is not None else out

        # matmul 会保留 x 的前置维度；这里显式 reshape 成前面算好的目标形状。
        out = out.reshape(out_shape)

        # 这个实现历史上希望返回至少 3D。
        # 如果输入是 [tokens, in_features]，matmul 后是 [tokens, out_features]，
        # 这里会变成 [1, tokens, out_features]。
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out


# 每个量化方法都要像AWQLinear这样实现init，from linear和forward方法
class AWQLinear_GEMM(LinearBase):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev
    ):
        super(LinearBase, self).__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        # 健壮性检查
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight", # [5120, 640], 列上每8个int4 pack为int
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",# [40, 640], 行上group size为128，列上每8个int4 pack为int
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",# [40, 5120] 行上group size为128
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",# [5120]
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):

        # from_linear 的作用：
        #   把一个已经完成 AWQ scale/clip 的原始 nn.Linear，转换成真正推理用的
        #   AWQLinear_GEMM。原始 linear.weight 是浮点权重，shape 为
        #   [out_features, in_features]；转换后保存的是 pack 过的 int4 权重。
        #
        # 参数来源：
        #   linear.in_features / linear.out_features 来自原始 nn.Linear。
        #   scales / zeros 由 quantizer.pseudo_quantize_tensor() 计算得到，
        #   原始 shape 是 [out_features, in_features / group_size]，在
        #   quantizer._apply_quant() 里转置后传进来，所以这里期望它们的 shape 是
        #   [in_features / group_size, out_features]。
        #
        # 主要流程：
        #   1. 根据原始 Linear 的尺寸创建 AWQLinear_GEMM 的 buffer。
        #   2. 保存 scales 和 bias。
        #   3. 按输入通道分组，把 fp16/bf16 weight 量化成 int4 数值。
        #   4. 每 8 个 int4 打包成 1 个 int32，得到 qweight。
        #   5. zeros 也按相同方式打包，得到 qzeros。
        #   6. 返回可替换原始 nn.Linear 的 awq_linear。
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        # 1. Quantize weight to int4.
        # 非对称量化公式：
        #   qx = round(x / scale + zero)
        # 等价写法：
        #   qx = round((x + scale * zero) / scale)
        #
        # 这里 zeros/scales 的 shape 都是 [n_groups, out_features]。
        # 以 Qwen2.5-14B 的 q_proj 为例：
        #   in_features=5120, out_features=5120, group_size=128
        #   n_groups = 5120 / 128 = 40
        #   scales.shape = zeros.shape = [40, 5120]
        scale_zeros = zeros * scales
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        # 4bit 时 pack_num=8，表示 8 个 int4 可以塞进 1 个 int32。
        pack_num = 32 // awq_linear.w_bit

        intweight = []
        # 真正量化
        
        for idx in range(awq_linear.in_features):
            # idx 遍历输入通道。
            # linear.weight.data[:, idx] 取的是某一个输入通道连接到所有输出通道的权重，
            # shape 为 [out_features]。
            #
            # idx // group_size 定位这个输入通道属于哪个 group。比如 group_size=128：
            #   idx=0..127   -> group 0
            #   idx=128..255 -> group 1
            #
            # scale_zeros[idx // group_size] 和 awq_linear.scales[idx // group_size]
            # 的 shape 都是 [out_features]，正好和当前这一列 weight 对齐。
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )

        # cat 后恢复成普通权重矩阵布局：[out_features, in_features]。
        intweight = torch.cat(intweight, dim=1)

        # GEMM kernel 这里采用 [in_features, out_features] 的存储布局，
        # 后面会沿 out_features 维度每 8 个 int4 打包成 1 个 int32。
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)

        # 2. Pack weight: 8 个 int4 -> 1 个 int32。
        # qweight.shape = [in_features, out_features / 8]。
        # q_proj 示例：[5120, 5120 / 8] = [5120, 640]。
        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device,
        )

        # 将 intweight pack 为 qweight。每个 col 对应一个 int32，
        # 里面按 4bit slot 保存 8 个输出通道的权重。
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                # AWQ/GEMM kernel 期望的 int4 排列顺序，不是简单的 0,1,2,...,7。
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                # qweight_col 逻辑上是 int4，实际 dtype 是 int32。
                # 通过左移 i * 4 bit，放进当前 int32 的第 i 个 4bit 槽位。
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        # 0000 0000 0000 0000  0000 0000 0000 0000
        # 0000 0000 0000 0000  0000 0000 0000 0001 i=0--->0000 0000 0000 0000  0000 0000 0000 0001
        # 0000 0000 0000 0000  0000 0000 0100<-0000(0100) i=1--->0000 0000 0000 0000  0000 0000 0100 0001

        awq_linear.qweight = qweight
        zeros = zeros.to(dtype=torch.int32)

        # 3. Pack zeros: 8 个 int4 zero point -> 1 个 int32。
        # zeros.shape  = [n_groups, out_features]
        # qzeros.shape = [n_groups, out_features / 8]
        # q_proj 示例：[40, 5120] -> [40, 640]。
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        return awq_linear

    def forward(self, x):
        # AWQLinear_GEMM 替换原始 nn.Linear 后，forward 入口仍然保持 Linear 风格：
        #   输入 x 的最后一维是 self.in_features
        #   输出 out 的最后一维是 self.out_features
        #
        # 例子：
        #   q_proj:    x[..., 5120]  -> out[..., 5120]
        #   k/v_proj:  x[..., 5120]  -> out[..., 1024]
        #   up/gate:   x[..., 5120]  -> out[..., 13824]
        #   down_proj: x[..., 13824] -> out[..., 5120]

        # 记录输入 dtype。Qwen 这类模型常见是 bf16，而当前反量化/matmul 路径内部会转 fp16。
        # forward 结束前会把输出转回 input_dtype，尽量保持和原始模型 dtype 行为一致。
        input_dtype = x.dtype

        # 早期 Triton kernel 常要求 fp16 输入；当前 LLMQT 里的 AWQ GEMM 路径不是 Triton kernel，
        # 真正的 fp16 转换放在 AWQLinearMMFunction.forward() 内部做。
        # if input_dtype != torch.float16:
        #     x = x.half()

        with torch.no_grad():
            # 传入的 qweight/qzeros 都是 from_linear() 里 pack 好的 int32 buffer。
            # AWQLinearMMFunction 会先反量化出浮点权重，再执行 x @ weight。
            out = AWQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )

        # 如果原始输入是 bf16，就把输出转回 bf16。
        # 这样替换前后的 Linear 对外 dtype 更一致。
        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        return out
