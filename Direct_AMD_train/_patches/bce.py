# -*- coding: utf-8 -*-
"""补丁 9：BCEWithLogitsLoss 改成在显卡上算（不再回落到 CPU）。

PyTorch 的 BCEWithLogitsLoss 内部走 aten::log_sigmoid_forward，
DirectML 不支持这个算子，torch-directml 会把它 fallback 到 CPU：

    UserWarning: The operator 'aten::log_sigmoid_forward' is not currently
    supported on the DML backend and will fall back to run on the CPU.

分类损失是 (bs, num_anchors, nc) 大小的张量（640 输入下就是几百万个元素），
每个 iteration 都要在 CPU 和显卡之间来回搬两趟，既慢又吃内存带宽。

这里自己实现一份，公式与 PyTorch 的完全等价：

    loss = softplus(z) - z * t
    softplus(z) = max(z, 0) + log1p(exp(-|z|))       （数值稳定）

反向：

    d/dz = sigmoid(z) - t

用到的算子（sigmoid / exp / log1p / abs / clamp / mul / sub）在 DirectML 上
都有原生实现，一个都不会回落 CPU。
"""
from __future__ import annotations


def patch_bce() -> None:
    import torch
    import torch.nn.functional as F

    from ._once import already_patched, mark

    if already_patched(torch.nn.BCEWithLogitsLoss.forward):
        return

    _orig_forward = torch.nn.BCEWithLogitsLoss.forward

    class _BCEWithLogitsOnGpu(torch.autograd.Function):
        """BCEWithLogitsLoss 的显卡实现（公式见模块 docstring）。"""

        @staticmethod
        def forward(ctx, inp, target):
            ctx.save_for_backward(inp, target)
            return F.softplus(inp) - inp * target

        @staticmethod
        def backward(ctx, grad_out):
            inp, target = ctx.saved_tensors
            return (torch.sigmoid(inp) - target) * grad_out, None

    def forward(self, input, target):
        if (torch.is_tensor(input) and input.device.type == "privateuseone"
                and self.weight is None and self.pos_weight is None):
            out = _BCEWithLogitsOnGpu.apply(input, target.to(input.dtype))
            if self.reduction == "mean":
                return out.mean()
            if self.reduction == "sum":
                return out.sum()
            return out
        return _orig_forward(self, input, target)

    torch.nn.BCEWithLogitsLoss.forward = mark(forward)
