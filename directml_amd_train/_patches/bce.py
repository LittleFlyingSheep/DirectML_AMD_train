# -*- coding: utf-8 -*-
"""补丁 9：绕开 DirectML 上 BCEWithLogits 的 CPU fallback。

BCEWithLogits 内部用 aten::log_sigmoid_forward，DirectML 不支持，
torch-directml 会把它 fallback 到 CPU 上算。前向看着没问题（数值对），
但反向时梯度未必能原样搬回显卡。

这里自己实现一份：前向在 CPU 上算（数值可靠），反向用
d/dz BCE(z, t) = sigmoid(z) - t 在 CPU 上算完再搬回显卡。
"""
from __future__ import annotations


def patch_bce() -> None:
    import torch
    import torch.nn.functional as F

    _orig_forward = torch.nn.BCEWithLogitsLoss.forward

    class _BCEOnCpu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, target):
            ctx.save_for_backward(inp, target)
            ctx.dev = inp.device
            out = F.binary_cross_entropy_with_logits(inp.cpu(), target.cpu(), reduction="none")
            return out.to(inp.device)

        @staticmethod
        def backward(ctx, grad_out):
            inp, target = ctx.saved_tensors
            g = (torch.sigmoid(inp.cpu()) - target.cpu()) * grad_out.cpu()
            return g.to(ctx.dev), None

    def forward(self, input, target):
        if (isinstance(input, torch.Tensor) and input.device.type == "privateuseone"
                and self.weight is None and self.pos_weight is None):
            out = _BCEOnCpu.apply(input, target)
            if self.reduction == "mean":
                return out.mean()
            if self.reduction == "sum":
                return out.sum()
            return out
        return _orig_forward(self, input, target)

    torch.nn.BCEWithLogitsLoss.forward = forward
