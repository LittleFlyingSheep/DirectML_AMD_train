# -*- coding: utf-8 -*-
"""补丁 10：避开 DirectML 上 addmv 的 CPU fallback（DFL 损失用它）。

aten::addmv.out 在 DirectML 上不支持，会 fallback 到 CPU；
改成纯 elementwise 的写法：sum(mat * vec, dim=1)，语义等价。
"""
from __future__ import annotations


def patch_addmv() -> None:
    import torch

    _orig = torch.addmv

    def addmv(input, mat, vec, beta=1, alpha=1):
        if (isinstance(input, torch.Tensor) and input.device.type == "privateuseone"
                and mat.dim() == 2 and vec.dim() == 1):
            out = (mat * vec.reshape(1, -1)).sum(dim=1)
            return alpha * out + beta * input
        return _orig(input, mat, vec, beta=beta, alpha=alpha)

    torch.addmv = addmv
