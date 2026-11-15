# -*- coding: utf-8 -*-
"""补丁 4：绕开 DirectML 的 scatter_add_ 缺口。

ultralytics 拼训练标签时用 offsets.scatter_add_，
DirectML 不支持这个原地算子。这里改成「回 CPU 算，再写回去」。
"""
from __future__ import annotations


def patch_scatter() -> None:
    import torch

    _orig = torch.Tensor.scatter_add_

    def scatter_add_(self, dim, index, src, reduce=None):
        if self.device.type != "privateuseone":
            return _orig(self, dim, index, src, reduce) if reduce is not None \
                else _orig(self, dim, index, src)
        out = self.cpu().scatter_add_(dim, index.cpu(), src.cpu())
        self.copy_(out.to(self.device))
        return self

    torch.Tensor.scatter_add_ = scatter_add_
