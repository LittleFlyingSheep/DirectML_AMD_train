# -*- coding: utf-8 -*-
"""补丁 4：绕开 DirectML 的 scatter_add_ 缺口。

ultralytics 拼训练标签时用 offsets.scatter_add_，
DirectML 不支持这个原地算子，而且它的原生实现是错的（不累加）。

实测（torch-directml 0.2.5.dev240914）：

    base = torch.zeros(4); idx = torch.tensor([0,0,2]); src = torch.ones(3)
    base.scatter_add_(0, idx, src)
    CPU  -> [2, 0, 1, 0]   正确
    DML  -> [1, 0, 1, 0]   没累加，错的

修法分两种：

1) dim=0 且 index 是一维 —— 这正是 ultralytics 的用法，
   可以用 index_put_(accumulate=True) 等价实现，而且 DirectML 的
   index_put_ 是对的，所以**整段留在显卡上**，不用来回搬。

       scatter_add_(0, index, src)  ==  index_put_((index,), src, accumulate=True)

2) 其它情况（dim != 0、多维 index）语义对不上 index_put_，
   老老实实回 CPU 算再写回去。
"""
from __future__ import annotations


def patch_scatter() -> None:
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.Tensor.scatter_add_):
        return

    _orig = torch.Tensor.scatter_add_

    def scatter_add_(self, dim, index, src, reduce=None):
        if self.device.type != "privateuseone":
            return _orig(self, dim, index, src, reduce) if reduce is not None \
                else _orig(self, dim, index, src)
        if dim == 0 and index.dim() == 1 and src.dim() >= 1 and src.shape[0] == index.shape[0]:
            # index_put_(accumulate=True) 在 DirectML 上是对的，而且不用搬数据
            try:
                self.index_put_((index,), src, accumulate=True)
                return self
            except Exception:
                pass  # 退回 CPU 路径
        out = self.cpu().scatter_add_(dim, index.cpu(), src.cpu())
        self.copy_(out.to(self.device))
        return self

    torch.Tensor.scatter_add_ = mark(scatter_add_)
