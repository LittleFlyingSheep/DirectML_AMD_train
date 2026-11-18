# -*- coding: utf-8 -*-
"""补丁 10：避开 DirectML 上 addmv / matmul 的 CPU fallback（损失计算里用到）。

实测（torch-directml 0.2.5.dev240914）：

    aten::addmv.out 在 DirectML 上不支持，会 fallback 到 CPU；
    而 matmul(N-D, 1-D) 在 DirectML 上正好被拆成 addmv，
    于是 ultralytics 的 bbox_decode 里这一行会整块搬到 CPU 上算：

        pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj)

    （ultralytics/utils/loss.py 的 bbox_decode）

改写方式：
    addmv(input, mat, vec)  ->  (mat * vec).sum(dim=1)
    A(N-D) @ v(1-D)         ->  (A * v).sum(-1)

语义等价（都是矩阵-向量乘），元素少的时候差别不大，
但省掉的是一整次 CPU/显卡之间的数据搬运。
"""
from __future__ import annotations


def patch_addmv() -> None:
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.addmv):
        return

    _orig_addmv = torch.addmv
    _orig_matmul = torch.matmul
    _orig_tensor_matmul = torch.Tensor.matmul

    def addmv(input, mat, vec, beta=1, alpha=1):
        if (isinstance(input, torch.Tensor) and input.device.type == "privateuseone"
                and mat.dim() == 2 and vec.dim() == 1):
            out = (mat * vec.reshape(1, -1)).sum(dim=1)
            return alpha * out + beta * input
        return _orig_addmv(input, mat, vec, beta=beta, alpha=alpha)

    def _needs_rewrite(a, b):
        # DirectML 上 (N-D, 1-D) 的 matmul 会被拆成 addmv -> CPU fallback
        return (isinstance(a, torch.Tensor) and a.device.type == "privateuseone"
                and isinstance(b, torch.Tensor)
                and b.dim() == 1 and a.dim() > 2)

    def matmul(input, other, *args, **kw):
        if _needs_rewrite(input, other):
            return (input * other).sum(-1)
        return _orig_matmul(input, other, *args, **kw)

    def tensor_matmul(self, other):
        # ultralytics 用的是方法形式 t.matmul(v)，只补函数形式会漏掉
        if _needs_rewrite(self, other):
            return (self * other).sum(-1)
        return _orig_tensor_matmul(self, other)

    torch.addmv = mark(addmv)
    torch.matmul = mark(matmul)
    torch.Tensor.matmul = mark(tensor_matmul)
