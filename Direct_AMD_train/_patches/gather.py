# -*- coding: utf-8 -*-
"""补丁 8：修 DirectML 上 gather 的反向 —— 散射回来的梯度被摊薄了。

实测（torch-directml 0.2.5.dev240914）：
    x.gather(1, 全 0 索引).sum().backward()
    CPU  -> x.grad[0] = 64（64 份贡献全落在同一个位置）
    DML  -> x.grad[0] = 1  （被当成平均值摊掉了）
也就是说 DirectML 把「散射累加」实现成了「取平均」。

同族算子的实测结论（都用同一份输入数据逐个验过）：
    amax / max(dim) / where / sum / mean / softmax / 高级索引 / scatter_  反向都正确
    唯一有问题的是 gather 的反向。

注意要同时拦两种写法：
    torch.gather(x, dim, index)   —— 函数形式
    x.gather(dim, index)          —— 方法形式（内部不会走 torch.gather）
只补函数形式的话，方法形式完全漏掉。

YOLO 的训练路径（ultralytics/utils/tal.py + utils/loss.py）里
取预测分数用的是高级索引 pd_scores[indices[0], indices[2], cls]，
不是 gather，所以本补丁只在真正调用到 gather 时才生效。
"""
from __future__ import annotations


def patch_gather() -> None:
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.gather):
        return

    _orig_gather = torch.gather
    _orig_tensor_gather = torch.Tensor.gather

    class _GatherFix(torch.autograd.Function):
        """gather 的反向：把输出梯度按 index 散射累加回输入。

        DirectML 把这一步做成了「取平均」，这里改成在 CPU 上做 scatter_add
        （PyTorch 的 scatter_add 语义可靠），再把梯度搬回原设备。
        """

        @staticmethod
        def forward(ctx, inp, idx, dim_):
            ctx.save_for_backward(idx)
            ctx.dim_ = dim_
            ctx.shape = inp.shape
            return _orig_gather(inp, dim_, idx)

        @staticmethod
        def backward(ctx, grad_out):
            (idx,) = ctx.saved_tensors
            g = torch.zeros(ctx.shape, dtype=grad_out.dtype)
            # idx 可能是显卡上的张量，scatter_add_ 要求三个操作数同一个设备
            g.scatter_add_(ctx.dim_, idx.cpu(), grad_out.cpu())
            return g.to(grad_out.device), None, None

    def _needs_fix(inp, index):
        return (isinstance(inp, torch.Tensor) and inp.device.type == "privateuseone"
                and isinstance(index, torch.Tensor) and index.dtype == torch.int64
                and inp.requires_grad)

    def gather(input, dim, index, *args, **kw):
        if _needs_fix(input, index):
            return _GatherFix.apply(input, index, dim)
        return _orig_gather(input, dim, index, *args, **kw)

    def tensor_gather(self, dim, index, *args, **kw):
        if _needs_fix(self, index):
            return _GatherFix.apply(self, index, dim)
        return _orig_tensor_gather(self, dim, index, *args, **kw)

    torch.gather = mark(gather)
    torch.Tensor.gather = mark(tensor_gather)
