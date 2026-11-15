# -*- coding: utf-8 -*-
"""补丁 5：把 DirectML 的布尔/整型混合运算语义修成和 CPU 一致。

实测（AMD Radeon Pro VII / torch-directml 0.2.5.dev240914）：
    float32 * bool  -> 结果 dtype 被判成 bool，数值也被截断（全 False）
    float32 * int8  -> 结果 dtype 被判成 bool
    int8   * bool   -> 结果 dtype 被判成 bool
CPU 上这三行分别是 float32 / float32 / int8。

这会让 ultralytics 的标签分配炸掉：
    align_metric *= mask_pos        # mask_pos 是 bool/int8 掩码
    align_metric.mul_(pos_overlaps) # -> RuntimeError: result type Float
                                     #    can't be cast to the desired
                                     #    output type Bool

修法：两个操作数 dtype 不一致时，把「不是浮点的那一侧」转成对方的 dtype，
再走原算子。CPU 路径上这个转换与 PyTorch 自己的类型提升结果一致，
所以不影响正确性。
"""
from __future__ import annotations


def patch_bool_ops() -> None:
    import torch

    def _cast(self, other):
        if not isinstance(other, torch.Tensor) or other.dtype == self.dtype:
            return other
        other_float = other.dtype.is_floating_point
        self_float = self.dtype.is_floating_point
        if other.dtype == torch.bool or (self_float and not other_float):
            return other.to(self.dtype)
        return other

    def _wrap(name):
        orig = getattr(torch.Tensor, name)

        def fn(self, *args, **kw):
            if args:
                args = (_cast(self, args[0]),) + args[1:]
            return orig(self, *args, **kw)

        setattr(torch.Tensor, name, fn)

    for _n in ("mul", "mul_", "div", "div_", "add", "add_", "sub", "sub_"):
        _wrap(_n)

    # dunder 走的是 C 层槽位，不会用到上面替换掉的方法，这里一并指过去
    torch.Tensor.__mul__ = lambda self, other: torch.Tensor.mul(self, other)
    torch.Tensor.__imul__ = lambda self, other: torch.Tensor.mul_(self, other)
    torch.Tensor.__truediv__ = lambda self, other: torch.Tensor.div(self, other)
    torch.Tensor.__itruediv__ = lambda self, other: torch.Tensor.div_(self, other)
    torch.Tensor.__add__ = lambda self, other: torch.Tensor.add(self, other)
    torch.Tensor.__iadd__ = lambda self, other: torch.Tensor.add_(self, other)
    torch.Tensor.__sub__ = lambda self, other: torch.Tensor.sub(self, other)
    torch.Tensor.__isub__ = lambda self, other: torch.Tensor.sub_(self, other)
