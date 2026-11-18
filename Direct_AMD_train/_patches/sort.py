# -*- coding: utf-8 -*-
"""补丁 13：给 DirectML 的排序类算子兜底（argsort / sort / topk）。

DirectML 的排序在「待排的值全部相等」时会直接报
UnicodeDecodeError —— 后端给的错误消息本身是乱码，解不出来：

    torch.argsort(torch.full((64,), 0.9, device=DML), descending=True)
    -> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb2 ...

sort / topk 一样会炸，stable=True 也救不回来（走的是同一个内核）。

这不是稀罕场景：NMS 里所有检测框置信度相同时、
匹配算法对相同的代价做 topk 时，都会撞上。

修法：先试原生，炸了就回 CPU 重算。实测算子失败之后
DirectML 上下文照常可用（后面的 matmul 数值正常），所以这条路是安全的。

回 CPU 不影响自动微分：.cpu() 本身可微，CPU 上的 sort / topk 也可微，
搬回显卡后梯度与 CPU 逐位相同（回归测试里有断言）。
"""
from __future__ import annotations


def _to_device(out, device):
    import torch

    if isinstance(out, tuple):
        return tuple(t.to(device) if torch.is_tensor(t) else t for t in out)
    return out.to(device)


def _cpu_args(args, kw):
    import torch

    a = [x.cpu() if torch.is_tensor(x) else x for x in args]
    k = {key: (v.cpu() if torch.is_tensor(v) else v) for key, v in kw.items()}
    return a, k


def patch_sort() -> None:
    """给 argsort / sort / topk 加 CPU 兜底。"""
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.argsort):
        return

    def _wrap(orig):
        def wrapper(*args, **kw):
            try:
                return orig(*args, **kw)
            except Exception:
                # 原生这条路炸了 -> 搬回 CPU 重算
                dev = None
                for x in list(args) + list(kw.values()):
                    if torch.is_tensor(x) and x.device.type == "privateuseone":
                        dev = x.device
                        break
                if dev is None:
                    raise
                a, k = _cpu_args(args, kw)
                return _to_device(orig(*a, **k), dev)

        return mark(wrapper)

    torch.argsort = _wrap(torch.argsort)
    torch.sort = _wrap(torch.sort)
    torch.topk = _wrap(torch.topk)
    torch.Tensor.argsort = _wrap(torch.Tensor.argsort)
    torch.Tensor.sort = _wrap(torch.Tensor.sort)
    torch.Tensor.topk = _wrap(torch.Tensor.topk)
