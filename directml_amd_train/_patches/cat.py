# -*- coding: utf-8 -*-
"""补丁 7：绕开 DirectML 的 cat 缺口 —— 拼接维上长度为 0 的张量不能参与 cat。

实测（torch-directml 0.2.5.dev240914）：
    torch.cat((a, b, c, empty_0col), 1) -> UnicodeDecodeError
    torch.cat((a, b, c), 1)             -> 正常
ultralytics 的 NMS 里 x.split((4, nc, extra), 1) 切出来的 mask 是 0 列张量，
再和其它三块 cat 就必然踩雷（后端把 D3D12 的失败错误地报成了编码错）。

语义上空张量不贡献任何元素，把它跳过，结果与原来完全等价。
"""
from __future__ import annotations


def patch_cat() -> None:
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.cat):
        return

    _orig = torch.cat

    def cat(tensors, dim=0, *a, **kw):
        ts = list(tensors)
        if ts:
            keep = [t for t in ts if not (t.dim() > dim and t.size(dim) == 0)]
            if len(keep) != len(ts):
                if not keep:
                    t0 = ts[0]
                    shape = list(t0.shape)
                    shape[dim] = 0
                    return t0.new_empty(shape)
                ts = keep
        return _orig(ts, dim, *a, **kw)

    torch.cat = mark(cat)
