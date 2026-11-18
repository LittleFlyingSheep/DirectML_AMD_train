# -*- coding: utf-8 -*-
"""补丁 3：绕开 DirectML 的 unique 算子缺口。

DirectML 的 unique 有三种坏法（实测 torch-directml 0.2.5.dev240914）：

1) 输入是二维以上时，**直接把进程打崩**：

       dml_tensor_desc.cc:76  Check failed: broadcast_sizes.size() >= sizes.size()

   退出码 0xC0000409 —— 这是原生层断言，不是 Python 异常，
   try/except 拦不住，整个训练进程直接没了。

2) return_counts=True 时报 RuntimeError：

       Return counts not implemented for unique operator for DirectML

3) 一维输入且 dtype 是 bool 时**数值就是错的**：

       torch.unique(torch.tensor([True, False, True], device=DML))
       CPU -> [False, True]
       DML -> [True]           <- 少了 False

所以这里按情况处理：

  * return_counts / 指定了 dim / bool dtype -> 回 CPU 算
  * 一维 + 数值类型 -> 留在显卡上（实测与 CPU 逐位一致）

ultralytics 的标签匹配用到 unique(return_counts=True)，
本来就要回 CPU（张量很小，只处理索引）；
这条补丁同时堵上了「普通 unique 把进程打崩」这个坑。
"""
from __future__ import annotations


def _to_device(out, device):
    import torch

    if isinstance(out, tuple):
        return tuple(t.to(device) if torch.is_tensor(t) else t for t in out)
    return out.to(device)


def patch_unique() -> None:
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.unique):
        return

    _orig = torch.unique

    def unique(input, *args, **kw):
        if not (torch.is_tensor(input) and input.device.type == "privateuseone"):
            return _orig(input, *args, **kw)

        # 签名是 unique(input, sorted=True, return_counts=False, dim=None)
        want_counts = kw.get("return_counts")
        if want_counts is None and len(args) >= 2:
            want_counts = args[1]
        dim = kw.get("dim", None)
        if dim is None and len(args) >= 3:
            dim = args[2]

        # 一维 + 数值类型：DirectML 上实测与 CPU 逐位一致，留在显卡上
        safe_on_device = (not want_counts
                           and dim is None
                           and input.dim() == 1
                           and input.dtype != torch.bool)
        if safe_on_device:
            return _orig(input, *args, **kw)

        out = _orig(input.cpu(), *args, **kw)
        return _to_device(out, input.device)

    torch.unique = mark(unique)
