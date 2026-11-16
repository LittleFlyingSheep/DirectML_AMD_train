# -*- coding: utf-8 -*-
"""补丁 3：绕开 DirectML 的 unique 算子缺口。

DirectML 版的 unique 不支持 return_counts，
而 ultralytics 的标签匹配会用到（RuntimeError:
Return counts not implemented for unique operator for DirectML）。

这里把 unique 换成：先搬回 CPU 算，再把结果搬回显卡。
只在真的要 counts、而且张量在显卡上时才绕，其它情况走原逻辑。
"""
from __future__ import annotations


def patch_unique() -> None:
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.unique):
        return

    _orig = torch.unique

    def unique(input, *args, **kw):
        want_counts = kw.get("return_counts")
        if want_counts is None and len(args) >= 3:
            want_counts = args[2]
        if (want_counts and isinstance(input, torch.Tensor)
                and input.device.type == "privateuseone"):
            out = _orig(input.cpu(), *args, **kw)
            if isinstance(out, tuple):
                return tuple(t.to(input.device) if torch.is_tensor(t) else t
                             for t in out)
            return out.to(input.device)
        return _orig(input, *args, **kw)

    torch.unique = mark(unique)
