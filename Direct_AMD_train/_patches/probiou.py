# -*- coding: utf-8 -*-
"""补丁 14：OBB（旋转框）任务用的 batch_probiou 在 DirectML 上不支持。

ultralytics 的 TorchNMS.fast_nms 在 rotated=True 时会调 batch_probiou
（旋转框的 IoU）。DirectML 上这个算子直接报 UnicodeDecodeError
（后端错误消息解不出来）：

    ultralytics.utils.metrics.batch_probiou(boxes1, boxes2)
    -> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb2 ...

检测框数量不多（几十到几百），回 CPU 算完再搬回来就行。

注意 ultralytics.utils.nms 在 import 时就把 batch_probiou 的名字
绑到自己身上了，所以要连已加载模块里的引用一起换掉。
"""
from __future__ import annotations

import sys

_MODULES = (
    "ultralytics.utils.nms",
    "ultralytics.utils.metrics",
    "ultralytics.models.yolo.obb.val",
)


def patch_probiou() -> None:
    """把 batch_probiou 换成「回 CPU 算」的版本（ultralytics 层）。"""
    import torch
    from ultralytics.utils import metrics as _metrics

    from ._once import already_patched, mark

    if already_patched(_metrics.batch_probiou):
        return

    _orig = _metrics.batch_probiou

    def batch_probiou(boxes1, boxes2, *args, **kw):
        if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
            return _orig(boxes1, boxes2, *args, **kw)
        if boxes1.device.type != "privateuseone" and boxes2.device.type != "privateuseone":
            return _orig(boxes1, boxes2, *args, **kw)
        dev = boxes1.device if boxes1.device.type == "privateuseone" else boxes2.device
        out = _orig(boxes1.cpu(), boxes2.cpu(), *args, **kw)
        if torch.is_tensor(out):
            return out.to(dev)
        if isinstance(out, tuple):
            return tuple(t.to(dev) if torch.is_tensor(t) else t for t in out)
        return out

    _metrics.batch_probiou = mark(batch_probiou)
    for mod_name in _MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "batch_probiou"):
            mod.batch_probiou = batch_probiou
