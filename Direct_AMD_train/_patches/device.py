# -*- coding: utf-8 -*-
"""补丁 2：让 ultralytics 认得 privateuseone:0 这个设备。

ultralytics 的 select_device 只认 cpu/cuda/npu/xpu，
传 privateuseone:0 会被当成 cuda 报 "Invalid CUDA device"。

这里把 select_device 换掉：保留 privateuseone 的设备编号，其它设备走原来的逻辑。
"""
from __future__ import annotations

import sys

# 各个模块是 import 时就把 select_device 的名字绑到自己身上了，
# 光改 ultralytics.utils.torch_utils 不够，得把已加载模块里的引用一起换掉。
_MODULES = (
    "ultralytics.engine.trainer",
    "ultralytics.engine.predictor",
    "ultralytics.engine.exporter",
    "ultralytics.models.yolo.detect.train",
    "ultralytics.models.yolo.detect.val",
    "ultralytics.nn.autobackend",
    "ultralytics.utils.torch_utils",
)


def patch_device() -> None:
    from ultralytics.utils import torch_utils

    from ..trainer import resolve_device
    from ._once import already_patched, mark

    if already_patched(torch_utils.select_device):
        return

    _orig = torch_utils.select_device

    def select_device(device="", *args, **kw):
        # 签名随 ultralytics 版本变，这里一律透传，只把 privateuseone 拦下来
        spec = str(device).strip().lower()
        if spec == "privateuseone" or spec.startswith("privateuseone:"):
            return resolve_device(device)
        return _orig(device, *args, **kw)

    torch_utils.select_device = mark(select_device)
    for mod_name in _MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "select_device"):
            mod.select_device = select_device
