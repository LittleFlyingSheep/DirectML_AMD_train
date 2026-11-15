# -*- coding: utf-8 -*-
"""DirectML 后端差异补丁集。

调用顺序有讲究：patch_inference_mode() 必须在 import ultralytics 之前生效
（ultralytics 的 smart_inference_mode 装饰器在 import 时就把
torch.inference_mode 这个类抓在手里），所以 enable_directml_amd() 会
先打这个补丁，再去 import ultralytics 打后面几个。
"""
from __future__ import annotations

from .addmv import patch_addmv
from .bool_ops import patch_bool_ops
from .bce import patch_bce
from .cat import patch_cat
from .device import patch_device
from .gather import patch_gather
from .inference_mode import patch_inference_mode
from .loss_on_cpu import patch_loss_on_cpu
from .memory import patch_memory
from .scatter import patch_scatter
from .unique import patch_unique

__all__ = [
    "enable_directml_amd",
    "patch_inference_mode",
    "patch_device",
    "patch_unique",
    "patch_scatter",
    "patch_bool_ops",
    "patch_memory",
    "patch_cat",
    "patch_gather",
    "patch_bce",
    "patch_addmv",
    "patch_loss_on_cpu",
]

# 不依赖 ultralytics 的补丁；inference_mode 尤其要最先
_PATCH_ORDER = (
    patch_inference_mode,
    patch_unique,
    patch_scatter,
    patch_bool_ops,
    patch_memory,
    patch_cat,
    patch_gather,
    patch_bce,
    patch_addmv,
)


def enable_directml_amd(verbose: bool = True) -> str:
    """给 DirectML 打上 ultralytics 训练需要的全部补丁。

    必须在 import ultralytics 之前调用。返回 DirectML 设备名。

    典型用法::

        from directml_amd_train import enable_directml_amd
        enable_directml_amd()          # 先打补丁
        from ultralytics import YOLO    # 再 import ultralytics
        YOLO("yolov8s.pt").train(...)
    """
    # 1) 不依赖 ultralytics 的补丁先打
    for fn in _PATCH_ORDER:
        fn()

    # 2) 需要 ultralytics 的补丁（此时 import 已经安全了）
    patch_device()
    patch_loss_on_cpu()

    name = ""
    try:
        import torch_directml
        name = torch_directml.device_name(0)
    except Exception:
        pass
    if verbose:
        print("[directml_amd_train] 补丁已生效，设备: %s" % (name or "privateuseone:0"))
    return name
