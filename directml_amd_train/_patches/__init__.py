# -*- coding: utf-8 -*-
"""DirectML 后端差异补丁集。

调用顺序有讲究：patch_inference_mode() 必须在 import ultralytics 之前生效
（ultralytics 的 smart_inference_mode 装饰器在 import 时就把
torch.inference_mode 这个类抓在手里），所以 enable_directml_amd() 会
先打这个补丁，再去 import ultralytics 打后面几个。

另外 patch_slot_assign() 必须排在 patch_bool_ops() 后面 ——
bool_ops 也会替换 Tensor.__imul__ 等 dunder，谁后打谁生效。
"""
from __future__ import annotations

from .addmv import patch_addmv
from .bool_ops import patch_bool_ops
from .bce import patch_bce
from .cat import patch_cat
from .device import patch_device
from .flat_index import patch_flat_index
from .gather import patch_gather
from .inference_mode import patch_inference_mode
from .loss_on_cpu import patch_loss_on_cpu
from .memory import patch_memory
from .scatter import patch_scatter
from .slot_assign import patch_slot_assign
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
    "patch_slot_assign",
    "patch_flat_index",
    "patch_loss_on_cpu",
]

# 不依赖 ultralytics 的补丁；inference_mode 尤其要最先，
# slot_assign 要在 bool_ops 之后（两者都改 Tensor 的 dunder）
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
    patch_slot_assign,
)


def enable_directml_amd(verbose: bool = True, loss_on_cpu: bool = False,
                       verify_index_bounds: bool = True,
                       flat_index: bool = True) -> str:
    """给 DirectML 打上 ultralytics 训练需要的全部补丁。

    必须在 import ultralytics 之前调用。返回 DirectML 设备名。

    参数
    ----
    verbose:
        打印生效的设备名。
    loss_on_cpu:
        把损失计算改回 CPU（默认 False，也就是损失整段留在显卡上）。
        DirectML 每个算子都有固定开销，小模型上损失在 CPU 反而快一点；
        大模型/大输入用默认的显卡路径更划算。详见 patch_loss_on_cpu。
    verify_index_bounds:
        高级索引（补丁 12）是否保留 PyTorch 的越界检查（越界抛 IndexError）。
        默认开着：DirectML 的 gather 越界不报错，会直接读到野内存。
        关掉能再省一点时间（每个索引点约 0.3 ms）。
    flat_index:
        是否启用补丁 12（把高级索引改写成展平 gather）。
        默认开着 —— 实测损失前向快 3.3 倍、整轮训练快 4.3%（配对 A/B）。
        关掉可以拿来做 A/B 对照（数值一样，只是慢）。

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
    if loss_on_cpu:
        patch_loss_on_cpu()

    # 3) 高级索引改写成展平 gather（带参数，单独调）
    if flat_index:
        patch_flat_index(verify_bounds=verify_index_bounds)

    name = ""
    try:
        import torch_directml
        name = torch_directml.device_name(0)
    except Exception:
        pass
    if verbose:
        print("[directml_amd_train] 补丁已生效，设备: %s" % (name or "privateuseone:0"))
    return name
