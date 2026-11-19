# -*- coding: utf-8 -*-
"""Direct_AMD_train —— 让 AMD 显卡（DirectX 12）跑 PyTorch 训练与调试。

定位：Windows 上没有 ROCm 的 AMD 卡（GCN5 / Vega 20 / gfx906 这类，
以及没有装 ROCm 的机器）上，唯一能吃到显卡的 PyTorch 路线就是 torch-directml。
但 DirectML 不是 CUDA，有一批算子语义与 CPU 不一致 ——
典型表现是「跑得完、loss 也往下走、就是学不会」，
还有更直接的「直接把进程打崩」。

这个库把这些差异一条条补上，并且用可复现的回归测试守住。

分两层：

  * **torch 层**：改 torch / torch_directml 自己的算子语义，与框架无关。
    任何在 DirectML 上跑 PyTorch 的项目都该打这一层。
  * **ultralytics 层**：改 ultralytics 的内部行为（设备选择、显存日志、
    OBB 的 probiou、损失搬 CPU）。

用法::

    from Direct_AMD_train import enable
    enable()            # 必须在 import ultralytics 之前
    from ultralytics import YOLO
    YOLO("yolov8s.pt").train(data="data.yaml", device="privateuseone:0")

纯 PyTorch 项目（不装 ultralytics）也可以用::

    from Direct_AMD_train import enable
    enable(framework="torch")

或者直接用命令行::

    python -m Direct_AMD_train train --data data.yaml --epochs 25
    python -m Direct_AMD_train doctor      # 环境与数值对齐自检
    python -m Direct_AMD_train probe       # 算子能力探测
    python -m Direct_AMD_train patches     # 补丁清单
"""
from __future__ import annotations

from ._patches import (
    enable,
    patch_addmv,
    patch_bool_ops,
    patch_bce,
    patch_cat,
    patch_device,
    patch_empty_ops,
    patch_flat_index,
    patch_gather,
    patch_inference_mode,
    patch_loss_on_cpu,
    patch_memory,
    patch_probiou,
    patch_scatter,
    patch_slot_assign,
    patch_sort,
    patch_unique,
    patch_ultralytics_memory,
)
from .probe import format_probe, probe_operators
from .registry import format_patches, list_patches
from .selftest import check_gradient_alignment
from .trainer import export_onnx, train_yolo

__version__ = "2.0.0.dev0"

__all__ = [
    "enable",
    "train_yolo",
    "export_onnx",
    "check_gradient_alignment",
    "probe_operators",
    "format_probe",
    "list_patches",
    "format_patches",
    "patch_inference_mode",
    "patch_device",
    "patch_unique",
    "patch_sort",
    "patch_scatter",
    "patch_bool_ops",
    "patch_memory",
    "patch_ultralytics_memory",
    "patch_cat",
    "patch_gather",
    "patch_bce",
    "patch_addmv",
    "patch_slot_assign",
    "patch_flat_index",
    "patch_empty_ops",
    "patch_probiou",
    "patch_loss_on_cpu",
    "__version__",
]
