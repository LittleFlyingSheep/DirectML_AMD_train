# -*- coding: utf-8 -*-
"""DirectML_AMD_train —— 让 AMD 显卡（DirectX 12）跑 PyTorch / ultralytics 训练。

针对的是 ROCm 覆盖不到的卡（GCN5 / Vega 20 / gfx906 这类），
以及 Windows 上没有 ROCm 的情况：Windows 上唯一能吃到 AMD 显卡的
PyTorch 路线就是 torch-directml。

但 DirectML 不是 CUDA，后面有一批算子语义与 CPU 不一致，
直接跑 ultralytics 的训练会「跑得完、但不收敛」。
这个库把这些差异一条条补上，并且用可复现的回归测试守住。

用法::

    from directml_amd_train import enable_directml_amd
    enable_directml_amd()            # 必须在 import ultralytics 之前
    from ultralytics import YOLO
    YOLO("yolov8s.pt").train(data="data.yaml", device="privateuseone:0")

或者直接用命令行::

    python -m directml_amd_train train --data data.yaml --epochs 25
    python -m directml_amd_train doctor      # 环境与数值对齐自检
"""
from __future__ import annotations

from ._patches import (
    enable_directml_amd,
    patch_addmv,
    patch_bool_ops,
    patch_bce,
    patch_cat,
    patch_device,
    patch_gather,
    patch_inference_mode,
    patch_loss_on_cpu,
    patch_memory,
    patch_scatter,
    patch_unique,
)
from .selftest import check_gradient_alignment
from .trainer import export_onnx, train_yolo

__version__ = "1.0.0"

__all__ = [
    "enable_directml_amd",
    "train_yolo",
    "export_onnx",
    "check_gradient_alignment",
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
    "__version__",
]
