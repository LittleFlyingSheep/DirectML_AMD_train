# -*- coding: utf-8 -*-
"""DirectML 后端差异补丁集。

补丁分两层：

* **torch 层**：改的是 torch / torch_directml 自己的算子语义，跟框架无关。
  任何在 DirectML 上跑 PyTorch 的项目（YOLO、扩散模型、自定义训练脚本……）
  都该打这一层。
* **ultralytics 层**：改的是 ultralytics 的内部行为（设备选择、显存日志、
  损失搬 CPU、OBB 的 probiou），只有用 ultralytics 时才需要。

enable(framework=...) 决定打哪一层：

    enable()                        # auto：装了 ultralytics 就两层，没装就只 torch 层
    enable(framework="torch")       # 只打 torch 层，绝不碰 ultralytics
    enable(framework="ultralytics") # 两层都打（会 import ultralytics）

调用顺序有讲究：patch_inference_mode() 必须在 import ultralytics 之前生效
（ultralytics 的 smart_inference_mode 装饰器在 import 时就把
torch.inference_mode 这个类抓在手里）。
另外 patch_slot_assign() 必须排在 patch_bool_ops() 后面 ——
bool_ops 也会替换 Tensor.__imul__ 等 dunder，谁后打谁生效。
"""
from __future__ import annotations

import importlib.util

from .addmv import patch_addmv
from .bool_ops import patch_bool_ops
from .bce import patch_bce
from .cat import patch_cat
from .device import patch_device
from .flat_index import patch_flat_index
from .gather import patch_gather
from .inference_mode import patch_inference_mode
from .loss_on_cpu import patch_loss_on_cpu
from .memory import patch_memory, patch_ultralytics_memory
from .probiou import patch_probiou
from .scatter import patch_scatter
from .slot_assign import patch_slot_assign
from .sort import patch_sort
from .unique import patch_unique

__all__ = [
    "enable",
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
    "patch_probiou",
    "patch_loss_on_cpu",
]

# torch 层：跟框架无关，任何在 DirectML 上跑 PyTorch 的项目都该打
_TORCH_LAYER = (
    patch_inference_mode,
    patch_unique,
    patch_sort,
    patch_scatter,
    patch_bool_ops,
    patch_memory,
    patch_cat,
    patch_gather,
    patch_bce,
    patch_addmv,
    patch_slot_assign,
)

# ultralytics 层：改 ultralytics 自己的行为
_ULTRALYTICS_LAYER = (
    patch_device,
    patch_ultralytics_memory,
    patch_probiou,
)

_FRAMEWORKS = ("auto", "torch", "ultralytics", "none")


def _ultralytics_installed() -> bool:
    """只查不导入：ultralytics 的 import 有副作用，不能为了探测就 import 它。"""
    try:
        return importlib.util.find_spec("ultralytics") is not None
    except Exception:
        return False


def enable(framework: str = "auto", verbose: bool = True,
           loss_on_cpu: bool = False,
           verify_index_bounds: bool = True,
           flat_index: bool = True) -> str:
    """给 DirectML 打上补丁。返回 DirectML 设备名。

    参数
    ----
    framework:
        "auto"（默认）—— 装了 ultralytics 就打两层，没装就只打 torch 层；
        "torch" —— 只打 torch 层（纯 PyTorch 项目用这个，不碰 ultralytics）；
        "ultralytics" —— 两层都打；
        "none" —— 一条都不打（只做环境探测时用）。
    verbose:
        打印生效的设备名与补丁层。
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

        from Direct_AMD_train import enable
        enable()          # 先打补丁
        from ultralytics import YOLO    # 再 import ultralytics
        YOLO("yolov8s.pt").train(...)
    """
    if framework not in _FRAMEWORKS:
        raise ValueError("framework 只能是 %s 之一" % (", ".join(_FRAMEWORKS),))

    if framework == "none":
        return ""

    layers = ["torch"]
    if framework == "ultralytics" or (framework == "auto" and _ultralytics_installed()):
        layers.append("ultralytics")

    # 1) torch 层（不依赖 ultralytics，必须在 import ultralytics 之前）
    for fn in _TORCH_LAYER:
        fn()

    # 2) 高级索引改写成展平 gather（带参数，单独调）
    if flat_index:
        patch_flat_index(verify_bounds=verify_index_bounds)

    # 3) ultralytics 层
    if "ultralytics" in layers:
        for fn in _ULTRALYTICS_LAYER:
            fn()
        if loss_on_cpu:
            patch_loss_on_cpu()

    name = ""
    try:
        import torch_directml
        name = torch_directml.device_name(0)
    except Exception:
        pass
    if verbose:
        print("[Direct_AMD_train] 补丁已生效（%s 层），设备: %s"
              % ("+".join(layers), name or "privateuseone:0"))
    return name
