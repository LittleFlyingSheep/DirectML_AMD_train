# -*- coding: utf-8 -*-
"""通用训练封装：让 ultralytics 的 YOLO 能在 AMD 显卡（DirectML）上训练。

注意本模块在 import 时不碰 ultralytics —— 补丁必须在 import ultralytics
之前打上，所以 ultralytics 的 import 一律放在函数里、且在
enable_directml_amd() 之后。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = ["train_yolo", "export_onnx", "resolve_device", "environment_info"]


def resolve_device(device: Any = "dml"):
    """把用户写的 device 规格解析成 torch.device。

    - "dml" / "directml" / "privateuseone" -> privateuseone:0
    - "cpu"                               -> cpu
    - 其它原样交给 ultralytics（"0" / "cuda:0" / ...）
    """
    import torch

    s = str(device).strip().lower()
    if s in ("dml", "directml", "privateuseone", "privateuseone:0", "gpu", "amd"):
        return torch.device("privateuseone:0")
    if s == "cpu":
        return torch.device("cpu")
    return device


def environment_info() -> Dict[str, Any]:
    """收集环境信息，给 doctor 命令用。"""
    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception as e:  # pragma: no cover
        info["torch"] = "未安装 (%s)" % e
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception as e:
        info["ultralytics"] = "未安装 (%s)" % e
    try:
        import torch_directml
        info["torch_directml"] = getattr(torch_directml, "__version__", "已安装")
        try:
            info["device_name"] = torch_directml.device_name(0)
        except Exception as e:
            info["device_name"] = "读取失败 (%s)" % e
    except Exception as e:
        info["torch_directml"] = "未安装 (%s)" % e
    return info


def train_yolo(
    data: str,
    *,
    model: str = "yolov8s.pt",
    epochs: int = 25,
    imgsz: int = 512,
    batch: int = 8,
    device: Any = "dml",
    project: Optional[str] = None,
    name: str = "exp",
    workers: int = 0,
    patience: int = 15,
    seed: int = 0,
    exist_ok: bool = True,
    loss_on_cpu: bool = False,
    verify_index_bounds: bool = True,
    flat_index: bool = True,
    **kwargs: Any,
):
    """在 DirectML 显卡上训练 YOLO。

    参数与 ultralytics 的 model.train() 对齐，多出来的
    device="dml" 是这里特有的写法（"cpu" 仍然走原生 CPU 路径）。

    loss_on_cpu=True 时把损失计算放到 CPU（默认 False，损失整段在显卡上）。
    小模型上 CPU 算损失反而快一点，详见 patch_loss_on_cpu 的说明。

    verify_index_bounds=True 时保留高级索引的越界检查（越界抛 IndexError）；
    DirectML 的 gather 越界不报错会读到野内存，默认开着。
    关掉能再省一点时间（每个索引点约 0.3 ms）。
    """
    from ._patches import enable_directml_amd

    dev = resolve_device(device)
    if getattr(dev, "type", "") == "privateuseone":
        enable_directml_amd(loss_on_cpu=loss_on_cpu,
                            verify_index_bounds=verify_index_bounds,
                            flat_index=flat_index)

    from ultralytics import YOLO  # noqa: E402  必须在补丁之后 import

    if os.path.isdir(data):
        data = os.path.join(data, "data.yaml")

    train_kwargs: Dict[str, Any] = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev,
        workers=workers,
        seed=seed,
        patience=patience,
        exist_ok=exist_ok,
        amp=False,          # DirectML 不支持 autocast
    )
    if project:
        train_kwargs["project"] = project
    train_kwargs["name"] = name
    # ultralytics 各版本参数略有差异，传不认识的参数会报错，这里挑着传
    safe = ("mosaic", "mixup", "degrees", "scale", "translate", "fliplr", "flipud",
            "hsv_h", "hsv_s", "hsv_v", "copy_paste", "close_mosaic", "optimizer",
            "lr0", "lrf", "cos_lr", "val", "plots", "save_period", "cache",
            "single_cls", "rect", "resume", "pretrained")
    for k in safe:
        if k in kwargs:
            train_kwargs[k] = kwargs[k]

    print("[directml_amd_train] 开始训练: model=%s device=%s" % (model, dev))
    m = YOLO(model)
    m.train(**train_kwargs)
    return m


def export_onnx(
    weights: str,
    out: str,
    *,
    imgsz: int = 512,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
    nms: bool = False,
) -> str:
    """把训练好的 .pt 导出成 ONNX（onnxruntime 跑这个）。"""
    from ultralytics import YOLO

    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, "model.onnx")
    YOLO(weights).export(format="onnx", imgsz=imgsz, opset=opset,
                          simplify=simplify, dynamic=dynamic, nms=nms,
                          device="cpu")
    # ultralytics 把 onnx 放在权重旁边，搬过来
    import shutil
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(weights)), "model.onnx"),
        os.path.join(os.path.dirname(os.path.abspath(weights)), "best.onnx"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.path.abspath(c) != os.path.abspath(target):
            shutil.copyfile(c, target)
            break
    if not os.path.isfile(target):
        raise FileNotFoundError("导出失败：没找到 onnx 文件（检查权重路径）")
    print("[directml_amd_train] ONNX 已导出: %s（%d 字节）"
          % (target, os.path.getsize(target)))
    return target
