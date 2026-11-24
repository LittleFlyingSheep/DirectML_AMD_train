# -*- coding: utf-8 -*-
"""通用训练封装：让 ultralytics 的 YOLO 能在 AMD 显卡（DirectML）上训练。

注意本模块在 import 时不碰 ultralytics —— 补丁必须在 import ultralytics
之前打上，所以 ultralytics 的 import 一律放在函数里、且在
enable() 之后。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = ["train_yolo", "predict_yolo", "export_onnx", "resolve_device", "environment_info",
           "save_weights"]


def resolve_device(device: Any = "dml"):
    """把用户写的 device 规格解析成 torch.device。

    - "dml" / "directml" / "privateuseone" -> privateuseone:0
    - "privateuseone:N"                   -> 指定的 DirectML 适配器
    - "cpu"                               -> cpu
    - 其它原样交给 ultralytics（"0" / "cuda:0" / ...）
    """
    import torch

    s = str(device).strip().lower()
    if s in ("dml", "directml", "privateuseone", "gpu", "amd") or s.startswith("privateuseone:"):
        import torch_directml

        index = torch.device(s).index if ":" in s else None
        return torch_directml.device(0 if index is None else index)
    if s == "cpu":
        return torch.device("cpu")
    return device


def environment_info() -> Dict[str, Any]:
    """收集环境信息，给 doctor 命令用。"""
    from importlib.metadata import PackageNotFoundError, version

    info: Dict[str, Any] = {}
    for package in ("torch", "ultralytics", "torch-directml"):
        try:
            info[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:
            info[package.replace("-", "_")] = "未安装"
    try:
        import torch_directml

        info["device_name"] = torch_directml.device_name(0).rstrip("\0") or "无可用设备"
    except Exception as e:
        info["device_name"] = "读取失败 (%s)" % e
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
    from ._patches import enable

    dev = resolve_device(device)
    if getattr(dev, "type", "") == "privateuseone":
        enable(framework="ultralytics", loss_on_cpu=loss_on_cpu,
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
    train_kwargs.update(kwargs)
    if getattr(dev, "type", "") == "privateuseone":
        train_kwargs["amp"] = False

    print("[Direct_AMD_train] 开始训练: model=%s device=%s" % (model, dev))
    m = YOLO(model)
    m.train(**train_kwargs)
    return m


def predict_yolo(source: Any, *, model: str = "yolov8s.pt", device: Any = "dml", **kwargs: Any):
    """用 PyTorch .pt 权重预测，返回 Ultralytics Results 列表或迭代器。"""
    from ._patches import enable

    dev = resolve_device(device)
    if getattr(dev, "type", "") == "privateuseone":
        enable(framework="ultralytics")
        kwargs["quantize"] = 32

    from ultralytics import YOLO

    return YOLO(model).predict(source=source, device=dev, **kwargs)


def save_weights(model, file) -> None:
    """将模型 state_dict 保存为 CPU 张量，供 torch.load(weights_only=True) 加载。"""
    import torch

    state = model.state_dict()
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            # DirectML 的 NumPy 序列化路径无法由默认的 weights_only 加载器读取。
            state[name] = value.cpu()
    torch.save(state, file)


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
    exported = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=opset,
                                    simplify=simplify, dynamic=dynamic, nms=nms,
                                    device="cpu")
    import shutil

    if not os.path.isfile(exported):
        raise FileNotFoundError("导出失败：未生成 %s" % exported)
    if not os.path.exists(target) or not os.path.samefile(exported, target):
        shutil.copyfile(exported, target)
    print("[Direct_AMD_train] ONNX 已导出: %s（%d 字节）"
          % (target, os.path.getsize(target)))
    return target
