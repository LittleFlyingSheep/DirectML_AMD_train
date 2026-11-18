# -*- coding: utf-8 -*-
"""补丁 1：把 torch.inference_mode 换成 torch.no_grad。

DirectML 的 BatchNorm 实现会原地写缓冲区，而 PyTorch 禁止对 inference tensor
做原地操作，于是验证阶段（ultralytics 的 validator 用 @smart_inference_mode()
整段包住）在第一个 batch 前向就报：

    RuntimeError: Cannot set version_counter for inference tensor

修法：把 torch.inference_mode 换成空上下文。梯度照旧由 ultralytics 同层的
torch.no_grad() 关掉，语义上等价于「关梯度但不进 inference 模式」。

注意时机：ultralytics 的 smart_inference_mode 装饰器在 import 时就把
torch.inference_mode 这个类抓在手里，之后再改 torch.inference_mode 不起作用。
所以这个补丁必须在 import ultralytics 之前生效。
"""
from __future__ import annotations


def patch_inference_mode() -> None:
    """替换 torch.inference_mode（必须在 import ultralytics 之前调用）。"""
    import torch

    # no_grad 既能做上下文管理器，也能当装饰器用，正好顶替 inference_mode
    torch.inference_mode = lambda mode=True, *a, **kw: torch.no_grad()
