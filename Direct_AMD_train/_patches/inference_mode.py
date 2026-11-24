# -*- coding: utf-8 -*-
"""补丁 1：把 torch.inference_mode 换成 torch.no_grad。

DirectML 的 BatchNorm 实现会原地写缓冲区，而 PyTorch 禁止对 inference tensor
做原地操作，于是验证阶段（ultralytics 的 validator 用 @smart_inference_mode()
整段包住）在第一个 batch 前向就报：

    RuntimeError: Cannot set version_counter for inference tensor

修法：用 torch.no_grad() 关梯度，避免创建 inference tensor。
显式传 mode=False 时保留 PyTorch 的原始行为。

注意时机：ultralytics 的 smart_inference_mode 装饰器在 import 时就把
torch.inference_mode 这个类抓在手里，之后再改 torch.inference_mode 不起作用。
所以这个补丁必须在 import ultralytics 之前生效。
"""
from __future__ import annotations


def patch_inference_mode() -> None:
    """替换 torch.inference_mode（必须在 import ultralytics 之前调用）。"""
    import torch

    from ._once import already_patched, mark

    original = torch.inference_mode
    if already_patched(original):
        return

    def inference_mode(mode=True):
        if not isinstance(mode, bool):
            return torch.no_grad()(mode)
        return torch.no_grad() if mode else original(False)

    torch.inference_mode = mark(inference_mode)
