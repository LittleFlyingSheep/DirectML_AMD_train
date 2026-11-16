# -*- coding: utf-8 -*-
"""可选补丁：把损失计算放到 CPU 上算（默认不启用）。

=============================================================================
什么时候需要它
=============================================================================
默认路径（patch_slot_assign）已经把损失计算整段留在显卡上，而且梯度
与 CPU 对齐 —— 这是推荐用法。

但 DirectML 每个算子都有固定开销（实测 ~0.2 ~ 3 ms），而 ultralytics
的损失是「很多小算子」堆出来的。在小模型 / 小输入上，CPU 多线程算
这种小张量反而更快：

    YOLOv8s @ 512px, batch 8, AMD Radeon Pro VII 实测
        损失在显卡：前向 70 ms + 反向 28 ms
        损失在 CPU ：前向 25 ms + 反向 13 ms
        （还要额外把预测张量搬回 CPU，约 20 ms）

所以「大模型 / 大输入」用默认的显卡路径更划算，「小模型 / 小输入」
可以用这个开关换 CPU。

用法::

    from directml_amd_train import enable_directml_amd
    enable_directml_amd(loss_on_cpu=True)     # 损失回 CPU 算

或者在训练封装里::

    train_yolo("data.yaml", epochs=25, device="dml", loss_on_cpu=True)
"""
from __future__ import annotations


def patch_loss_on_cpu() -> None:
    """把 v8DetectionLoss 的损失计算搬到 CPU（模型前后向不变）。

    需要 ultralytics 已经 import 之后才调用。
    """
    import torch
    from ultralytics.utils import loss as _loss_mod

    from ._once import already_patched, mark

    if already_patched(_loss_mod.v8DetectionLoss.get_assigned_targets_and_loss):
        return

    _orig = _loss_mod.v8DetectionLoss.get_assigned_targets_and_loss

    def _to_cpu(v):
        if torch.is_tensor(v):
            return v.cpu()
        if isinstance(v, (list, tuple)):
            return type(v)(_to_cpu(t) for t in v)
        return v

    def patched(self, preds, batch, **kw):
        dev = preds["boxes"].device if isinstance(preds, dict) else None
        if dev is None or dev.type != "privateuseone":
            return _orig(self, preds, batch, **kw)

        preds_cpu = {k: _to_cpu(v) for k, v in preds.items()}
        # 损失只用得到 batch_idx / cls / bboxes，img 很大就别搬了
        batch_cpu = {k: _to_cpu(batch[k]) for k in
                     ("batch_idx", "cls", "bboxes") if k in batch}

        old = (self.device, self.proj, self.stride)
        self.device = torch.device("cpu")
        self.proj = self.proj.cpu()
        self.stride = self.stride.cpu()
        try:
            out = list(_orig(self, preds_cpu, batch_cpu, **kw))
        finally:
            self.device, self.proj, self.stride = old

        # loss / loss_items 搬回显卡，trainer 那边的设备期望不变
        out[1] = out[1].to(dev)
        out[2] = {k: (v.to(dev) if torch.is_tensor(v) else v)
                  for k, v in out[2].items()}
        return tuple(out)

    _loss_mod.v8DetectionLoss.get_assigned_targets_and_loss = mark(patched)
