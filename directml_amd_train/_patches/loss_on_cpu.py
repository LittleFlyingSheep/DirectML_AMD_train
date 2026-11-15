# -*- coding: utf-8 -*-
"""补丁 11（最重要的一条）：DirectML 上把「损失计算」搬到 CPU 上做。

模型的前向/反向仍然留在显卡上，只有损失那点计算搬到 CPU。

=============================================================================
这就是「A 卡训练跑得完却不收敛」的根本原因
=============================================================================

实测（torch-directml 0.2.5.dev240914，同一份输入、同一段代码）：

    x = torch.zeros(3, requires_grad=True)
    y = x.clone()
    y[1] = 1.0                 # 覆盖第 1 个槽位
    y.sum().backward()
    CPU  -> x.grad = [1, 0, 1]  被覆盖的槽位 1 不该再往回传，正确
    DML  -> x.grad = [0, 1, 1]  清零的成了槽位 0，槽位 1 反而留着

也就是说 DirectML 的 index_put_（原地槽位赋值）反向算错了：
「被覆盖位置的梯度」没有清掉，而是清掉了旁边一个位置。
前向是对的（loss 数值和 CPU 一模一样），所以训练不会报错，
只是梯度悄悄错位 —— 这是最难查的一类 bug。

而 ultralytics 的损失恰恰就是这么拼装的（ultralytics/utils/loss.py）：

    loss = torch.zeros(3, device=self.device)
    loss[1] = bce_loss.sum() / target_scores_sum   # 分类
    loss[0], loss[2] = self.bbox_loss(...)          # 回归 / DFL
    loss[0] *= self.hyp.box
    loss[1] *= self.hyp.cls
    loss[2] *= self.hyp.dfl

结果：分类分支（Detect 的 cv3）拿到的梯度恒为 0，
cls_loss 永远降不下来，模型学不会分类 ——
A 卡 25 轮跑完 mAP50 只有 0.60，而 CPU 同样参数是 0.99。

修法：损失那点计算量（几个小张量的乘加）相对卷积主干可以忽略，
干脆把损失整个搬到 CPU 上用原生 PyTorch 算，梯度再原路搬回显卡。
这样绕开了 DirectML 原地写入的反向差异，而模型的前向/反向
仍然全程跑在显卡上 —— 训练确实是在显卡上做的，不是「显卡跑一半」。

修完之后，A 卡与 CPU 的梯度逐项对齐（同一份权重、同一份输入）：

    cv3（分类分支）24/24 个参数梯度非零，两边 max/sum 完全一致
    cv2（回归分支）65/81 非零，max 9257.61 vs 9257.6
    其它参数            78/78 非零，max 30312.4 vs 30312.3
    scores 张量钩子梯度  1.2573893 vs 1.2573889

A 卡 25 轮最终验证：P 0.998 / R 1.000 / mAP50 0.995 / mAP50-95 0.983
（CPU 同样参数：P 0.998 / R 1.000 / mAP50 0.995 / mAP50-95 0.985）
"""
from __future__ import annotations


def patch_loss_on_cpu() -> None:
    """把 v8DetectionLoss 的损失计算搬到 CPU（模型前后向不变）。"""
    import torch
    from ultralytics.utils import loss as _loss_mod

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

    _loss_mod.v8DetectionLoss.get_assigned_targets_and_loss = patched
