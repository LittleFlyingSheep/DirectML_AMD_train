# -*- coding: utf-8 -*-
"""数值对齐自检：同一份权重、同一份输入，比较 CPU 与 DirectML 的梯度。

这是这个库最重要的一道保险 —— DirectML 的差异大多是「前向对、反向错」，
光看 loss 数值发现不了（loss 数值两边一模一样），只有逐参数比对梯度才能暴露。

用法::

    python -m directml_amd_train doctor
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = ["check_gradient_alignment"]

# 判定阈值。
#   聚合相对误差：把每个参数的梯度范数加总再比，这是语义是否一致的判据，
#                 实测在 1e-6 量级（纯浮点噪声）。
#   单参数最大相对误差：DirectML 的卷积 kernel 与 CPU 算法不同，
#                 逐元素会有 1e-3 量级噪声，放宽到 5e-2。
_AGG_TOL = 1e-2
_MAX_TOL = 5e-2


def _build_model(nc: int, seed: int = 0):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(seed)
    m = DetectionModel(cfg="yolov8s.yaml", ch=3, nc=nc, verbose=False)
    if not hasattr(m, "args") or m.args is None:
        m.args = get_cfg()
    m.train()
    return m


def _make_batch(n: int = 8, seed: int = 7):
    import torch

    torch.manual_seed(seed)
    bi = torch.zeros(n)
    bi[n // 2:] = 1.0
    cls = torch.randint(0, 2, (n,)).float()
    bb = torch.rand(n, 4) * 0.4 + 0.1
    return {"batch_idx": bi, "cls": cls, "bboxes": bb}


def _run(model, x, batch, device):
    from ultralytics.utils.loss import v8DetectionLoss

    model.zero_grad(set_to_none=True)
    preds = model(x.to(device))
    if isinstance(preds, tuple):
        preds = preds[1] if isinstance(preds[1], dict) else preds[0]
    crit = v8DetectionLoss(model)
    loss, _items = crit(preds, {k: v.to(device) for k, v in batch.items()})
    loss.sum().backward()
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().float().cpu().clone()
    return [float(v) for v in loss.detach().tolist()], grads


def check_gradient_alignment(verbose: bool = True, nc: int = 2,
                             imgsz: int = 320, batch: int = 2) -> Dict[str, Any]:
    """跑一遍 CPU / DirectML 对照，返回逐项结果。

    返回的字典里 ok 为 True 表示梯度对齐（这是训练能收敛的前提）。
    """
    import torch

    from ._patches import enable_directml_amd

    enable_directml_amd(verbose=False)

    torch.manual_seed(3)
    x_cpu = torch.rand(batch, 3, imgsz, imgsz)
    batch_cpu = _make_batch()

    model = _build_model(nc)
    cpu_loss, cpu_grads = _run(model, x_cpu, batch_cpu, torch.device("cpu"))

    model = model.to(torch.device("privateuseone:0"))
    dml_loss, dml_grads = _run(model, x_cpu, batch_cpu, torch.device("privateuseone:0"))

    missing = [k for k in cpu_grads if k not in dml_grads]
    zero_on_dml = []
    per_param = []
    agg_cpu = 0.0
    agg_dml = 0.0
    worst = 0.0
    worst_name = ""
    for name, g in cpu_grads.items():
        gd = dml_grads.get(name)
        if gd is None:
            continue
        s_cpu = float(g.abs().sum())
        s_dml = float(gd.abs().sum())
        agg_cpu += s_cpu
        agg_dml += s_dml
        denom = max(float(g.abs().max()), 1e-12)
        rel = float((g - gd).abs().max()) / denom
        if rel > worst:
            worst, worst_name = rel, name
        per_param.append((name, s_cpu, s_dml, rel))
        if s_cpu > 0 and s_dml == 0.0:
            zero_on_dml.append(name)

    agg_rel = abs(agg_cpu - agg_dml) / max(agg_cpu, 1e-12)

    # 分类分支单独看一眼（历史上就是它挂的）
    cls_names = [k for k in cpu_grads if ".cv3." in k]
    cls_nonzero = sum(1 for k in cls_names
                      if float(dml_grads.get(k, torch.zeros(1)).abs().sum()) > 0)

    result = {
        "ok": (not zero_on_dml) and (not missing) and agg_rel < _AGG_TOL and worst < _MAX_TOL,
        "cpu_loss": cpu_loss,
        "dml_loss": dml_loss,
        "param_count": len(cpu_grads),
        "cls_param_count": len(cls_names),
        "cls_nonzero_on_dml": cls_nonzero,
        "agg_rel": agg_rel,
        "agg_cpu": agg_cpu,
        "agg_dml": agg_dml,
        "worst_rel": worst,
        "worst_name": worst_name,
        "zero_on_dml": zero_on_dml[:10],
        "missing": missing[:10],
    }

    if verbose:
        print("[自检] CPU  loss = %s" % ["%.6f" % v for v in cpu_loss])
        print("[自检] DML  loss = %s" % ["%.6f" % v for v in dml_loss])
        print("[自检] 参数梯度 %d 个，分类分支 %d 个（DirectML 上非零 %d 个）"
              % (len(cpu_grads), len(cls_names), cls_nonzero))
        print("[自检] 梯度范数合计 CPU %.6g / DML %.6g，聚合相对误差 %.3e"
              % (agg_cpu, agg_dml, agg_rel))
        print("[自检] 单参数最大相对误差 %.3e（%s）" % (worst, worst_name))
        if zero_on_dml:
            print("[自检] DirectML 上梯度被清零的参数: %s" % zero_on_dml[:5])
        print("[自检] 结论: %s" % ("梯度对齐，训练可收敛" if result["ok"]
                                 else "梯度未对齐，训练不会收敛"))
    return result
