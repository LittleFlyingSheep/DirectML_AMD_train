# -*- coding: utf-8 -*-
"""算子能力探测：把「DirectML 上哪些算子能用」变成一张可复现的表。

DirectML 不是 CUDA，算子支持是「打折」的，而且缺口往往很隐蔽
（前向对、反向错；或者干脆把进程打崩）。
这个模块把常用算子挨个跑一遍，出一张能力表：

    python -m Direct_AMD_train probe            # 打补丁后的能力
    python -m Direct_AMD_train probe --native   # 原生能力（不补丁）
    python -m Direct_AMD_train probe --json

换 torch-directml 版本时跑一遍，就知道有没有新的坑。
"""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["probe_operators", "format_probe", "OPERATOR_GROUPS"]

OPERATOR_GROUPS = ("逐元素", "归约", "索引/散射", "矩阵/卷积", "归一化/损失",
                 "形状/插值", "搬运/同步", "ultralytics")


def _cases():
    import torch
    import torch.nn.functional as F

    def rnd(*shape, device=None):
        return torch.randn(*shape, device=device)

    C = []

    def add(group, name, fn, dangerous=False, note=""):
        C.append({"group": group, "name": name, "fn": fn,
                  "dangerous": dangerous, "note": note})

    # ---------------- 逐元素 ----------------
    g = "逐元素"
    add(g, "add", lambda d: rnd(4, 4, device=d) + rnd(4, 4, device=d))
    add(g, "mul", lambda d: rnd(4, 4, device=d) * rnd(4, 4, device=d))
    add(g, "sub", lambda d: rnd(4, 4, device=d) - rnd(4, 4, device=d))
    add(g, "div", lambda d: rnd(4, 4, device=d) / rnd(4, 4, device=d))
    add(g, "pow", lambda d: rnd(4, 4, device=d) ** 2)
    add(g, "sqrt", lambda d: torch.sqrt(rnd(4, 4, device=d).abs()))
    add(g, "exp", lambda d: torch.exp(rnd(4, 4, device=d)))
    add(g, "log", lambda d: torch.log(rnd(4, 4, device=d).abs() + 1))
    add(g, "sigmoid", lambda d: torch.sigmoid(rnd(4, 4, device=d)))
    add(g, "tanh", lambda d: torch.tanh(rnd(4, 4, device=d)))
    add(g, "gelu", lambda d: F.gelu(rnd(4, 4, device=d)))
    add(g, "silu", lambda d: F.silu(rnd(4, 4, device=d)))
    add(g, "clamp", lambda d: torch.clamp(rnd(4, 4, device=d), -1, 1))
    add(g, "where", lambda d: torch.where(rnd(4, 4, device=d) > 0,
                                           rnd(4, 4, device=d), rnd(4, 4, device=d)))
    add(g, "masked_fill", lambda d: rnd(4, 4, device=d).masked_fill(
        torch.tensor([True, False, True, False], device=d), 0.0))

    # ---------------- 归约 ----------------
    g = "归约"
    add(g, "sum", lambda d: rnd(4, 4, device=d).sum())
    add(g, "mean", lambda d: rnd(4, 4, device=d).mean())
    add(g, "max", lambda d: rnd(4, 4, device=d).max())
    add(g, "min", lambda d: rnd(4, 4, device=d).min())
    add(g, "amax", lambda d: rnd(4, 4, device=d).amax(1))
    add(g, "argmax", lambda d: rnd(4, 4, device=d).argmax(1))
    add(g, "any", lambda d: (rnd(4, 4, device=d) > 0).any())
    add(g, "all", lambda d: (rnd(4, 4, device=d) > 0).all())
    add(g, "std", lambda d: rnd(4, 4, device=d).std(), note="CPU 回落")
    add(g, "var", lambda d: rnd(4, 4, device=d).var(), note="CPU 回落")
    add(g, "cumsum", lambda d: rnd(4, 4, device=d).cumsum(1))
    add(g, "norm", lambda d: rnd(4, 4, device=d).norm())

    # ---------------- 索引 / 散射 ----------------
    g = "索引/散射"
    def _idx(d):
        return torch.tensor([0, 2], device=d)

    add(g, "index_select", lambda d: rnd(4, 4, device=d).index_select(0, _idx(d)))
    add(g, "gather", lambda d: rnd(4, 4, device=d).gather(
        1, torch.tensor([[0, 1], [2, 3]], device=d)))
    add(g, "scatter_add_", lambda d: rnd(4, device=d).scatter_add_(
        0, _idx(d), rnd(2, device=d)))
    add(g, "index_put_", lambda d: rnd(4, 4, device=d).clone().index_put_(
        (_idx(d),), rnd(2, 4, device=d)))
    add(g, "masked_select", lambda d: rnd(4, 4, device=d)[rnd(4, 4, device=d) > 0])
    add(g, "nonzero", lambda d: (rnd(4, 4, device=d) > 0).nonzero())
    add(g, "unique(1维)", lambda d: torch.unique(rnd(8, device=d)))
    add(g, "unique(2维)", lambda d: torch.unique(rnd(4, 4, device=d)),
        dangerous=True, note="原生会崩进程")
    add(g, "unique(return_counts)", lambda d: torch.unique(rnd(8, device=d),
        return_counts=True))
    add(g, "take", lambda d: rnd(4, 4, device=d).take(_idx(d)), note="CPU 回落")
    add(g, "index_copy_", lambda d: rnd(4, 4, device=d).clone().index_copy_(
        0, _idx(d), rnd(2, 4, device=d)))
    add(g, "index_fill_", lambda d: rnd(4, 4, device=d).clone().index_fill_(
        0, _idx(d), 1.0))
    add(g, "tril", lambda d: torch.tril(rnd(4, 4, device=d)))
    add(g, "triu", lambda d: torch.triu(rnd(4, 4, device=d)))
    add(g, "argsort", lambda d: torch.argsort(rnd(8, device=d), descending=True))
    add(g, "argsort(全等值)", lambda d: torch.argsort(
        torch.full((64,), 0.9, device=d), descending=True),
        dangerous=True, note="原生会炸")
    add(g, "sort", lambda d: torch.sort(rnd(8, device=d), descending=True))
    add(g, "topk", lambda d: torch.topk(rnd(8, device=d), 3))

    # ---------------- 矩阵 / 卷积 ----------------
    g = "矩阵/卷积"
    add(g, "matmul", lambda d: rnd(4, 4, device=d) @ rnd(4, 4, device=d))
    add(g, "mm", lambda d: torch.mm(rnd(4, 4, device=d), rnd(4, 4, device=d)))
    add(g, "addmv", lambda d: torch.addmv(rnd(4, device=d), rnd(4, 4, device=d),
        rnd(4, device=d)), note="CPU 回落")
    add(g, "bmm", lambda d: torch.bmm(rnd(2, 3, 4, device=d), rnd(2, 4, 5, device=d)))
    add(g, "einsum", lambda d: torch.einsum("ij,jk->ik", rnd(4, 4, device=d),
        rnd(4, 4, device=d)))
    add(g, "linear", lambda d: F.linear(rnd(4, 4, device=d), rnd(5, 4, device=d)))
    add(g, "conv1d", lambda d: F.conv1d(rnd(1, 3, 8, device=d), rnd(4, 3, 3, device=d)))
    add(g, "conv2d", lambda d: F.conv2d(rnd(1, 3, 8, 8, device=d),
        rnd(6, 3, 3, 3, device=d)))
    add(g, "conv3d", lambda d: F.conv3d(rnd(1, 3, 4, 4, 4, device=d),
        rnd(4, 3, 3, 3, 3, device=d)), note="只支持 4 维输入")

    # ---------------- 归一化 / 损失 ----------------
    g = "归一化/损失"
    add(g, "layer_norm", lambda d: F.layer_norm(rnd(4, 4, device=d), (4,)))
    add(g, "group_norm", lambda d: F.group_norm(rnd(4, 4, device=d), 2))
    add(g, "softmax", lambda d: torch.softmax(rnd(4, 4, device=d), 1))
    add(g, "log_softmax", lambda d: torch.log_softmax(rnd(4, 4, device=d), 1))
    add(g, "cross_entropy", lambda d: F.cross_entropy(rnd(4, 4, device=d),
        torch.randint(0, 4, (4,), device=d)))
    add(g, "nll_loss", lambda d: F.nll_loss(torch.log_softmax(rnd(4, 4, device=d), 1),
        torch.randint(0, 4, (4,), device=d)))
    add(g, "bce_with_logits", lambda d: F.binary_cross_entropy_with_logits(
        rnd(4, 4, device=d), torch.randint(0, 2, (4, 4), device=d).float()),
        note="CPU 回落")
    add(g, "smooth_l1", lambda d: F.smooth_l1_loss(rnd(4, 4, device=d),
        torch.randint(0, 2, (4, 4), device=d).float()))
    add(g, "mse", lambda d: F.mse_loss(rnd(4, 4, device=d),
        torch.randint(0, 2, (4, 4), device=d).float()))
    add(g, "kl_div", lambda d: F.kl_div(torch.log_softmax(rnd(4, 4, device=d), 1),
        torch.softmax(rnd(4, 4, device=d), 1)))
    add(g, "cosine_similarity", lambda d: F.cosine_similarity(rnd(4, 4, device=d),
        torch.randint(0, 2, (4, 4), device=d).float()))

    # ---------------- 形状 / 插值 ----------------
    g = "形状/插值"
    add(g, "reshape", lambda d: rnd(4, 4, device=d).reshape(2, 8))
    add(g, "permute", lambda d: rnd(4, 4, device=d).permute(1, 0))
    add(g, "transpose", lambda d: rnd(4, 4, device=d).t())
    add(g, "contiguous", lambda d: rnd(4, 4, device=d).t().contiguous())
    add(g, "expand", lambda d: rnd(4, 4, device=d)[:1].expand(4, 4))
    add(g, "cat", lambda d: torch.cat((rnd(4, 4, device=d), rnd(4, 4, device=d)), 1))
    add(g, "cat(带 0 列)", lambda d: torch.cat((rnd(4, 4, device=d), rnd(4, 0, device=d),
        rnd(4, 4, device=d)), 1), dangerous=True, note="原生会炸")
    add(g, "stack", lambda d: torch.stack((rnd(4, 4, device=d), rnd(4, 4, device=d))))
    add(g, "split", lambda d: torch.split(rnd(4, 4, device=d), 2, 1))
    add(g, "tile", lambda d: rnd(4, 4, device=d).tile((2, 1)))
    add(g, "flip", lambda d: rnd(4, 4, device=d).flip(0))
    add(g, "roll", lambda d: rnd(4, 4, device=d).roll(1, 0), note="CPU 回落")
    add(g, "pad", lambda d: F.pad(rnd(4, 4, device=d), (1, 1)))
    add(g, "interpolate(nearest)", lambda d: F.interpolate(rnd(1, 3, 8, 8, device=d),
        scale_factor=2, mode="nearest"))
    add(g, "interpolate(bilinear)", lambda d: F.interpolate(rnd(1, 3, 8, 8, device=d),
        scale_factor=2, mode="bilinear", align_corners=False))
    add(g, "max_pool2d", lambda d: F.max_pool2d(rnd(1, 3, 8, 8, device=d), 2))
    add(g, "avg_pool2d", lambda d: F.avg_pool2d(rnd(1, 3, 8, 8, device=d), 2))
    add(g, "adaptive_avg_pool2d", lambda d: F.adaptive_avg_pool2d(
        rnd(1, 3, 8, 8, device=d), 1))
    add(g, "arange", lambda d: torch.arange(4, device=d))
    add(g, "meshgrid", lambda d: torch.meshgrid(torch.arange(4, device=d),
        torch.arange(4, device=d)))
    add(g, "full_like", lambda d: torch.full_like(rnd(4, 4, device=d), 1.0))
    add(g, "to(float16)", lambda d: rnd(4, 4, device=d).to(torch.float16))

    # ---------------- 搬运 / 同步 ----------------
    g = "搬运/同步"
    add(g, "item", lambda d: rnd(4, 4, device=d)[0, 0].item())
    add(g, "cpu", lambda d: rnd(4, 4, device=d).cpu())
    add(g, "numpy", lambda d: rnd(4, 4, device=d).cpu().numpy())
    add(g, "isnan", lambda d: torch.isnan(rnd(4, 4, device=d)))
    add(g, "allclose", lambda d: torch.allclose(rnd(4, 4, device=d), rnd(4, 4, device=d)))

    # ---------------- ultralytics 相关 ----------------
    g = "ultralytics"

    def _boxes(d):
        b = rnd(5, 4, device=d)
        b[:, 2:] += b[:, :2] + 1
        return b

    def _box_iou(d):
        from ultralytics.utils.metrics import box_iou
        return box_iou(_boxes(d), _boxes(d))

    def _rboxes(d):
        # 旋转框是 5 列 (cx, cy, w, h, angle)
        b = rnd(5, 5, device=d)
        b[:, 2:4] += 1
        return b

    def _probiou(d):
        from ultralytics.utils.metrics import batch_probiou
        return batch_probiou(_rboxes(d), _rboxes(d))

    def _nms(d):
        from ultralytics.utils.nms import TorchNMS
        b = _boxes(d)
        return TorchNMS.nms(b, b[:, 0].contiguous(), 0.45)

    add(g, "box_iou", _box_iou)
    add(g, "batch_probiou(OBB)", _probiou, dangerous=True, note="原生会炸")
    add(g, "TorchNMS.nms", _nms)

    return C


OPERATOR_CASES = None  # 延迟构造（要 import torch）


def probe_operators(native: bool = False, verbose: bool = True,
                   device: str = "privateuseone:0",
                   only=None) -> Dict[str, Any]:
    """跑一遍算子能力探测，返回结构化结果。

    native=True 时**不打补丁**，测的是 DirectML 的原生能力。
    原生模式跳过已知可能导致进程崩溃的用例，须在未启用补丁的进程中运行。

    only 可以只跑指定分组（分组名见 OPERATOR_GROUPS），
    用来做快速抽查。
    """
    import torch
    from ._patches import _ultralytics_installed, enable
    from ._patches._once import already_patched
    from .trainer import resolve_device

    dev = torch.device(resolve_device(device))
    cases = _cases()
    if only:
        keep = set(only)
        cases = [c for c in cases if c["group"] in keep]
    has_ultralytics = any(c["group"] == "ultralytics" for c in cases) and _ultralytics_installed()
    if native and already_patched(torch.inference_mode):
        raise RuntimeError("请在未调用 enable() 的新进程中探测原生能力")
    if not native and dev.type == "privateuseone":
        enable(framework="ultralytics" if has_ultralytics else "torch", verbose=False)

    results: List[Dict[str, Any]] = []
    torch.manual_seed(0)

    for c in cases:
        row = {"group": c["group"], "name": c["name"],
               "ok": None, "error": "", "note": c["note"],
               "skipped": False}
        if c["group"] == "ultralytics" and not has_ultralytics:
            row["skipped"] = True
            row["note"] = "未安装 ultralytics"
        elif c["dangerous"] and native and dev.type == "privateuseone":
            row["skipped"] = True
            row["note"] = (c["note"] + "；原生模式下跳过").lstrip("；")
        if row["skipped"]:
            results.append(row)
            continue
        try:
            output = c["fn"](dev)
            for value in output if isinstance(output, tuple) else (output,):
                if isinstance(value, torch.Tensor):
                    value.cpu()  # 等待异步设备计算完成后再报告成功
            row["ok"] = True
        except Exception as e:
            row["ok"] = False
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:100])
        results.append(row)

    ok = sum(1 for r in results if r["ok"])
    bad = [r for r in results if r["ok"] is False]
    out = {"native": bool(native), "device": str(dev),
           "total": len(results), "ok": ok, "failed": len(bad),
           "results": results}

    if verbose:
        print(format_probe(out))
    return out


def format_probe(data: Dict[str, Any]) -> str:
    """把探测结果渲染成一张表。"""
    lines = []
    title = "原生能力（未打补丁）" if data.get("native") else "补丁后能力"
    lines.append("=== DirectML 算子能力：%s（%s）===" % (title, data.get("device")))
    lines.append("通过 %d / %d，失败 %d" % (data.get("ok", 0), data.get("total", 0),
                                    data.get("failed", 0)))
    lines.append("")
    last = None
    for r in data.get("results", []):
        if r["group"] != last:
            last = r["group"]
            lines.append("[%s]" % last)
        if r.get("skipped"):
            mark = "跳过"
        elif r["ok"]:
            mark = "可用"
        else:
            mark = "不可用"
        note = ("  <- " + r["note"]) if r.get("note") else ""
        err = ("  " + r["error"]) if r.get("error") else ""
        lines.append("  %-4s %-24s%s%s" % (mark, r["name"], note, err))
    return "\n".join(lines)
