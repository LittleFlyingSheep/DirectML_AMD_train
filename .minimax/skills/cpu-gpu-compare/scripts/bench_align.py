# -*- coding: utf-8 -*-
"""数值对齐检查：CPU vs A卡(DirectML)，以及 DirectML 上补丁12 开/关。

在同一个进程里跑三遍：
  1) CPU                （纯 CPU，不打补丁）
  2) A卡 + 补丁12 开
  3) A卡 + 补丁12 关    （把 Tensor.__getitem__ 换回原生实现）
逐参数比对梯度。

**顺序陷阱**：原生 `Tensor.__getitem__` 必须在 `enable()` **之前**捕获。
补丁 12 是通过替换 `Tensor.__getitem__` 生效的
（见 `_patches/flat_index.py` 的 `patch_flat_index`），
如果在 enable() 之后才捕获，拿到的是打过补丁的版本，
再"还原"回去等于什么都没关 —— 第 3 遍其实还是打了补丁，
比对结果必然是"逐位相同"，成了自己跟自己比。

用法:
    python bench_align.py
输出:
    RESULT {json}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ENVS = os.path.join(REPO, "_debug", "envs")
for sub in reversed(("yolo_dml_env", "yolo_train_env")):
    p = os.path.join(ENVS, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)
sys.path.insert(0, REPO)

import torch

# 必须先捕获原生实现，再打补丁 —— 见文件头的说明。
_orig_getitem = torch.Tensor.__getitem__

from Direct_AMD_train import enable

enable(framework="torch", verbose=False, flat_index=True)

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

IMGSZ = 512
BATCH = 8


def build_model():
    torch.manual_seed(0)
    m = DetectionModel(cfg="yolov8s.yaml", ch=3, nc=1, verbose=False)
    if getattr(m, "args", None) is None:
        m.args = get_cfg()
    m.train()
    return m


def make_batch(n=BATCH):
    torch.manual_seed(7)
    bi = torch.arange(n) // 2
    cls = torch.zeros(n)
    bb = torch.rand(n, 4) * 0.4 + 0.1
    return {"batch_idx": bi, "cls": cls, "bboxes": bb}


def run(dev):
    model = build_model().to(dev)
    crit = model.init_criterion()
    torch.manual_seed(3)
    x = torch.rand(BATCH, 3, IMGSZ, IMGSZ).to(dev)
    b = {k: v.to(dev) for k, v in make_batch().items()}
    model.zero_grad(set_to_none=True)
    preds = model(x)
    loss, items = crit(preds, b)
    lv = [float(v) for v in loss.detach().flatten().tolist()]
    it = ({k: float(v) for k, v in items.items()} if isinstance(items, dict)
          else ([float(v) for v in items.detach().flatten().tolist()] if torch.is_tensor(items) else []))
    loss.sum().backward()
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().float().cpu().clone()
    return lv, it, grads, str(loss.device), model


out = {}

lv_cpu, it_cpu, g_cpu, dev_cpu, model = run(torch.device("cpu"))
out["cpu_loss"] = lv_cpu
out["cpu_loss_items"] = it_cpu
out["cpu_loss_device"] = dev_cpu
out["cpu_param_count"] = len(g_cpu)

lv_flat, it_flat, g_flat, dev_flat, _ = run(torch.device("privateuseone:0"))
out["dml_flat_loss"] = lv_flat
out["dml_flat_loss_items"] = it_flat
out["dml_flat_loss_device"] = dev_flat

# 关掉补丁 12：换回**原生** Tensor.__getitem__（在 enable() 之前捕获的那份）
torch.Tensor.__getitem__ = _orig_getitem
lv_nf, it_nf, g_nf, dev_nf, _ = run(torch.device("privateuseone:0"))
out["dml_noflat_loss"] = lv_nf
out["dml_noflat_loss_items"] = it_nf
out["dml_noflat_loss_device"] = dev_nf


def cmp(a, b):
    """逐参数比对两组梯度。"""
    same = 0
    diff = 0
    worst = 0.0
    worst_name = ""
    agg_a = agg_b = 0.0
    n_diff_elems = 0
    for name, ga in a.items():
        gb = b.get(name)
        if gb is None:
            diff += 1
            continue
        agg_a += float(ga.abs().sum())
        agg_b += float(gb.abs().sum())
        d = (ga - gb).abs()
        ndiff = int((d > 0).sum())
        n_diff_elems += ndiff
        if ndiff == 0:
            same += 1
        else:
            diff += 1
        denom = max(float(ga.abs().max()), 1e-12)
        rel = float(d.max()) / denom
        if rel > worst:
            worst, worst_name = rel, name
    return {
        "params_compared": len(a),
        "params_bit_identical": same,
        "params_differing": diff,
        "differing_elements": n_diff_elems,
        "agg_l1_a": agg_a,
        "agg_l1_b": agg_b,
        "agg_rel_err": abs(agg_a - agg_b) / max(agg_a, 1e-12),
        "max_param_rel_err": worst,
        "worst_param": worst_name,
    }


out["cmp_dml_flat_vs_noflat"] = cmp(g_flat, g_nf)
out["cmp_dml_flat_vs_noflat"]["loss_abs_diff"] = max(
    abs(x - y) for x, y in zip(lv_flat, lv_nf))
out["cmp_cpu_vs_dml_flat"] = cmp(g_cpu, g_flat)
out["cmp_cpu_vs_dml_flat"]["loss_abs_diff"] = max(
    abs(x - y) for x, y in zip(lv_cpu, lv_flat))
out["cmp_cpu_vs_dml_flat"]["loss_rel_diff"] = max(
    abs(x - y) / max(abs(x), 1e-12) for x, y in zip(lv_cpu, lv_flat))
out["cmp_cpu_vs_dml_noflat"] = cmp(g_cpu, g_nf)
out["cmp_cpu_vs_dml_noflat"]["loss_abs_diff"] = max(
    abs(x - y) for x, y in zip(lv_cpu, lv_nf))

print("RESULT " + json.dumps(out))
sys.stdout.flush()
# Windows 上 torch 的线程池在解释器退出时会死锁，直接结束进程
os._exit(0)
