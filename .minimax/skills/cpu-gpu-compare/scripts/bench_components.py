# -*- coding: utf-8 -*-
"""组件级耗时：CPU vs A 卡（DirectML）。

用法：
    python bench_components.py cpu        # CPU 一遍
    python bench_components.py dml        # A 卡一遍

必须分进程跑：CPU 多线程会把同进程里后面的 DML 计时拖慢。
每次迭代后会把显卡命令队列排空（DirectML 反复算损失会卡队列，
原生 aten::index 路径同样卡，是后端既有问题），排空不计入计时区间。

输出：RESULT <mode> loss_fwd=.. loss_bwd=.. model_fwd=.. model_bwd=.. iter=..
"""
import os
import sys
import time
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ENVS = os.path.join(REPO, "_debug", "envs")
for sub in ("yolo_dml_env", "yolo_train_env"):
    p = os.path.join(ENVS, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)
sys.path.insert(0, REPO)

MODE = sys.argv[1] if len(sys.argv) > 1 else "dml"

import torch
import torch_directml  # noqa: F401  注册 privateuseone 后端

DML = torch.device("privateuseone:0")
DEV = torch.device("cpu") if MODE == "cpu" else DML

from ultralytics.utils.loss import v8DetectionLoss
from Direct_AMD_train import selftest as st

torch.manual_seed(3)
X = torch.rand(8, 3, 512, 512)
BATCH = st._make_batch(n=8)


def drain():
    """把显卡命令队列排空（DirectML 连续反复算损失会卡队列）。"""
    if DEV.type == "privateuseone":
        float(torch.zeros(1, device=DEV)[0])


def loss_only(dev, n=4):
    """损失：前向 / 反向。"""
    model = st._build_model(2).to(dev)
    preds = model(X.to(dev))
    p = preds[1] if isinstance(preds, tuple) else preds
    crit = v8DetectionLoss(model)
    pd = {k: (v.detach().to(dev).requires_grad_(True) if torch.is_tensor(v) else v)
          for k, v in p.items()}
    bt = {k: v.to(dev) for k, v in BATCH.items()}
    for _ in range(2):                     # 预热
        loss, _ = crit(pd, bt)
        float(loss.sum())
        loss.sum().backward()
        drain()
    fs, bs_ = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        loss, _ = crit(pd, bt)
        float(loss.sum())
        t1 = time.perf_counter()
        loss.sum().backward()
        float(loss.sum())
        t2 = time.perf_counter()
        fs.append(t1 - t0)
        bs_.append(t2 - t1)
        drain()
    return statistics.mean(fs), statistics.mean(bs_)


def model_only(dev, n=3):
    """模型：前向 / 反向（不含损失）。"""
    model = st._build_model(2).to(dev)
    for _ in range(2):
        preds = model(X.to(dev))
        p = preds[1] if isinstance(preds, tuple) else preds
        p["scores"].sum().backward()
        model.zero_grad(set_to_none=True)
        drain()
    fs, bs_ = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        preds = model(X.to(dev))
        p = preds[1] if isinstance(preds, tuple) else preds
        float(p["scores"].sum())
        t1 = time.perf_counter()
        p["scores"].sum().backward()
        float(p["scores"].sum())
        t2 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        fs.append(t1 - t0)
        bs_.append(t2 - t1)
        drain()
    return statistics.mean(fs), statistics.mean(bs_)


def full_iter(dev, n=3):
    """一个 iteration：模型前向 + 损失 + 反向。"""
    model = st._build_model(2).to(dev)
    crit = v8DetectionLoss(model)
    bt = {k: v.to(dev) for k, v in BATCH.items()}
    for _ in range(1):
        preds = model(X.to(dev))
        p = preds[1] if isinstance(preds, tuple) else preds
        loss, _ = crit(p, bt)
        loss.sum().backward()
        model.zero_grad(set_to_none=True)
        drain()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        preds = model(X.to(dev))
        p = preds[1] if isinstance(preds, tuple) else preds
        loss, _ = crit(p, bt)
        loss.sum().backward()
        float(loss.sum())
        ts.append(time.perf_counter() - t0)
        model.zero_grad(set_to_none=True)
        drain()
    return statistics.mean(ts)


lf, lb = loss_only(DEV)
mf, mb = model_only(DEV)
it = full_iter(DEV)
print("RESULT %s loss_fwd=%.4f loss_bwd=%.4f model_fwd=%.4f model_bwd=%.4f iter=%.4f"
      % (MODE, lf, lb, mf, mb, it))
sys.stdout.flush()
# Windows 上 torch 的线程池在解释器退出时会死锁，直接结束进程
os._exit(0)
