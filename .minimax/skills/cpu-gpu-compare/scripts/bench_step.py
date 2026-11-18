# -*- coding: utf-8 -*-
"""真实训练循环里的每步耗时（不是合成的 microbenchmark）。

用 ultralytics 的 on_train_batch_start / on_train_batch_end 回调量整个训练步，
再包一层 v8DetectionLoss.loss 量损失前向。测的是真实数据（含 mosaic 增广），
比合成 batch 更接近训练时的实际情况。

用法:
    python bench_step.py {dml_flat|dml_noflat|cpu} [epochs]

输出:
    RESULT {json}     含 loss_fwd_median / step_median / step_all / loss_fwd_all

注意:
* cpu 模式用 framework="none"（纯 PyTorch 基线，不打补丁）；
  dml 模式用 framework="ultralytics"（补丁全开，flat_index 决定补丁 12 的开关）。
* DirectML 的命令队列在连续反复算同一个损失时会卡死，本脚本走真实训练循环
  （迭代之间有数据加载等 CPU 工作），训练循环本身不受影响。
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ENVS = os.path.join(REPO, "_debug", "envs")
# 顺序要紧：yolo_dml_env 必须排在 yolo_train_env 前面（torch 版本要对得上
# torch_directml_native 的编译版本，否则 DLL 加载失败）。
for sub in reversed(("yolo_dml_env", "yolo_train_env")):
    p = os.path.join(ENVS, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)
sys.path.insert(0, REPO)

MODE = sys.argv[1] if len(sys.argv) > 1 else "dml_flat"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

from Direct_AMD_train import enable

if MODE == "cpu":
    enable(framework="none", verbose=False)
else:
    enable(framework="ultralytics", verbose=False, flat_index=(MODE == "dml_flat"))

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

stats = {"loss_fwd": [], "step": []}

_orig_loss = v8DetectionLoss.loss


def timed_loss(self, preds, batch):
    t0 = time.perf_counter()
    out = _orig_loss(self, preds, batch)
    stats["loss_fwd"].append(time.perf_counter() - t0)
    return out


v8DetectionLoss.loss = timed_loss


def on_start(trainer):
    trainer._t_step = time.perf_counter()


def on_end(trainer):
    stats["step"].append(time.perf_counter() - trainer._t_step)


DEV = "cpu" if MODE == "cpu" else "privateuseone:0"
model = YOLO(os.path.join(REPO, "yolov8s.pt"))
model.add_callback("on_train_batch_start", on_start)
model.add_callback("on_train_batch_end", on_end)
t0 = time.perf_counter()
model.train(data=os.path.join(REPO, "_debug", "data", "yolo_dataset", "data_local.yaml"),
            epochs=EPOCHS, imgsz=512, batch=8, device=DEV, workers=0,
            project=os.path.join(REPO, "_debug", "runs"), name="eval_step_%s" % MODE,
            exist_ok=True, seed=0, deterministic=True, patience=100,
            plots=False, close_mosaic=10, amp=False, verbose=False)
wall = time.perf_counter() - t0

out = {
    "mode": MODE,
    "epochs": EPOCHS,
    "wall_s": round(wall, 2),
    "n_steps": len(stats["step"]),
    "step_median": statistics.median(stats["step"]) if stats["step"] else None,
    "step_mean": statistics.mean(stats["step"]) if stats["step"] else None,
    "step_p90": sorted(stats["step"])[int(len(stats["step"]) * 0.9)] if stats["step"] else None,
    "loss_fwd_median": statistics.median(stats["loss_fwd"]) if stats["loss_fwd"] else None,
    "loss_fwd_mean": statistics.mean(stats["loss_fwd"]) if stats["loss_fwd"] else None,
    "step_all": [round(v, 4) for v in stats["step"]],
    "loss_fwd_all": [round(v, 4) for v in stats["loss_fwd"]],
}
print("RESULT " + json.dumps(out))
sys.stdout.flush()
# Windows 上 torch 的线程池在解释器退出时会死锁，直接结束进程
os._exit(0)
