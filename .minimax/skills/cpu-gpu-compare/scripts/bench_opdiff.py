# -*- coding: utf-8 -*-
"""补丁 12 开/关 的 aten 算子直方图对比：它到底把哪些算子换成了哪些。

这是回答「补丁到底改了什么」最直接的证据：同样一份输入，两个模式各跑一遍，
把 TorchDispatchMode 抓到的 aten 算子调用次数做差。
实测（ultralytics 8.4.213）：补丁把损失前向里的 13 个 aten::index 改写成
12 个 gather（并行 kernel），模型前向一个算子都没碰。

用法:
    python bench_opdiff.py {dml_flat|dml_noflat}
输出:
    RESULT {json}     含 model_fwd / loss_fwd / bwd 三个算子直方图
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

MODE = sys.argv[1] if len(sys.argv) > 1 else "dml_flat"

import torch

from Direct_AMD_train import enable

enable(framework="torch", verbose=False, flat_index=(MODE == "dml_flat"))

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from torch.utils._python_dispatch import TorchDispatchMode

DEV = torch.device("privateuseone:0")
IMGSZ, BATCH = 512, 8


class Cnt(TorchDispatchMode):
    """统计 aten 算子调用次数。"""

    def __init__(self):
        super().__init__()
        self.h = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = getattr(func, "overloadpacket", None)
        name = op.__name__ if op is not None else str(func)
        self.h[name] = self.h.get(name, 0) + 1
        return out


def build_model():
    torch.manual_seed(0)
    m = DetectionModel(cfg="yolov8s.yaml", ch=3, nc=1, verbose=False)
    if getattr(m, "args", None) is None:
        m.args = get_cfg()
    m.train()
    return m


model = build_model().to(DEV)
crit = model.init_criterion()
torch.manual_seed(3)
XD = torch.rand(BATCH, 3, IMGSZ, IMGSZ).to(DEV)
BD = {k: v.to(DEV) for k, v in
      ({"batch_idx": torch.arange(BATCH) // 2, "cls": torch.zeros(BATCH),
        "bboxes": torch.rand(BATCH, 4) * 0.4 + 0.1}).items()}

model.zero_grad(set_to_none=True)
with Cnt() as c1:
    preds = model(XD)
with Cnt() as c2:
    loss, _items = crit(preds, BD)
with Cnt() as c3:
    loss.sum().backward()
float(torch.zeros(1, device=DEV)[0])

out = {"mode": MODE,
       "model_fwd": c1.h, "loss_fwd": c2.h, "bwd": c3.h,
       "model_fwd_total": sum(c1.h.values()),
       "loss_fwd_total": sum(c2.h.values()),
       "bwd_total": sum(c3.h.values())}
print("RESULT " + json.dumps(out))
sys.stdout.flush()
# Windows 上 torch 的线程池在解释器退出时会死锁，直接结束进程
os._exit(0)
