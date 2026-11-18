# -*- coding: utf-8 -*-
"""端到端配对 A/B：同一台机器、同一时段，只差一个开关。

用法：
    python bench_ab.py --epochs 25 --param flat_index

顺序跑两遍训练（param=True / param=False），读 results.csv 里的总耗时，
并对比最终指标。改写类补丁的指标应当逐位相同。

    python bench_ab.py --epochs 3        # 先小跑一遍验证流程
    python bench_ab.py --epochs 25 --timeout 1800

两个必要的防护（少了会得出错误结论）：

* **超时**：DirectML 的命令队列在某些情况下会卡死，训练进程会无限挂起。
  --timeout 到点就终止子进程并如实报「超时」，不让它挂一整夜。
* **产物新鲜度**：跑之前如果同名 run 目录已存在，里面的 results.csv 是上一轮的。
  读数前会比对修改时间，只认本次运行产生的 CSV，否则报「没有本次运行的产物」。
  （中途被杀掉的那一臂就是这种情况，直接读会拿到上一轮的数当成新的。）
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PY = sys.executable


def run_one(name, param, value, epochs, data, timeout=None):
    """跑一遍训练，返回 (总秒数, 最后一行 CSV, 是否超时)。"""
    code = (
        "from Direct_AMD_train.trainer import train_yolo; "
        "train_yolo(data=%r, epochs=%d, imgsz=512, batch=8, device='dml', "
        "project=%r, name=%r, workers=0, mosaic=0.0, fliplr=0.0, scale=0.3, hsv_s=0.4, "
        "%s=%r)" % (data, epochs, os.path.join(REPO, "_debug", "runs"), name, param, value)
    )
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run([PY, "-u", "-c", code], cwd=REPO, timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = -9
        print("训练超时（%ss），已终止：%s" % (timeout, name))
    dt = time.perf_counter() - t0

    csv = os.path.join(REPO, "_debug", "runs", name, "results.csv")
    last = ""
    if os.path.isfile(csv):
        fresh = os.path.getmtime(csv) >= (t0 - 2)
        if fresh:
            with open(csv, "r", encoding="utf-8") as fh:
                rows = [ln.strip() for ln in fh if ln.strip()]
            last = rows[-1] if rows else ""
        else:
            print("没有本次运行的产物（%s 的 results.csv 是旧的，已忽略）" % name)
    if not timed_out and rc != 0:
        print("训练失败 returncode=%s" % rc)
    return dt, last, timed_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--param", default="flat_index",
                    help="被 A/B 的开关名（train_yolo 的关键字参数）")
    ap.add_argument("--data", default=os.path.join(
        REPO, "_debug", "data", "yolo_dataset", "data_local.yaml"))
    ap.add_argument("--prefix", default="ab")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="单次训练的超时秒数（DirectML 队列卡死时兜底）")
    args = ap.parse_args()

    for name, value in (("%s_on" % args.prefix, True), ("%s_off" % args.prefix, False)):
        dt, last, timed_out = run_one(name, args.param, value, args.epochs,
                                      args.data, args.timeout)
        total = last.split(",")[1] if "," in last else "?"
        flag = " TIMEOUT" if timed_out else ""
        print("RESULT %s param=%s=%s wall=%.1fs total=%s%s" % (name, args.param, value, dt, total, flag))
        print("METRICS %s %s" % (name, last))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
