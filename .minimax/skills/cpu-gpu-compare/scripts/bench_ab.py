# -*- coding: utf-8 -*-
"""端到端配对 A/B：同一台机器、同一时段，只差一个开关。

用法：
    python bench_ab.py --epochs 25 --param flat_index

顺序跑两遍训练（param=True / param=False），读 results.csv 里的总耗时，
并对比最终指标。改写类补丁的指标应当逐位相同。

    python bench_ab.py --epochs 3        # 先小跑一遍验证流程
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = sys.executable


def run_one(name, param, value, epochs, data):
    """跑一遍训练，返回 (总秒数, 最后一行 CSV)。"""
    code = (
        "from directml_amd_train.trainer import train_yolo; "
        "train_yolo(data=%r, epochs=%d, imgsz=512, batch=8, device='dml', "
        "project=%r, name=%r, workers=0, mosaic=0.0, fliplr=0.0, scale=0.3, hsv_s=0.4, "
        "%s=%r)" % (data, epochs, os.path.join(REPO, "_debug", "runs"), name, param, value)
    )
    t0 = time.perf_counter()
    proc = subprocess.run([PY, "-u", "-c", code], cwd=REPO)
    dt = time.perf_counter() - t0
    csv = os.path.join(REPO, "_debug", "runs", name, "results.csv")
    last = ""
    if os.path.isfile(csv):
        with open(csv, "r", encoding="utf-8") as fh:
            rows = [ln.strip() for ln in fh if ln.strip()]
        last = rows[-1] if rows else ""
    if proc.returncode != 0:
        print("训练失败 returncode=%s" % proc.returncode)
    return dt, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--param", default="flat_index",
                    help="被 A/B 的开关名（train_yolo 的关键字参数）")
    ap.add_argument("--data", default=os.path.join(
        REPO, "_debug", "data", "yolo_dataset", "data_local.yaml"))
    ap.add_argument("--prefix", default="ab")
    args = ap.parse_args()

    for name, value in (("%s_on" % args.prefix, True), ("%s_off" % args.prefix, False)):
        dt, last = run_one(name, args.param, value, args.epochs, args.data)
        total = last.split(",")[1] if "," in last else "?"
        print("RESULT %s param=%s=%s wall=%.1fs total=%s" % (name, args.param, value, dt, total))
        print("METRICS %s %s" % (name, last))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
