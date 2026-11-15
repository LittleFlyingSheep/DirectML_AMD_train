# -*- coding: utf-8 -*-
"""最小示例：在 AMD 显卡（DirectML）上训练 YOLO。

    python examples/train_yolo.py --data path/to/data.yaml --epochs 25

关键只有一句：enable_directml_amd() 必须在 import ultralytics 之前调用。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directml_amd_train import enable_directml_amd, train_yolo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="dml")
    ap.add_argument("--project", default=os.path.join("runs", "directml"))
    ap.add_argument("--name", default="exp")
    args = ap.parse_args()

    # 1) 先打补丁（必须早于 import ultralytics）
    enable_directml_amd()

    # 2) 再训练
    train_yolo(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
