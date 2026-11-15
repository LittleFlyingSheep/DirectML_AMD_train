# -*- coding: utf-8 -*-
"""命令行入口。

    python -m directml_amd_train doctor              # 环境信息 + 数值对齐自检
    python -m directml_amd_train train --data data.yaml --epochs 25
    python -m directml_amd_train export --weights runs/exp/weights/best.pt --out out/
"""
from __future__ import annotations

import argparse
import sys


def _cmd_doctor(args) -> int:
    from .trainer import environment_info

    info = environment_info()
    print("=== 环境 ===")
    for k, v in info.items():
        print("  %-14s %s" % (k, v))

    print()
    print("=== 数值对齐自检（CPU vs DirectML）===")
    from .selftest import check_gradient_alignment
    res = check_gradient_alignment(verbose=True)
    return 0 if res.get("ok") else 1


def _cmd_train(args) -> int:
    from .trainer import train_yolo

    train_yolo(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
    )
    return 0


def _cmd_export(args) -> int:
    from .trainer import export_onnx

    export_onnx(args.weights, args.out, imgsz=args.imgsz,
                opset=args.opset, simplify=not args.no_simplify)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="directml_amd_train",
        description="让 ultralytics YOLO 在 AMD 显卡（DirectML / DirectX 12）上训练")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="环境信息 + 梯度数值对齐自检")
    d.set_defaults(func=_cmd_doctor)

    t = sub.add_parser("train", help="训练 YOLO")
    t.add_argument("--data", required=True, help="data.yaml 或它所在的目录")
    t.add_argument("--model", default="yolov8s.pt", help="起始权重")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--imgsz", type=int, default=512)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--device", default="dml", help="dml（AMD 显卡）| cpu | 其它 ultralytics 写法")
    t.add_argument("--project", default=None, help="输出根目录")
    t.add_argument("--name", default="exp")
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("export", help="导出 ONNX")
    e.add_argument("--weights", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--imgsz", type=int, default=512)
    e.add_argument("--opset", type=int, default=12)
    e.add_argument("--no-simplify", action="store_true")
    e.set_defaults(func=_cmd_export)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
