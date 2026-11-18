# -*- coding: utf-8 -*-
"""命令行入口。

    python -m Direct_AMD_train doctor              # 环境信息 + 数值对齐自检
    python -m Direct_AMD_train probe               # 算子能力探测
    python -m Direct_AMD_train patches             # 补丁清单
    python -m Direct_AMD_train train --data data.yaml --epochs 25
    python -m Direct_AMD_train export --weights runs/exp/weights/best.pt --out out/
"""
from __future__ import annotations

import argparse
import json
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


def _cmd_probe(args) -> int:
    from .probe import probe_operators

    data = probe_operators(native=args.native, verbose=not args.json)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _cmd_patches(args) -> int:
    from .registry import format_patches

    print(format_patches(layer=args.layer))
    return 0


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
        prog="Direct_AMD_train",
        description="让 PyTorch / ultralytics 在 AMD 显卡（DirectML / DirectX 12）上正常训练")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="环境信息 + 梯度数值对齐自检")
    d.set_defaults(func=_cmd_doctor)

    pr = sub.add_parser("probe", help="DirectML 算子能力探测")
    pr.add_argument("--native", action="store_true",
                    help="测原生能力（不打补丁；已知会崩进程的算子会跳过）")
    pr.add_argument("--json", action="store_true", help="输出 JSON")
    pr.set_defaults(func=_cmd_probe)

    pa = sub.add_parser("patches", help="补丁清单")
    pa.add_argument("--layer", choices=("torch", "ultralytics"), default=None,
                    help="只看某一层")
    pa.set_defaults(func=_cmd_patches)

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
