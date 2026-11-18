# -*- coding: utf-8 -*-
"""补丁注册表：每条补丁的元数据。

这个库从「给 ultralytics 打补丁」升级成「给 DirectML 后端打补丁」，
补丁也就分成了两层：

  * torch 层       —— 改 torch / torch_directml 自己的算子语义，与框架无关
  * ultralytics 层 —— 改 ultralytics 的内部行为

这里把每条补丁的症状、修法、影响范围记在一处，
`python -m Direct_AMD_train patches` 可以直接打出来看，
换 torch-directml 版本时照着这张表复核即可。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["PatchInfo", "list_patches", "format_patches", "LAYERS"]

LAYERS = ("torch", "ultralytics")


class PatchInfo(object):
    """一条补丁的元数据。"""

    __slots__ = ("name", "layer", "ops", "symptom", "fix", "tasks", "note")

    def __init__(self, name: str, layer: str, ops, symptom: str,
                 fix: str, tasks=("*",), note: str = ""):
        self.name = name
        self.layer = layer
        self.ops = tuple(ops)
        self.symptom = symptom
        self.fix = fix
        self.tasks = tuple(tasks)
        self.note = note

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "ops": self.ops,
            "symptom": self.symptom,
            "fix": self.fix,
            "tasks": self.tasks,
            "note": self.note,
        }


_PATCHES = (
    PatchInfo(
        "patch_inference_mode", "torch", ("torch.inference_mode",),
        "BatchNorm 原地写缓冲区与 inference tensor 冲突，报 "
        "Cannot set version_counter for inference tensor",
        "降级成 no_grad（语义等价，验证阶段不再报错）",
        tasks=("*",),
    ),
    PatchInfo(
        "patch_unique", "torch", ("torch.unique",),
        "二维以上输入直接把进程打崩（0xC0000409）；return_counts 不支持；"
        "bool 输入数值错（少一个 False）",
        "按情况回 CPU；一维数值类型留在显卡上",
        tasks=("*",),
        note="ultralytics 的标签匹配用 unique(return_counts=True)，本来就要回 CPU",
    ),
    PatchInfo(
        "patch_sort", "torch", ("torch.argsort", "torch.sort", "torch.topk"),
        "待排的值全部相等时 UnicodeDecodeError（后端错误消息解不出来），"
        "sort / topk 一样，stable=True 也救不回来",
        "先试原生，炸了回 CPU 重算（.cpu() 可微，梯度与 CPU 逐位相同）",
        tasks=("*",),
    ),
    PatchInfo(
        "patch_scatter", "torch", ("Tensor.scatter_add_",),
        "不支持这个原地算子，而且原生实现不累加（数值就是错的）",
        "dim=0 且一维 index 时改走 index_put_(accumulate=True)，整段留在显卡上；"
        "其余回 CPU",
        tasks=("*",),
    ),
    PatchInfo(
        "patch_bool_ops", "torch",
        ("Tensor.__mul__", "Tensor.__add__", "Tensor.__sub__", "Tensor.__div__"),
        "float32 * bool 的结果 dtype 被判成 bool，数值被截断成全 False",
        "非浮点操作数转成对方 dtype",
        tasks=("*",),
        note="ultralytics 的 align_metric *= mask_pos 就是这个写法",
    ),
    PatchInfo(
        "patch_memory", "torch", ("torch_directml.PrivateUse1Module",),
        "没有 memory_reserved / memory_allocated 等方法，"
        "打日志时抛 AttributeError 把训练打断",
        "补一组恒为 0 的实现",
        tasks=("*",),
    ),
    PatchInfo(
        "patch_cat", "torch", ("torch.cat",),
        "拼接维上长度为 0 的张量不能参与 cat（UnicodeDecodeError）",
        "跳过空张量（语义与 CPU 等价）",
        tasks=("*",),
        note="ultralytics 的 NMS 里 mask 是 0 列张量",
    ),
    PatchInfo(
        "patch_gather", "torch", ("torch.gather",),
        "gather 的反向把散射累加做成了「取平均」（64 份贡献摊成 1）",
        "反向在 CPU 上 scatter_add_",
        tasks=("*",),
    ),
    PatchInfo(
        "patch_bce", "torch", ("torch.nn.BCEWithLogitsLoss",),
        "内部 log_sigmoid_forward 不支持，回落 CPU；反向梯度搬不回显卡，"
        "分类分支梯度直接是 0",
        "用 softplus 公式自己实现，整段留在显卡上算",
        tasks=("detect", "segment", "pose", "obb", "classify"),
    ),
    PatchInfo(
        "patch_addmv", "torch", ("torch.addmv",),
        "aten::addmv.out 不支持（matmul(N-D, 1-D) 在 DML 上正好被拆成它），"
        "CPU 回落且回归分支的梯度比 CPU 大 3.5 倍",
        "改写成 elementwise 形式",
        tasks=("detect", "segment", "pose", "obb"),
    ),
    PatchInfo(
        "patch_slot_assign", "torch",
        ("Tensor.__setitem__", "Tensor.__imul__", "Tensor.__iadd__"),
        "视图的 storage_offset 恒为 0，原地赋值的反向把梯度写错槽位"
        "（致命：分类分支梯度恒为 0，训练不收敛）",
        "槽位赋值改走 index_put_，视图上的原地算子改成 out-of-place 计算 + no_grad 写回",
        tasks=("detect", "segment", "pose", "obb"),
    ),
    PatchInfo(
        "patch_flat_index", "torch", ("Tensor.__getitem__",),
        "高级索引（t[i0, i1, ...]）带梯度时不是并行 kernel，而是一串 copy_"
        "（实测端到端快 4~11%）",
        "折算成扁平偏移 + gather（并行 kernel），反向用 index_put_(accumulate=True) "
        "在显卡上散射累加",
        tasks=("detect", "segment", "pose", "obb"),
        note="空张量 / 空索引会回退原生（gather 表达不出空输出）",
    ),
    PatchInfo(
        "patch_device", "ultralytics", ("ultralytics.utils.torch_utils.select_device",),
        "select_device 不认 privateuseone，传 privateuseone:0 报 Invalid CUDA device",
        "拦下 privateuseone 直通",
        tasks=("*",),
    ),
    PatchInfo(
        "patch_ultralytics_memory", "ultralytics",
        ("ultralytics.engine.trainer.BaseTrainer._get_memory",),
        "训练进度条问显存，DirectML 没有这个方法",
        "恒返回 0",
        tasks=("*",),
    ),
    PatchInfo(
        "patch_probiou", "ultralytics", ("ultralytics.utils.metrics.batch_probiou",),
        "OBB 的旋转框 IoU 在 DirectML 上不支持（UnicodeDecodeError）",
        "回 CPU 算完再搬回显卡",
        tasks=("obb",),
    ),
    PatchInfo(
        "patch_loss_on_cpu", "ultralytics",
        ("ultralytics.utils.loss.v8DetectionLoss",),
        "小模型 / 小输入上，损失在 CPU 反而更快（DirectML 每个算子有 0.2~3 ms 固定开销）",
        "可选开关：把损失计算搬回 CPU，模型前后向仍留在显卡",
        tasks=("detect",),
        note="默认关闭；实测损失前向 52 ms(显卡) vs 24 ms(CPU)",
    ),
)


def list_patches(layer: Optional[str] = None) -> List[Dict[str, Any]]:
    """返回补丁清单（layer 为 "torch" / "ultralytics" 时只给那一层）。"""
    out = []
    for p in _PATCHES:
        if layer and p.layer != layer:
            continue
        out.append(p.as_dict())
    return out


def format_patches(layer: Optional[str] = None) -> str:
    """把补丁清单渲染成一张可读的表。"""
    rows = list_patches(layer)
    if not rows:
        return "（没有补丁）"
    lines = []
    lines.append("%-24s %-12s %s" % ("补丁", "层", "修的问题"))
    lines.append("-" * 96)
    for r in rows:
        lines.append("%-24s %-12s %s" % (r["name"], r["layer"], r["symptom"]))
        lines.append("%-24s %-12s   -> %s" % ("", "", r["fix"]))
        if r["tasks"] != ("*",):
            lines.append("%-24s %-12s   适用: %s" % ("", "", ", ".join(r["tasks"])))
    return "\n".join(lines)
