# -*- coding: utf-8 -*-
"""小工具：保证补丁只生效一次。

enable() 在一个进程里可能被调用多次（比如先在用户代码里
调一次，之后又 train_yolo() 内部调一次）。如果补丁不加保护，第二次调用
会把第一次的替换再包一层，轻则梯度被算两遍，重则把别的补丁顶掉 ——
比如 patch_bool_ops 会重新设置 Tensor.__imul__，把 patch_slot_assign
的修复悄悄顶掉，损失计算的梯度又错位了（而且不报错）。

所以每个补丁在动手之前先问一下：目标函数是不是已经被本库打过标记了。
"""
from __future__ import annotations

_FLAG = "_directml_amd_patched"


def already_patched(fn) -> bool:
    """这个函数是不是已经被本库替换过了。"""
    return bool(getattr(fn, _FLAG, False))


def mark(fn):
    """给替换后的函数打标记。"""
    try:
        setattr(fn, _FLAG, True)
    except (AttributeError, TypeError):  # pragma: no cover - 内置类型可能不可写
        pass
    return fn
