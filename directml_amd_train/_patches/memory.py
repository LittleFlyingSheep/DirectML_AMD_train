# -*- coding: utf-8 -*-
"""补丁 6：让 ultralytics 能打印 DirectML 设备的「显存占用」。

ultralytics 打印训练进度时会问 accelerator 要 memory_reserved()，
DirectML 的 PrivateUse1Module 没有这个方法，会直接抛 AttributeError
把整轮训练打断。这只是日志，不影响训练本身，这里补一个恒为 0 的实现。
"""
from __future__ import annotations


def patch_memory() -> None:
    try:
        import torch_directml
        mod = getattr(torch_directml, "PrivateUse1Module", None)
        if mod is not None:
            # 注意：ultralytics 的 get_torch_device_backend() 返回的是
            # PrivateUse1Module 这个「类」而不是实例，所以这些方法要写成
            # staticmethod，否则类属性访问不会自动传 self。
            for _name, _fn in (("memory_reserved", lambda: 0),
                               ("memory_allocated", lambda: 0),
                               ("empty_cache", lambda: None),
                               ("synchronize", lambda: None)):
                if not hasattr(mod, _name):
                    setattr(mod, _name, staticmethod(_fn))
            if not hasattr(mod, "get_device_properties"):
                class _Props(object):
                    total_memory = 8 * 1024 ** 3

                setattr(mod, "get_device_properties", staticmethod(lambda device: _Props()))
    except Exception:
        pass

    try:
        from ultralytics.engine import trainer as _tr
        _tr.BaseTrainer._get_memory = lambda self, *a, **kw: 0.0
    except Exception:
        pass
