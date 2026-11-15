# -*- coding: utf-8 -*-
"""逐条补丁的回归测试：同一份输入，CPU 与 DirectML 的结果必须一致。

这些测试锁住的是「DirectML 与 CPU 的语义差异」本身 ——
将来换 torch-directml 版本时，跑一遍就知道有没有新的坑。

    pytest tests/ -v

没有 DirectML 的机器上会整体跳过（用 -m "not dml" 也可以手动排除）。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directml_amd_train._patches import enable_directml_amd  # noqa: E402

DML = "privateuseone:0"

try:
    enable_directml_amd(verbose=False)
    import torch
    import torch_directml  # noqa: F401
    _HAS_DML = True
except Exception:  # pragma: no cover
    _HAS_DML = False

pytestmark = pytest.mark.skipif(not _HAS_DML, reason="没有可用的 DirectML 设备")


def _leaf(shape, dev):
    import torch
    return torch.randn(*shape).to(dev).requires_grad_(True)


def _gs(x):
    return x.grad.detach().float().cpu()


def test_select_backward():
    """getitem/select 的反向：DirectML 应与 CPU 一致。"""
    for dev in ("cpu", DML):
        x = _leaf((3,), torch.device(dev))
        (x[1] * 3.0).sum().backward()
        assert _gs(x).tolist() == [0.0, 3.0, 0.0]


def test_permute_contiguous_backward():
    import torch
    grads = {}
    for dev in ("cpu", DML):
        x = _leaf((2, 2, 2100), torch.device(dev))
        x.permute(0, 2, 1).contiguous().sum().backward()
        grads[dev] = float(_gs(x).abs().sum())
    assert abs(grads[DML] - grads["cpu"]) / grads["cpu"] < 1e-3


def test_bool_multiply_semantics():
    """float * bool 的 dtype 与数值必须与 CPU 一致。"""
    import torch
    for dev in ("cpu", DML):
        f = torch.ones(4, dtype=torch.float32, device=torch.device(dev))
        b = torch.tensor([True, False, True, False], device=torch.device(dev))
        out = f * b
        assert out.dtype == torch.float32, "dev=%s dtype=%s" % (dev, out.dtype)
        assert out.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_gather_backward():
    """gather 的反向要按 index 散射累加，而不是取平均。"""
    import torch
    grads = {}
    for dev in ("cpu", DML):
        x = _leaf((4,), torch.device(dev))
        idx = torch.zeros(4, dtype=torch.int64, device=torch.device(dev))
        x.gather(0, idx).sum().backward()
        grads[dev] = _gs(x).tolist()
    assert grads[DML] == grads["cpu"], "gather 反向被摊薄: %s" % grads


def test_gather_backward_function_form():
    """函数形式 torch.gather(x, ...) 也要被修到。"""
    import torch
    grads = {}
    for dev in ("cpu", DML):
        x = _leaf((4,), torch.device(dev))
        idx = torch.zeros(4, dtype=torch.int64, device=torch.device(dev))
        torch.gather(x, 0, idx).sum().backward()
        grads[dev] = _gs(x).tolist()
    assert grads[DML] == grads["cpu"], "gather 反向被摊薄: %s" % grads


def test_unique_return_counts():
    import torch
    for dev in ("cpu", DML):
        x = torch.tensor([1, 1, 2, 3, 3], device=torch.device(dev))
        vals, counts = torch.unique(x, return_counts=True)
        assert counts.tolist() == [2, 1, 2]


def test_cat_with_zero_column():
    """拼接维上长度为 0 的张量不能参与 cat（DirectML 原生会报错）。"""
    import torch
    for dev in ("cpu", DML):
        a = torch.randn(2, 3, device=torch.device(dev))
        b = torch.randn(2, 0, device=torch.device(dev))
        c = torch.randn(2, 4, device=torch.device(dev))
        out = torch.cat((a, b, c), 1)
        assert out.shape == (2, 7)


def test_index_put_raw_difference_is_known():
    """记录 DirectML 原生 index_put_ 的已知差异。

    DirectML 的原地槽位赋值反向会清错槽位（CPU 清槽位 1，它清槽位 0），
    这是「A 卡训练跑得完却不收敛」的根因。本库没有去修这个算子，
    而是用 patch_loss_on_cpu 把损失计算搬到 CPU 绕开它。

    所以这里断言的是「差异仍然存在」—— 万一哪天 torch-directml 修了它，
    这条断言会失败，那时就该考虑把 patch_loss_on_cpu 撤掉、直接用原生算子。
    """
    import torch
    grads = {}
    for dev in ("cpu", DML):
        x = _leaf((3,), torch.device(dev))
        y = x.clone()
        y[1] = 1.0
        y.sum().backward()
        grads[dev] = _gs(x).tolist()
    assert grads["cpu"] == [1.0, 0.0, 1.0]
    assert grads[DML] == [0.0, 1.0, 1.0], (
        "DirectML 的 index_put_ 行为变了：%s —— patch_loss_on_cpu 可能可以撤了" % grads[DML])
