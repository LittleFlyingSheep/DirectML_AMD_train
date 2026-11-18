# -*- coding: utf-8 -*-
"""端到端对齐测试：CPU 与 DirectML 的逐参数梯度必须一致。

这是「训练能不能收敛」的前置条件 —— DirectML 的差异大多是前向对、反向错，
光比 loss 数值看不出来（两边 loss 一模一样）。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Direct_AMD_train.selftest import check_gradient_alignment  # noqa: E402

try:
    import torch_directml  # noqa: F401
    _HAS_DML = True
except Exception:
    _HAS_DML = False

pytestmark = pytest.mark.skipif(not _HAS_DML, reason="没有可用的 DirectML 设备")


def test_loss_values_match():
    res = check_gradient_alignment(verbose=False)
    for a, b in zip(res["cpu_loss"], res["dml_loss"]):
        assert abs(a - b) / max(abs(a), 1e-9) < 1e-3


def test_gradients_match():
    res = check_gradient_alignment(verbose=False)
    assert res["ok"], "梯度未对齐: worst=%s (%s)" % (res["worst_rel"], res["worst_name"])
    assert res["zero_on_dml"] == [], "DirectML 上有参数梯度被清零: %s" % res["zero_on_dml"]


def test_cls_branch_gets_gradient():
    """分类分支（Detect.cv3）必须有梯度 —— 历史上就是它恒为 0。"""
    res = check_gradient_alignment(verbose=False)
    assert res["cls_param_count"] > 0
    assert res["cls_nonzero_on_dml"] == res["cls_param_count"]


def test_loss_is_computed_on_gpu():
    """损失计算必须整段留在显卡上（以前是搬到 CPU 算的）。"""
    res = check_gradient_alignment(verbose=False)
    assert res["loss_on_gpu"], "损失不在显卡上算: %s" % res["loss_device"]
    assert res["loss_device"].startswith("privateuseone")
