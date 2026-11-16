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


def test_slot_assign_matches_cpu():
    """原地槽位赋值的反向必须与 CPU 一致。

    DirectML 原生实现会把梯度写错槽位（视图的 storage_offset 恒为 0，
    CopySlices 反向因此定位错位置），patch_slot_assign 把它改成走 index_put_。
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
    assert grads[DML] == grads["cpu"], "槽位赋值的反向错位了: %s" % grads[DML]


def test_slot_assign_row_matches_cpu():
    """二维张量的整行赋值，反向也必须与 CPU 一致。"""
    import torch
    grads = {}
    for dev in ("cpu", DML):
        x = _leaf((3, 4), torch.device(dev))
        y = x.clone()
        y[1] = torch.ones(4, device=torch.device(dev))
        y.sum().backward()
        grads[dev] = _gs(x).abs().sum(1).tolist()
    assert grads["cpu"] == [4.0, 0.0, 4.0]  # 被覆盖的那一行不再往回传
    assert grads[DML] == grads["cpu"], "整行赋值的反向错位了: %s" % grads[DML]


def test_inplace_on_view_matches_cpu():
    """视图上的原地算子（t[i] *= v），反向必须与 CPU 一致。

    ultralytics 的损失就是用这个写法给损失乘权重的：
        loss[0] *= self.hyp.box
    """
    import torch
    grads = {}
    for dev in ("cpu", DML):
        x = _leaf((3,), torch.device(dev))
        y = x.clone()
        y[1] = y[0] * 2.0   # 写入的值本身依赖 x，梯度才有意义
        y[1] *= 4.0
        y.sum().backward()
        grads[dev] = _gs(x).tolist()
    assert grads[DML] == grads["cpu"], "视图原地算子的反向错位了: %s" % grads[DML]


def test_bce_with_logits_matches_cpu():
    """BCEWithLogitsLoss 在显卡上算，数值必须与 CPU 一致。"""
    import torch
    torch.manual_seed(0)
    z0 = torch.randn(16, 8)
    t0 = torch.randint(0, 2, (16, 8)).float()
    outs = {}
    for dev in ("cpu", DML):
        z = z0.to(dev).detach().clone().requires_grad_(True)
        t = t0.to(dev)
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")(z, t)
        loss.sum().backward()
        outs[dev] = (loss.detach().cpu(), z.grad.detach().cpu())
    assert torch.allclose(outs["cpu"][0], outs[DML][0], atol=1e-5)
    assert torch.allclose(outs["cpu"][1], outs[DML][1], atol=1e-5)


def test_matmul_nd_1d_matches_cpu():
    """(N-D, 1-D) 的 matmul：DirectML 上原本会拆成 addmv 回落到 CPU。"""
    import torch
    torch.manual_seed(0)
    a0 = torch.randn(2, 5, 4, 16)
    v0 = torch.randn(16)
    outs = {}
    for dev in ("cpu", DML):
        a = a0.to(dev).detach().clone().requires_grad_(True)
        v = v0.to(dev)
        out = a.matmul(v)
        out.sum().backward()
        outs[dev] = (out.detach().cpu(), a.grad.detach().cpu())
    assert torch.allclose(outs["cpu"][0], outs[DML][0], atol=1e-4)
    assert torch.allclose(outs["cpu"][1], outs[DML][1], atol=1e-4)


def test_scatter_add_matches_cpu():
    """scatter_add_ 在 DirectML 上原生实现是错的，补丁后必须与 CPU 一致。

    DirectML 的 scatter_add_ 不累加（[1,0,1,0] vs CPU 的 [2,0,1,0]），
    补丁改成 dim=0 时用 index_put_(accumulate=True)，整段留在显卡上。
    """
    import torch
    outs = {}
    for dev in ("cpu", DML):
        base = torch.zeros(4, device=dev)
        idx = torch.tensor([0, 0, 2], device=dev)
        src = torch.ones(3, device=dev)
        base.scatter_add_(0, idx, src)
        outs[dev] = base.cpu().tolist()
    assert outs["cpu"] == [2.0, 0.0, 1.0, 0.0]
    assert outs[DML] == outs["cpu"], "scatter_add_ 没累加: %s" % outs[DML]


def test_enable_twice_keeps_fixes():
    """enable_directml_amd() 调两次，修复不能被后来的补丁顶掉。

    patch_bool_ops 也会替换 Tensor.__imul__ 等 dunder，如果它不幂等，
    第二次 enable 时就会把 patch_slot_assign 的修复悄悄覆盖掉 ——
    损失计算的梯度又错位，而且不报错。
    """
    import torch

    from directml_amd_train._patches import enable_directml_amd

    enable_directml_amd(verbose=False)
    enable_directml_amd(verbose=False)

    grads = {}
    for dev in ("cpu", DML):
        x = _leaf((3,), torch.device(dev))
        y = x.clone()
        y[1] = 1.0
        y[1] *= 4.0
        y.sum().backward()
        grads[dev] = _gs(x).tolist()
    assert grads[DML] == grads["cpu"], "重复 enable 之后修复失效了: %s" % grads[DML]


def test_loss_on_cpu_switch_is_optional():
    """可选的 loss_on_cpu 开关：打开后损失回到 CPU，但梯度仍与显卡一致。"""
    import torch

    from ultralytics.utils import loss as _loss_mod
    from directml_amd_train._patches import enable_directml_amd

    original = _loss_mod.v8DetectionLoss.get_assigned_targets_and_loss
    try:
        enable_directml_amd(verbose=False, loss_on_cpu=True)
        from directml_amd_train import selftest as st
        res = st.check_gradient_alignment(verbose=False)
        assert res["ok"], "CPU 损失路径下梯度没对齐: %s" % res["worst_name"]
        assert res["cls_nonzero_on_dml"] == res["cls_param_count"]
    finally:
        # 复位：后面的测试要用默认的「损失在显卡」路径
        _loss_mod.v8DetectionLoss.get_assigned_targets_and_loss = original


def test_flat_index_matches_cpu():
    """高级索引改写成展平 gather 后，数值与反向都必须与 CPU 一致。

    补丁 12 把 t[i0, i1, ...] 折算成扁平偏移再用 gather 取，
    DirectML 的 aten::index 本身不是并行 kernel（是一串 copy_），
    改写后单点实测快 3~28 倍。
    """
    import torch
    torch.manual_seed(0)

    base3 = torch.randn(6, 7, 3)
    base4 = torch.randn(3, 4, 5, 6)
    base1 = torch.randn(10)

    i0 = torch.tensor([0, 2, 5, 1])
    i1 = torch.tensor([3, 6, 0, 2])
    i1b = torch.tensor([1, 3])          # 适配 base4 的第二维（长度 4）

    cases = [
        (base3, (i0, i1)),                       # 两个索引张量 + 保留尾维
        (base4, (i0[:2], i1b, 2)),             # 尾部整数选择
        (base4, (i0[:2], i1b, 2, 1)),          # 尾部两个整数
        (base1, (torch.tensor([9, 0, 4]),)),     # 一维 + 一维索引
        (base3, (i0, -1)),                       # 负下标
        (base3, (i0.reshape(2, 2), i1.reshape(2, 2))),   # 二维索引
        (base3, (i0.reshape(4, 1), i1.reshape(1, 4))),   # 广播
        (base3, (torch.tensor([1, 1, 1]),)),     # 重复索引
    ]

    for base, idx in cases:
        outs = {}
        for dev in ("cpu", DML):
            x = base.to(dev).detach().clone().requires_grad_(True)
            ix = tuple(i.to(dev) if torch.is_tensor(i) else i for i in idx)
            out = x[ix]
            w = torch.arange(1, out.numel() + 1, dtype=out.dtype,
                             device=torch.device(dev)).reshape(out.shape)
            (out * w).sum().backward()
            outs[dev] = (out.detach().cpu(), x.grad.detach().cpu())
        assert torch.allclose(outs["cpu"][0], outs[DML][0], atol=1e-5), "高级索引的值对不上: %s" % (idx,)
        assert torch.allclose(outs["cpu"][1], outs[DML][1], atol=1e-5), "高级索引的反向对不上: %s" % (idx,)


def test_flat_index_out_of_bounds_raises():
    """越界索引要像原生 aten::index 一样抛 IndexError。

    DirectML 的 gather 越界不报错，会直接读到野内存，
    所以补丁 12 自己保留了逐维越界检查。
    """
    import torch
    for dev in ("cpu", DML):
        x = torch.randn(10, device=torch.device(dev))
        with pytest.raises(IndexError):
            x[torch.tensor([0, 99], device=torch.device(dev))]


def test_flat_index_falls_back_to_native():
    """切片 / bool 掩码这类改写不了的索引，要原样交给原生实现。"""
    import torch
    torch.manual_seed(0)
    base = torch.randn(4, 5)
    for dev in ("cpu", DML):
        x = base.to(dev)
        assert torch.allclose(x[:, 1:3].cpu(), base[:, 1:3])
        mask = torch.tensor([True, False, True, False], device=torch.device(dev))
        assert torch.allclose(x[mask].cpu(), base[mask.cpu()])


def test_flat_index_bounds_check_can_be_disabled():
    """verify_bounds=False 只是关掉越界检查，取值语义不变。"""
    import torch

    from directml_amd_train._patches import patch_flat_index

    torch.manual_seed(0)
    base = torch.randn(6, 4)
    idx = torch.tensor([0, 3, 5])
    try:
        patch_flat_index(verify_bounds=False)
        outs = {}
        for dev in ("cpu", DML):
            x = base.to(dev).detach().clone().requires_grad_(True)
            out = x[idx.to(dev)]
            out.sum().backward()
            outs[dev] = (out.detach().cpu(), x.grad.detach().cpu())
        assert torch.allclose(outs["cpu"][0], outs[DML][0], atol=1e-5)
        assert torch.allclose(outs["cpu"][1], outs[DML][1], atol=1e-5)
    finally:
        patch_flat_index(verify_bounds=True)
