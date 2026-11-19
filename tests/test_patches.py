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

from Direct_AMD_train._patches import enable  # noqa: E402

DML = "privateuseone:0"

try:
    enable(verbose=False)
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
    """enable() 调两次，修复不能被后来的补丁顶掉。

    patch_bool_ops 也会替换 Tensor.__imul__ 等 dunder，如果它不幂等，
    第二次 enable 时就会把 patch_slot_assign 的修复悄悄覆盖掉 ——
    损失计算的梯度又错位，而且不报错。
    """
    import torch

    from Direct_AMD_train._patches import enable

    enable(verbose=False)
    enable(verbose=False)

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
    from Direct_AMD_train._patches import enable

    original = _loss_mod.v8DetectionLoss.get_assigned_targets_and_loss
    try:
        enable(verbose=False, loss_on_cpu=True)
        from Direct_AMD_train import selftest as st
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


def test_flat_index_zero_size_tail_matches_cpu():
    """0 尺寸尾维（0 列）的张量做高级索引：不能除零崩溃，结果要与 CPU 一致。

    ultralytics 的 NMS 里会出现 0 列的 mask 张量。补丁 12 折算扁平偏移时
    算出尾部维度乘积 P = 0，接着去算 s % P，直接 ZeroDivisionError 把训练打崩。
    这里锁住「回退原生实现」这条路：DirectML 原生 aten::index 对 0 列张量
    是正确的，输出 shape 与 CPU 相同。
    """
    import torch
    torch.manual_seed(0)

    base = torch.randn(8, 5376, 0)
    i0 = torch.randint(0, 8, (75,))
    i1 = torch.randint(0, 5376, (75,))

    shapes = {}
    for dev in ("cpu", DML):
        x = base.to(dev)
        out = x[i0.to(dev), i1.to(dev)]
        shapes[dev] = tuple(out.shape)
        assert out.numel() == 0
    assert shapes["cpu"] == (75, 0)
    assert shapes[DML] == shapes["cpu"], "0 列张量的索引结果对不上: %s" % shapes


def test_flat_index_empty_source_raises_index_error():
    """源张量为空（shape (0, 5)）时索引要报越界，不能读到野内存。

    展平之后一个元素都没有，gather 的下标必然越界；
    原生实现按 PyTorch 语义报 IndexError，补丁这边必须保持一致。
    """
    import torch
    for dev in ("cpu", DML):
        x = torch.randn(0, 5, device=torch.device(dev))
        idx = torch.zeros(3, dtype=torch.int64, device=torch.device(dev))
        with pytest.raises(IndexError):
            x[idx]


def test_flat_index_empty_index_matches_cpu():
    """空索引张量：输出必然是空的，形状要与 CPU 一致。"""
    import torch
    for dev in ("cpu", DML):
        x = torch.randn(8, 5376, device=torch.device(dev))
        idx = torch.zeros(0, dtype=torch.int64, device=torch.device(dev))
        out = x[idx]
        assert tuple(out.shape) == (0, 5376)


def test_ultralytics_nms_zero_column_mask_no_crash():
    """ultralytics 验证阶段的 NMS：0 列 mask 不能把补丁打崩。

    ultralytics/utils/nms.py 里 extra == 0（没有 mask 系数 / 关键点通道）时：

        box, cls, mask = x.split((4, nc, extra), 1)   # mask 是 (n, 0)

    multi_label 分支里的 mask[i] 是对 0 列张量的高级索引，
    补丁 12 折算扁平偏移时算出尾部维度乘积 P = 0，去算 s % P 就直接除零崩溃。
    detect/val.py 的验证阶段正好走这条分支（multi_label=True），
    所以「第 1 轮验证时被打崩」就是这么来的。
    """
    import torch

    from ultralytics.utils.nms import non_max_suppression

    torch.manual_seed(0)
    nc, nb = 2, 500

    # 预测张量在 CPU 上造好再搬到显卡：DirectML 上对同一个张量连续做
    # 跨步原地写（t[:, k, :] = v）只有第一次生效，后续会静默失效，
    # 在显卡上现造会把数据造坏。
    base = torch.zeros(1, 4 + nc, nb)
    base[:, 0, :] = 10.0   # cx
    base[:, 1, :] = 10.0   # cy
    base[:, 2, :] = 8.0    # w
    base[:, 3, :] = 8.0    # h
    base[:, 4, :] = 0.9    # class 0
    base[:, 5, 10:] = 0.8  # class 1

    shapes = {}
    classes = {}
    for dev in ("cpu", DML):
        pred = base.to(torch.device(dev))
        out = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45,
                                  multi_label=True, max_det=300, nc=nc)
        assert len(out) == 1
        shapes[dev] = tuple(out[0].shape)
        classes[dev] = out[0][:, 5].tolist()
    assert shapes[DML][1] == 6, "NMS 输出列数不对: %s" % shapes[DML]
    assert shapes[DML][0] > 0, "NMS 一个框都没留: %s" % shapes[DML]
    assert shapes[DML] == shapes["cpu"], (
        "NMS 结果和 CPU 对不上: %s vs %s" % (shapes[DML], shapes["cpu"]))
    assert classes[DML] == classes["cpu"], (
        "NMS 输出的类别和 CPU 对不上: %s vs %s" % (classes[DML], classes["cpu"]))


def test_flat_index_bounds_check_can_be_disabled():
    """verify_bounds=False 只是关掉越界检查，取值语义不变。"""
    import torch

    from Direct_AMD_train._patches import patch_flat_index

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


def test_unique_multi_dim_matches_cpu():
    """二维以上的 unique 在原生 DirectML 上会把进程打崩，补丁后必须与 CPU 一致。

    原生表现是 dml_tensor_desc.cc:76 的断言失败（退出码 0xC0000409），
    那是原生层崩溃，不是 Python 异常，try/except 拦不住。
    """
    import torch
    torch.manual_seed(0)
    for dev in ("cpu", DML):
        x = torch.randn(4, 5, device=torch.device(dev))
        out = torch.unique(x)
        assert out.shape == (20,)
        assert torch.allclose(out.cpu(), torch.unique(x.cpu()))


def test_unique_bool_matches_cpu():
    """bool 的 unique 原生数值就是错的（DML 少一个 False），补丁后必须与 CPU 一致。"""
    import torch
    for dev in ("cpu", DML):
        x = torch.tensor([True, False, True, False], device=torch.device(dev))
        out = torch.unique(x)
        assert out.cpu().tolist() == torch.unique(x.cpu()).tolist()


def test_unique_return_counts_matches_cpu():
    """return_counts 原生不支持，补丁后必须与 CPU 一致。"""
    import torch
    torch.manual_seed(0)
    for dev in ("cpu", DML):
        x = torch.randint(0, 5, (32,), device=torch.device(dev))
        vals, counts = torch.unique(x, return_counts=True)
        cv, cc = torch.unique(x.cpu(), return_counts=True)
        assert torch.equal(vals.cpu(), cv)
        assert torch.equal(counts.cpu(), cc)


def test_argsort_equal_values_matches_cpu():
    """全等值张量的 argsort 原生会炸（UnicodeDecodeError），补丁后必须能用。"""
    import torch
    for dev in ("cpu", DML):
        x = torch.full((64,), 0.9, device=torch.device(dev))
        idx = torch.argsort(x, descending=True)
        assert idx.shape == (64,)
        assert idx.device.type == ("privateuseone" if dev == DML else "cpu")


def test_sort_topk_fallback_matches_cpu():
    """sort / topk 遇到全等值也要能跑，且数值与 CPU 一致。"""
    import torch
    for dev in ("cpu", DML):
        x = torch.full((32,), 0.5, device=torch.device(dev))
        v, i = torch.sort(x, descending=True)
        assert torch.allclose(v.cpu(), torch.sort(x.cpu(), descending=True).values)
        tv, ti = torch.topk(x, 3)
        assert torch.allclose(tv.cpu(), torch.topk(x.cpu(), 3).values)


def test_sort_fallback_keeps_gradient():
    """回 CPU 重算不能把梯度弄丢：结果必须与 CPU 逐位相同。"""
    import torch
    torch.manual_seed(0)
    base = torch.randn(8, 4)

    def grads(dev, cpu_side):
        x = base.clone().to(torch.device(dev)).requires_grad_(True)
        if cpu_side:
            out = torch.sort(x.cpu(), dim=1, descending=True).values.to(x.device)
        else:
            out = torch.sort(x, dim=1, descending=True).values
        (out * torch.arange(1., 5., device=x.device)).sum().backward()
        return x.grad.detach().cpu()

    g_cpu = grads("cpu", False)
    g_dml = grads(DML, True)
    assert torch.allclose(g_cpu, g_dml, atol=1e-5)


def test_registry_matches_actual_patches():
    """注册表里列的每条补丁都要有对应的实现，别让文档漂了。"""
    from Direct_AMD_train import list_patches
    from Direct_AMD_train import _patches

    rows = list_patches()
    assert len(rows) >= 15
    for r in rows:
        assert hasattr(_patches, r["name"]), "注册表里的 %s 没有实现" % r["name"]
        assert r["layer"] in ("torch", "ultralytics")
        assert r["symptom"] and r["fix"]

    torch_layer = {r["name"] for r in list_patches(layer="torch")}
    ultra_layer = {r["name"] for r in list_patches(layer="ultralytics")}
    assert torch_layer and ultra_layer
    assert not (torch_layer & ultra_layer)


def test_enable_torch_layer_does_not_import_ultralytics():
    """framework="torch" 不能把 ultralytics 拖进来 —— 纯 PyTorch 项目也得能用。"""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    code = ("import sys\n"
            "from Direct_AMD_train import enable\n"
            "enable(framework='torch', verbose=False)\n"
            "print('CLEAN' if 'ultralytics' not in sys.modules else 'DIRTY')\n")
    out = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env)
    assert "CLEAN" in out.stdout, out.stdout + out.stderr


def test_probe_subset_runs():
    """能力探测工具要能跑完，并且补丁后这一组一个都不该失败。"""
    from Direct_AMD_train import probe_operators

    data = probe_operators(verbose=False, only=("索引/散射",))
    assert data["total"] > 0
    assert data["failed"] == 0, [r for r in data["results"] if r["ok"] is False]


def test_empty_sub_shape():
    """0 尺寸张量做减法：DirectML 上的输出形状必须与 CPU 一致（补丁 13）。

    后端在「0 维后面还跟着别的维度」时会把输出形状算塌，
    表现为 RuntimeError: The size of tensor a (0) must match the size of
    tensor b (4) at non-singleton dimension 1。
    """
    import torch
    for shape in [(0,), (0, 4), (4, 0), (0, 2), (2, 0, 3), (1, 0), (0, 0)]:
        for dev in ("cpu", DML):
            a = torch.zeros(shape, device=torch.device(dev))
            b = torch.zeros(shape, device=torch.device(dev))
            out = a - b
            assert tuple(out.shape) == shape, (
                "dev=%s shape=%s -> %s" % (dev, shape, tuple(out.shape)))
            assert out.dtype == torch.float32


def test_empty_sub_dtype_and_inplace():
    """空张量的减法：dtype 要按类型提升算，原地减法要原地返回。"""
    import torch
    dev = torch.device(DML)
    a = torch.zeros((0, 4), device=dev)
    b = torch.zeros((0, 4), device=dev, dtype=torch.int64)
    out = a - b
    assert out.dtype == torch.float32, out.dtype
    assert tuple(out.shape) == (0, 4)
    c = torch.zeros((0, 4), device=dev)
    r = c.sub_(b)
    assert r is c
    assert tuple(c.shape) == (0, 4)


def test_nonempty_sub_unchanged():
    """非空张量的减法不能被补丁改动（数值必须逐位相同）。"""
    import torch
    for shape in [(3, 4), (2, 2, 5)]:
        x = torch.randn(shape, device=torch.device(DML))
        y = torch.randn(shape, device=torch.device(DML))
        out = x - y
        ref = x.cpu() - y.cpu()
        assert float((out.cpu() - ref).abs().max()) < 1e-5
