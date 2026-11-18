# -*- coding: utf-8 -*-
"""补丁 12：把显卡上的高级索引改写成「展平偏移 + gather」，让索引也吃并行 kernel。

=============================================================================
为什么要改写
=============================================================================
DirectML 的 aten::index（多个索引张量的高级索引）并不是并行 kernel，
而是一串 copy_ / _to_copy 实测损失前向里出现了 195 次 copy_、162 次 _to_copy，
aten::index 这一项就占掉了前向 60% 以上的 CPU 时间。
而 gather 在 DirectML 上是正经的并行 kernel，单点实测比 aten::index 快 3~28 倍。
损失本体实测：前向 175 ms -> 53 ms（3.3 倍），整轮训练 12.78 s -> 12.23 s（4.3%）。

ultralytics 的检测损失里正好有 9 处这样的高级索引
（ultralytics/utils/tal.py 与 utils/loss.py），每步训练都要走一遍。

=============================================================================
怎么改写
=============================================================================
连续张量是行主序的，所以多维索引可以折算成一个扁平偏移量：

    offset = Σ_j idx_j * stride_j  (+ 固定维度选中的偏移)

剩下的尾部维度如果还在，就等于「以 P = 尾部维度乘积 为步长的一段连续内存」。
于是 t[i0, i1, ...] 可以改写成对扁平视图的一次 gather：

    P == 1 : gather(t.reshape(-1), 0, offset)
    P >  1 : gather(t.reshape(-1), 0, (offset[:,None] * P + arange(P)).reshape(-1))

gather 的输出形状等于 index 的形状（不是输入的形状），
所以索引要先摊成一维，结果再 view 回原来的形状。

反向自己写：DirectML 的 gather 反向是坏的（把多份贡献摊成 1 份，
见补丁 8），这里在显卡上用 index_put_(accumulate=True) 散射累加，
既避开坏掉的 gather 反向，又不用把梯度搬回 CPU。

=============================================================================
越界语义
=============================================================================
原生 aten::index 越界会抛 IndexError；gather 越界在 DirectML 上不报错，
会直接读到野内存（实测取到 [0.0, 9.0] 这种随机值）。
所以这里保留逐维越界检查：把 index 张量拷回主机后查 min/max，
一次 DML->CPU 同步约 0.3 ms；同一个 index 张量重复使用时走缓存，
只在张量被原地改过（版本号变化）时重新查。
不需要这份严格性时可以用 patch_flat_index(verify_bounds=False) 关掉。

=============================================================================
适用范围与已知限制
=============================================================================
只接管最常见的形态：「连续的索引张量前缀 + 尾部整数」。
切片、bool 掩码、0 维索引张量、非连续张量、CPU 张量等一律回退原生实现。

  * 非连续张量回退原生：折算偏移的前提是行主序连续内存。
  * 0 维索引张量在 PyTorch 里等价于整数下标，这里直接回退原生实现。
  * 空张量回退原生：尾部保留的维度里有 0 时 P = 0，
    折算偏移要算 s % P 会直接除零；而且 gather 表达不出空输出。
    （ultralytics 的 NMS 里就有一个 0 列的 mask 张量会走到这里。）
  * 空索引张量回退原生：输出必然是空的，
    而 DirectML 的 gather 处理空下标会抛 UnicodeDecodeError。
  * 索引张量必须和被索引张量在同一个设备上。
  * 越界检查是逐维的（和 PyTorch 一样），
    一次同步会打断显卡的流水线，实测每个索引点约 0.3 ms。
"""
from __future__ import annotations

# 越界检查开关，由 patch_flat_index(verify_bounds=...) 设置
_VERIFY_BOUNDS = True


def _check_bounds(ix, size: int) -> None:
    """逐维检查索引张量是否越界，越界就按 PyTorch 的格式抛 IndexError。

    实现上不把整个索引张量拷回主机，而是在显卡上算 min/max、只回读标量：
    一次标量回读约 0.2 ms，比整块 D2H 拷贝省，也省掉一次几百 KB 的拷贝。

    顺带记一笔：DirectML 上「连续反复计算同一个损失」会把显卡命令队列
    卡死（下一次前向的第一个算子永远不返回，堆栈停在 make_anchors 的
    meshgrid 上）。实测**原生 aten::index 路径也一样卡**，所以这是后端 /
    损失本身的既有问题，不是本补丁引入的；训练循环里迭代之间有数据加载
    等 CPU 工作，队列能排空，不受影响。

    结果缓存在张量身上：同一个索引张量被重复使用时不用再查，
    张量一旦被原地改过（_version 变化）缓存自动失效。
    """
    ver = getattr(ix, "_version", 0)
    cache = getattr(ix, "_dml_amd_bounds", None)
    if cache is not None and cache[0] == ver and cache[1] == size:
        return
    if ix.numel() == 0:
        return
    lo = float(ix.min())
    hi = float(ix.max())
    if hi >= size:
        raise IndexError(
            "index %d is out of bounds for dimension 0 with size %d" % (int(hi), size))
    if lo < -size:
        raise IndexError(
            "index %d is out of bounds for dimension 0 with size %d" % (int(lo), size))
    try:
        ix._dml_amd_bounds = (ver, size)
    except (AttributeError, TypeError):  # pragma: no cover - 某些张量不允许挂属性
        pass


def _plan(t, index):
    """把 t[index] 折算成 (offset, P, out_shape)；折算不了就返回 None。

    offset : int64 张量，单位是「块」，一块 P 个连续元素
    P      : 尾部保留下来的维度乘积（P == 1 表示每个索引取一个标量）
    out_shape : 改写后输出的形状（= 广播后的索引形状 + 保留的尾部维度）
    """
    import torch

    items = index if isinstance(index, tuple) else (index,)
    if not items:
        return None

    # ---- 先做便宜的排除，避免在用不上的索引上浪费工夫 ----
    n_tensor = 0
    for it in items:
        if torch.is_tensor(it):
            if it.dim() == 0 or it.dtype not in (torch.int64, torch.int32):
                return None
            if it.device != t.device:
                return None
            if it.numel() == 0:
                # 空索引 -> 输出必然是空的。
                # DirectML 的 gather 处理空下标会直接抛 UnicodeDecodeError
                # （原生 aten::index 反而正常），这种情况回退原生实现。
                return None
            n_tensor += 1
        elif isinstance(it, int) and not isinstance(it, bool):
            continue
        else:
            return None  # 切片 / None / Ellipsis / bool 掩码 / numpy 数组 ...
    if n_tensor == 0:
        return None

    kinds = ["T" if torch.is_tensor(it) else "I" for it in items]
    first_tensor = kinds.index("T")
    if "I" in kinds[:first_tensor]:
        return None  # 整数下标出现在索引张量前面，折算不了
    k = kinds.index("I") if "I" in kinds else len(kinds)

    shape = list(t.shape)
    n = len(shape)
    m = len(items) - k
    if k + m > n:
        return None

    # 源张量本身是空的（任意一维为 0）：展平之后没有元素可取，
    # gather 的下标必然越界，原生实现则按 PyTorch 语义报 IndexError
    # 或者返回空结果。老老实实回退原生，别自己造语义。
    if t.numel() == 0:
        return None

    tail = shape[k + m:]           # 还会留在输出里的维度
    P = 1
    for d in tail:
        P *= d
    if P == 0:
        # 留在输出里的尾部维度里有 0，输出必然是空的。
        # 下面要算 s % P / s // P，P == 0 会直接除零崩溃
        # （ultralytics 的 NMS 里就有一个 0 列的 mask 张量会走到这里）。
        # gather 也表达不出「空输出」，这种情况回退原生实现。
        return None

    # 连续张量的行主序步长
    strides = [1] * n
    for d in range(n - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]

    off = None
    for j, it in enumerate(items):
        if torch.is_tensor(it):
            s = strides[j]
            if s % P:
                return None
            n_j = shape[j]
            if _VERIFY_BOUNDS:
                _check_bounds(it, n_j)
            ix = it.to(torch.int64)
            # 负下标折算成正下标（PyTorch 允许 -size <= idx < size）
            ix = torch.where(ix < 0, ix + n_j, ix)
            term = ix * (s // P)
        else:
            dim = j              # 整数下标排在索引张量后面，位置即维度号
            s = strides[dim]
            if s % P:
                return None
            v = it + shape[dim] if it < 0 else it
            if not 0 <= v < shape[dim]:
                return None      # 越界交给原生实现报错
            term = v * (s // P)
        off = term if off is None else off + term

    out_shape = tuple(off.shape) + tuple(tail)
    return off, P, out_shape


def patch_flat_index(verify_bounds: bool = True) -> None:
    """把显卡上的高级索引改写成展平 gather（补丁 12）。

    参数
    ----
    verify_bounds:
        是否保留 PyTorch 的越界检查（IndexError）。关掉能再省一点时间，
        但越界索引会读到野内存而不是报错，默认开着。
    """
    import torch

    from ._once import already_patched, mark

    global _VERIFY_BOUNDS
    _VERIFY_BOUNDS = bool(verify_bounds)

    if already_patched(torch.Tensor.__getitem__):
        return

    _orig_getitem = torch.Tensor.__getitem__

    class _FlatIndex(torch.autograd.Function):
        """高级索引 -> 展平 gather。

        前向是一次 gather（DirectML 上的并行 kernel）；
        反向用 index_put_(accumulate=True) 在显卡上散射累加，
        既避开 DirectML 那个把贡献摊平的 gather 反向，也不用把梯度搬回 CPU。
        """

        @staticmethod
        def forward(ctx, t, off, P):
            flat = t.reshape(-1)
            S = off.numel()
            if P == 1:
                idx = off.reshape(-1)
                out = torch.ops.aten.gather(flat, 0, idx)
            else:
                # 一个索引对应连续 P 个元素，摊成一维索引
                idx = (off.reshape(-1, 1) * P
                       + torch.arange(P, dtype=torch.int64, device=off.device)).reshape(-1)
                out = torch.ops.aten.gather(flat, 0, idx).view(S, P)
            ctx.save_for_backward(idx)
            ctx.shape = t.shape
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (idx,) = ctx.saved_tensors
            g = grad_out.reshape(-1)
            # 直接按源张量的形状分配，别让中间的 view 逃出这个函数：
            # 反向是在 autograd 的工作线程上跑的，view 的基张量如果
            # 在那里被释放，DirectML 那边的缓冲回收会和命令队列打架。
            buf = torch.zeros(ctx.shape, dtype=g.dtype, device=g.device)
            buf.view(-1).index_put_((idx,), g, accumulate=True)
            return buf, None, None

    def _getitem(self, index):
        if self.device.type != "privateuseone" or not self.is_contiguous():
            return _orig_getitem(self, index)
        plan = _plan(self, index)
        if plan is None:
            return _orig_getitem(self, index)
        off, P, out_shape = plan
        # view 放在 Function 外面：Function 只产出扁平的取数结果，
        # reshape 由 autograd 正常处理，不牵扯 DirectML 的缓冲释放。
        return _FlatIndex.apply(self, off, P).view(out_shape)

    torch.Tensor.__getitem__ = mark(_getitem)
