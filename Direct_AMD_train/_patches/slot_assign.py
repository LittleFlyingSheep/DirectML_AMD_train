# -*- coding: utf-8 -*-
"""补丁 11：修 DirectML 上「原地槽位赋值」与「视图上的原地算子」的反向。

这条是让损失计算留在显卡上的前提 —— 以前的做法（patch_loss_on_cpu）
是整段把损失搬到 CPU 绕开它，代价是每个 iteration 都要在 CPU/显卡之间
搬一整套 (bs, anchors, nc) 的张量。

=============================================================================
根因：DirectML 的视图没有真正的 storage_offset
=============================================================================
ultralytics 的检测损失是这么拼装的（ultralytics/utils/loss.py）：

    loss = torch.zeros(3, device=self.device)
    loss[1] = bce_loss.sum() / target_scores_sum   # 分类
    loss[0], loss[2] = self.bbox_loss(...)          # 回归 / DFL
    loss[0] *= self.hyp.box
    loss[1] *= self.hyp.cls
    loss[2] *= self.hyp.dfl

「往张量的某个位置写值」在 PyTorch 里是「写一个视图」，自动微分用
CopySlices 节点记录它。CopySlices 的反向
（torch/csrc/autograd/functions/tensor.cpp）靠「视图相对基张量的偏移量」
定位被覆盖的那一段：

    auto offset = view.sym_storage_offset() - base.sym_storage_offset();
    grad_slice = result.as_strided_symint(view.sym_sizes(), view.sym_strides(), offset);

而 DirectML 的视图根本没有真正的 storage —— 实测：

    t = torch.arange(12.).reshape(3, 4).to('privateuseone:0')
    t[1:2].storage_offset()    CPU: 4     DirectML: 0    ← 永远是 0

于是 offset 算成 0，梯度被写到「第一个」槽位，而不是被覆盖的那个：

    x = torch.zeros(3, requires_grad=True)
    y = x.clone()
    y[1] = 1.0
    y.sum().backward()
    CPU  -> x.grad = [1, 0, 1]   被覆盖的槽位 1 不该再往回传，正确
    DML  -> x.grad = [0, 1, 1]   清零的成了槽位 0，槽位 1 反而留着

前向是对的（loss 数值两边一模一样），所以训练不会报错，
只是梯度悄悄错位 —— 分类分支（Detect 的 cv3）拿到的梯度恒为 0，
cls_loss 永远降不下来，模型学不会分类。

=============================================================================
修法：绕开 CopySlices，损失整段留在显卡上
=============================================================================
DirectML 的 index_put_ 前向和反向都是对的，实测与 CPU 完全一致：

    y = x.clone()
    y.index_put_((torch.tensor([1]),), torch.tensor([1.0]))
    CPU / DML 的 x.grad 都是 [1, 0, 1]

所以这里做两件事：

1) 槽位赋值（t[i] = v）改走 index_put_：
   index_put_ 的反向不走 CopySlices，直接按 index 算梯度，位置不会错。

2) 视图上的原地算子（t[i] *= v）改成「out-of-place 计算 + no_grad 写回」：
   原地算子作用在视图上必然产生 CopySlices，所以干脆不产生它 ——
   值照算不误（写回视图，前向语义不变），反向走 out-of-place 的路径。

只在这两种情况下接管，其余情况（CPU 张量、不涉及梯度的赋值、
复杂索引）都走原实现，避免影响别的路径。

修完之后，损失计算可以整段留在 A 卡上：
    损失数值与 CPU 一致（误差 ~1e-6）
    逐参数梯度与 CPU 对齐（聚合相对误差 ~1e-5）
    分类分支 24/24 个参数梯度非零
"""
from __future__ import annotations


def _reduce_to(g, shape):
    """把广播出来的梯度归约回 shape（广播求和，和 PyTorch 的规则一致）。"""
    if g.dim() > len(shape):
        for _ in range(g.dim() - len(shape)):
            g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdim=True)
    return g.reshape(shape)


def patch_slot_assign() -> None:
    """修 DirectML 上槽位赋值 / 视图原地算子的反向（损失计算留在显卡的前提）。"""
    import torch

    from ._once import already_patched, mark

    if already_patched(torch.Tensor.__setitem__):
        return

    # ------------------------------------------------------------------
    # 视图上的原地算子：out-of-place 计算 + no_grad 写回
    # ------------------------------------------------------------------
    class _InplaceOnView(torch.autograd.Function):
        """视图上的原地算子（t[i] *= v 走这里）。

        反向不走 CopySlices（DirectML 上 CopySlices 会把梯度写错槽位），
        而是按 out-of-place 的公式算。存的是 x 的拷贝而不是 x 本身，
        写回不会触发 autograd 的版本号冲突。
        """

        @staticmethod
        def forward(ctx, x, other, op):
            ctx.op = op
            ctx.x_shape = x.shape
            ctx.x_dtype = x.dtype
            ctx.other_shape = other.shape if torch.is_tensor(other) else None
            ctx.other_dtype = other.dtype if torch.is_tensor(other) else None
            ctx.other_scalar = None if torch.is_tensor(other) else other
            saved = [x.detach().clone()]
            if torch.is_tensor(other):
                saved.append(other)
            ctx.save_for_backward(*saved)

            if op == "mul":
                r = torch.mul(x, other)
            elif op == "div":
                r = torch.div(x, other)
            elif op == "add":
                r = torch.add(x, other)
            else:
                r = torch.sub(x, other)
            if r.dtype != x.dtype:
                # 原地算子不做类型提升，结果要落回 x 的 dtype
                r = r.to(x.dtype)

            with torch.no_grad():  # 写回视图：前向语义与原地算子一致
                x.copy_(r)
            return r

        @staticmethod
        def backward(ctx, grad_out):
            saved = ctx.saved_tensors
            x_old = saved[0]
            op = ctx.op
            has_other = ctx.other_shape is not None
            other = saved[1] if has_other else ctx.other_scalar
            if op == "mul":
                grad_x = _reduce_to(grad_out * other, ctx.x_shape)
                grad_other = (_reduce_to(grad_out * x_old, ctx.other_shape)
                              if has_other else None)
            elif op == "div":
                grad_x = _reduce_to(grad_out / other, ctx.x_shape)
                grad_other = (_reduce_to(-(grad_out * x_old) / (other * other), ctx.other_shape)
                              if has_other else None)
            elif op == "add":
                grad_x = grad_out
                grad_other = _reduce_to(grad_out, ctx.other_shape) if has_other else None
            else:  # sub
                grad_x = grad_out
                grad_other = _reduce_to(-grad_out, ctx.other_shape) if has_other else None
            if grad_x.dtype != ctx.x_dtype:
                grad_x = grad_x.to(ctx.x_dtype)
            if grad_other is not None and grad_other.dtype != ctx.other_dtype:
                grad_other = grad_other.to(ctx.other_dtype)
            return grad_x, grad_other, None

    # ------------------------------------------------------------------
    # 1) t[i] = v  ->  index_put_
    # ------------------------------------------------------------------
    _orig_setitem = torch.Tensor.__setitem__

    def _slot_assign(self, index, value):
        """用 index_put_ 实现标量槽位赋值（index_put_ 的反向不会错位）。"""
        if self.device.type != "privateuseone":
            return _orig_setitem(self, index, value)
        if type(index) is not int:  # bool 也是 int 的子类，type() 精确判断
            return _orig_setitem(self, index, value)
        if self._is_view() or self.dim() == 0:
            return _orig_setitem(self, index, value)
        if not (self.requires_grad
                or (torch.is_tensor(value) and value.requires_grad)):
            # 不涉及梯度的话原实现的反向本来就没问题，保持原样（也更快）
            return _orig_setitem(self, index, value)

        n = self.size(0)
        idx = index + n if index < 0 else index
        if not 0 <= idx < n:  # 越界交给原实现报错
            return _orig_setitem(self, index, value)

        v = value
        if not torch.is_tensor(v):
            v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        else:
            if v.device != self.device:
                v = v.to(self.device)
            if v.dtype != self.dtype:
                v = v.to(self.dtype)

        # t[i] = v 覆盖的是第 i 行，值的形状必须能广播到 self.shape[1:]
        try:
            region = torch.broadcast_shapes(self.shape[1:], v.shape)
        except RuntimeError:
            return _orig_setitem(self, index, value)  # 让原实现报原本的形状错误
        try:
            v = torch.broadcast_to(v, (1,) + tuple(region))
        except RuntimeError:
            return _orig_setitem(self, index, value)

        self.index_put_((torch.tensor([idx], dtype=torch.long, device=self.device),), v)
        return None

    torch.Tensor.__setitem__ = mark(_slot_assign)

    # ------------------------------------------------------------------
    # 2) t[i] *= v（视图上的原地算子）
    # ------------------------------------------------------------------
    def _make_inplace(name, op):
        _orig = getattr(torch.Tensor, name)

        def inplace(self, other):
            needs_fix = (self.device.type == "privateuseone"
                         and self._is_view()
                         and (self.requires_grad
                              or (torch.is_tensor(other) and other.requires_grad)))
            if not needs_fix:
                return _orig(self, other)
            return _InplaceOnView.apply(self, other, op)

        setattr(torch.Tensor, name, mark(inplace))

    for _name, _op in (("__imul__", "mul"), ("__iadd__", "add"),
                       ("__isub__", "sub"), ("__itruediv__", "div")):
        _make_inplace(_name, _op)
