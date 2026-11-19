# -*- coding: utf-8 -*-
"""补丁 13：修 DirectML 上「0 尺寸张量做减法」会算错输出形状。

=============================================================================
根因：aten::sub 在「前面有 0 维、后面还跟着维度」的张量上把输出形状算塌
=============================================================================
DirectML 的减法内核遇到 0 尺寸的维度时，后续维度的输出长度会一并变成 0，
于是 PyTorch 的形状校验直接报：

    RuntimeError: The size of tensor a (0) must match the size of tensor b (4)
    at non-singleton dimension 1

实测（AMD Radeon Pro VII / torch-directml 0.2.5.dev240914）：

    形状        sub       add   mul   div
    (0,)        OK        OK    OK    OK
    (0, 4)      炸        OK    OK    OK
    (4, 0)      OK        OK    OK    OK
    (0, 2)      炸        OK    OK    OK
    (2, 0, 3)   炸        OK    OK    OK
    (1, 0)      OK        OK    OK    OK

也就是只有「减法」和「0 维后面还跟着别的维度」这两个条件同时满足才炸。
纯一维的 (0,)、0 维在末尾的 (4, 0) 都没事 —— 看起来是内核按行块分配输出，
遇到空行块时把块内的长度也一起丢了。

=============================================================================
谁会踩到
=============================================================================
ultralytics 的 NMS 里：

    x = x[filt]                      # 按置信度筛
    x[:, :4] = xywh2xyxy(x[:, :4])   # ← 这一行排在空判断前面

    if not x.shape[0]:               # 空了才跳过，已经晚了
        continue

某一帧一个候选框都没有时（画面里什么都没有、模型又学得很确信），
x 是 (0, 4+nc)，xywh2xyxy 里那句 `xy - wh` 就是 (0,2) - (0,2) —— 直接炸，
整轮验证被打断，训练退出。

实测症状：训练跑到某一轮突然退出，日志里就是上面那条 RuntimeError，
而前面几十轮都正常。模型越准越容易踩到 —— 空画面上的置信度越来越低，
低于 val 的 conf 阈值就变成零候选。

=============================================================================
修法
=============================================================================
0 元素的二元运算，结果本来就是空的，没有数值可算；只有形状和 dtype 有意义。
所以直接按 CPU 的规则把结果造出来，不去后端：

  * 形状 = torch.broadcast_shapes(self.shape, other.shape)
  * dtype = torch.result_type(self, other)
  * 原地运算（sub_）不能改形状，结果就是 self 自己

形状不兼容时（本来就该报错的时候）仍然交给原实现去报原本的错，
不改变任何错误语义。

=============================================================================
顺序：必须排在 torch 层最后
=============================================================================
本补丁要包住前面所有补丁装好的入口，所以必须最后打。特别注意 dunder：
bool_ops 把 __sub__ 指向 torch.Tensor.sub，slot_assign 又把 __isub__ 换成
「视图上原地算子走 out-of-place」的实现（那条是分类分支能学起来的前提）。
所以这里对 dunder 是「包装当前实现」而不是「替换成 lambda」，
否则会把 slot_assign 的梯度对齐悄悄顶掉。
"""
from __future__ import annotations

# 本补丁自己的幂等标记（不能复用 _once 的通用标记：
# bool_ops 已经把 Tensor.sub 标记过了，用通用标记会被误判成「已打过」）
_FLAG = "_dml_amd_empty_ops"


def _empty_result(self, other, inplace: bool):
    """结果为空的减法：按 CPU 规则把空结果造出来。

    返回 None 表示「这个不该我管，走原实现」。
    """
    import torch

    if self.device.type != "privateuseone":
        return None
    if torch.is_tensor(other):
        try:
            shape = torch.broadcast_shapes(self.shape, other.shape)
        except RuntimeError:
            return None          # 形状不兼容：交给原实现报原本的错
    else:
        shape = self.shape       # 标量参与：形状不变
    if inplace:
        if tuple(shape) != tuple(self.shape):
            return None          # 原地运算改不了形状，交给原实现报错
        return self              # self 本来就是空的，值没有意义
    if 0 not in tuple(shape):
        return None              # 结果不为空，后端算得对
    dtype = torch.result_type(self, other)
    return torch.empty(tuple(shape), dtype=dtype, device=self.device)


def patch_empty_ops() -> None:
    """修 DirectML 上 0 尺寸张量做减法时输出形状算错（sub / sub_ / 减号）。"""
    import torch

    if getattr(torch.Tensor.sub, _FLAG, False):
        return

    def _mark(fn):
        try:
            setattr(fn, _FLAG, True)
        except (AttributeError, TypeError):  # pragma: no cover
            pass
        return fn

    # ------------------------------------------------------------------
    # 1) Tensor.sub / Tensor.sub_
    # ------------------------------------------------------------------
    _orig_sub = torch.Tensor.sub
    _orig_sub_ = torch.Tensor.sub_

    def sub(self, other, *args, **kw):
        r = _empty_result(self, other, inplace=False)
        if r is not None:
            return r
        return _orig_sub(self, other, *args, **kw)

    def sub_(self, other, *args, **kw):
        r = _empty_result(self, other, inplace=True)
        if r is not None:
            return r
        return _orig_sub_(self, other, *args, **kw)

    torch.Tensor.sub = _mark(sub)
    torch.Tensor.sub_ = _mark(sub_)

    # ------------------------------------------------------------------
    # 2) torch.sub（函数形式，走的是另一条 C++ 入口，也要管）
    # ------------------------------------------------------------------
    _orig_torch_sub = torch.sub

    def torch_sub(input, other, *args, **kw):
        r = _empty_result(input, other, inplace=False)
        if r is not None:
            return r
        return _orig_torch_sub(input, other, *args, **kw)

    torch.sub = _mark(torch_sub)

    # ------------------------------------------------------------------
    # 3) 减号 dunder —— 包装当前实现，不替换
    #    （slot_assign 把 __isub__ 换成了「视图上原地算子走 out-of-place」
    #      的版本，那条是分类分支梯度对齐的前提，不能被顶掉）
    # ------------------------------------------------------------------
    _orig_dunder_sub = torch.Tensor.__sub__
    _orig_dunder_isub = torch.Tensor.__isub__

    def dunder_sub(self, other):
        r = _empty_result(self, other, inplace=False)
        if r is not None:
            return r
        return _orig_dunder_sub(self, other)

    def dunder_isub(self, other):
        r = _empty_result(self, other, inplace=True)
        if r is not None:
            return r
        return _orig_dunder_isub(self, other)

    torch.Tensor.__sub__ = _mark(dunder_sub)
    torch.Tensor.__isub__ = _mark(dunder_isub)
