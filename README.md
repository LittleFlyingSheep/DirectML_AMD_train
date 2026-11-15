# DirectML_AMD_train

**让 ultralytics 的 YOLO 能在 AMD 显卡（DirectX 12）上真正训得起来。**

Windows 上没有 ROCm 的 AMD 卡（尤其是 GCN5 / Vega 20 / gfx906 这类），
PyTorch 能吃到的只有 `torch-directml`。但 DirectML 不是 CUDA，
有一批算子语义和 CPU 不一致：直接跑 ultralytics 的训练，
会「跑得完、loss 也往下走、mAP 就是上不去」。

这个库把这些差异一条条补上，并用可复现的回归测试守住。

---

## 现状

在 **AMD Radeon Pro VII（Vega 20 / gfx906）** 上实测，YOLOv8s 训练 25 轮：

| 指标 | CPU（参考） | A 卡 / DirectML |
| --- | --- | --- |
| mAP50 | 0.995 | **0.995** |
| mAP50-95 | 0.985 | **0.983** |
| Precision | 0.998 | **0.998** |
| Recall | 1.000 | **1.000** |
| 单轮耗时 | ~70 s | **~16 s（约 4 倍加速）** |

同一份数据、同一套超参、同一份随机种子，权重逐参数梯度与 CPU 对齐
（最大相对误差 1e-6 量级）。

> 修好之前，A 卡跑完 25 轮 mAP50 只有 **0.603**，cls_loss 反复震荡到 643。
> 根因见下面的「补丁 11」。

---

## 安装

~~~powershell
# 训练环境（建议独立于日常环境）
pip install torch torchvision ultralytics
pip install torch-directml

# 本体
pip install -e .
~~~

torch-directml 只发 Windows 轮子，且需要 Python 3.8+。

---

## 用法

### 命令行

~~~powershell
# 环境自检 + 梯度数值对齐自检（强烈建议先跑这个）
python -m directml_amd_train doctor

# 训练
python -m directml_amd_train train --data path/to/data.yaml --epochs 25 --device dml

# 导出 ONNX
python -m directml_amd_train export --weights runs/exp/weights/best.pt --out out/
~~~

### 代码里用

~~~python
from directml_amd_train import enable_directml_amd, train_yolo

enable_directml_amd()          # 必须在 import ultralytics 之前
train_yolo("data.yaml", epochs=25, device="dml")
~~~

或者自己接管：

~~~python
from directml_amd_train import enable_directml_amd
enable_directml_amd()
from ultralytics import YOLO          # 补丁打完之后再 import

YOLO("yolov8s.pt").train(data="data.yaml", epochs=25,
                         device="privateuseone:0", amp=False)
~~~

> `enable_directml_amd()` 必须在 `import ultralytics` **之前**调用 ——
> ultralytics 的 `smart_inference_mode` 装饰器在 import 时就把
> `torch.inference_mode` 这个类抓在手里，之后再改就不生效了。

---

## 补丁清单

| # | 补丁 | DirectML 的问题 | 处理方式 |
| --- | --- | --- | --- |
| 1 | `patch_inference_mode` | BatchNorm 原地写缓冲区 + inference tensor 冲突，报 `Cannot set version_counter for inference tensor` | 降级成 `no_grad` |
| 2 | `patch_device` | `select_device` 不认 `privateuseone`，报 `Invalid CUDA device` | 拦下 privateuseone 直通 |
| 3 | `patch_unique` | `unique` 不支持 `return_counts` | 回 CPU 算再搬回 |
| 4 | `patch_scatter` | `scatter_add_` 不支持 | 回 CPU 算再写回 |
| 5 | `patch_bool_ops` | `float32 * bool` 的结果 dtype 被判成 bool、数值被截断成全 False | 非浮点操作数转成对方 dtype |
| 6 | `patch_memory` | `PrivateUse1Module` 没有 `memory_reserved` 等方法，日志直接抛异常 | 补恒 0 实现 |
| 7 | `patch_cat` | `cat` 不能带 0 列张量（报 UnicodeDecodeError） | 跳过空张量 |
| 8 | `patch_gather` | `gather` 的反向把散射累加做成了「取平均」（64 份贡献摊成 1） | 反向在 CPU 上 `scatter_add_` |
| 9 | `patch_bce` | `BCEWithLogits` 内部 `log_sigmoid_forward` 不支持，CPU fallback 后梯度搬不回显卡 | 自实现前反向 |
| 10 | `patch_addmv` | `aten::addmv.out` 不支持，CPU fallback | 改写成 elementwise 形式 |
| 11 | `patch_loss_on_cpu` | **`index_put_` 反向清错槽位（致命）** | **损失计算搬到 CPU** |

### 补丁 11：真正让训练不收敛的那一条

ultralytics 的损失是这么拼装的（`ultralytics/utils/loss.py`）：

~~~python
loss = torch.zeros(3, device=self.device)
loss[1] = bce_loss.sum() / target_scores_sum   # 分类
loss[0], loss[2] = self.bbox_loss(...)          # 回归 / DFL
loss[0] *= self.hyp.box
loss[1] *= self.hyp.cls
loss[2] *= self.hyp.dfl
~~~

DirectML 的 `index_put_`（原地槽位赋值）**反向算错了**：

~~~python
x = torch.zeros(3, requires_grad=True)
y = x.clone()
y[1] = 1.0                 # 覆盖第 1 个槽位
y.sum().backward()
# CPU  -> x.grad = [1, 0, 1]   被覆盖的槽位 1 不该再往回传，正确
# DML  -> x.grad = [0, 1, 1]   清零的成了槽位 0，槽位 1 反而留着
~~~

前向是对的 —— 两边 loss 数值一模一样，所以训练**不会报错**，
只是梯度悄悄错位。于是分类分支（Detect 的 `cv3`）拿到的梯度恒为 0，
`cls_loss` 永远降不下来，模型学不会分类。

**修法**：损失那点计算量（几个小张量的乘加）相对卷积主干可以忽略，
把损失整个搬到 CPU 上用原生 PyTorch 算，梯度再原路搬回显卡。
模型的前向/反向仍然全程跑在显卡上 —— 训练确实是在显卡上做的，
不是「显卡跑一半」。

修完之后 A 卡与 CPU 的逐参数梯度对齐：

~~~
cv3（分类分支）  24/24 个参数梯度非零，max/sum 完全一致
cv2（回归分支）  65/81 非零，max 9257.61 vs 9257.6
其它参数         78/78 非零，max 30312.4 vs 30312.3
scores 钩子梯度  1.2573893 vs 1.2573889
~~~

---

## 回归测试

~~~powershell
pytest tests/ -v
~~~

* `tests/test_patches.py` —— 逐条补丁的语义对照（同一份输入，CPU 与 DirectML 必须一致）
* `tests/test_alignment.py` —— 端到端：逐参数梯度对齐，分类分支必须有梯度

没有 DirectML 的机器上会自动跳过。

---

## 为什么不是 ROCm

| 路线 | 结论 |
| --- | --- |
| ROCm on Linux | ROCm 5.0 起官方不再支持 GCN5/gfx906，6.x 官方矩阵里只有 RDNA2/RDNA3 |
| ROCm on Windows (HIP SDK) | 只发 RDNA3/RDNA4 的轮子，没有 gfx906 的 |
| WSL2 + ROCm | WSL 侧的 ROCm 同样不支持 gfx906 |
| **torch-directml** | **底层走 DirectX 12，任何 D3D12 显卡都能用 —— 本库走的这条** |

这是上游支持矩阵的问题，不是配置问题。

---

## 已知限制

* DirectML 不支持 `autocast`，所以 `amp` 一律关掉（混合精度在这条路上没有加速）。
* `amp` 关掉 + 损失在 CPU 上算，A 卡相对 CPU 的加速比约 4 倍；瓶颈主要在
  DirectML 的算子调度与 kernel 覆盖度，而不是显存带宽。
* 补丁只对 **torch-directml 0.2.5.dev240914** 验证过。换版本请先跑 `doctor`。
* 只覆盖 ultralytics 8.4.x 的检测任务。分割/姿态/OBB 的损失拼装方式不同，
  遇到类似问题请对照 `patch_loss_on_cpu` 的思路自行扩展。

---

## 许可证

MIT，见 [LICENSE](LICENSE)。
