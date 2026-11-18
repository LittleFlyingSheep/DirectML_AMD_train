# DirectML_AMD_train

**让 ultralytics 的 YOLO 能在 AMD 显卡（DirectX 12）上真正训得起来。**

Windows 上没有 ROCm 的 AMD 卡（尤其是 GCN5 / Vega 20 / gfx906 这类），
PyTorch 能吃到的只有 `torch-directml`。但 DirectML 不是 CUDA，
有一批算子语义和 CPU 不一致：直接跑 ultralytics 的训练，
会「跑得完、loss 也往下走、mAP 就是上不去」。

这个库把这些差异一条条补上，并用可复现的回归测试守住。

---

## 现状

在 **AMD Radeon Pro VII（Vega 20 / gfx906）** 上实测，YOLOv8s 训练 25 轮
（512px / batch 8 / 单类别 / 136 张训练图）：

| 指标 | CPU（参考） | A 卡 / DirectML |
| --- | --- | --- |
| mAP50 | 0.993 | **0.995** |
| mAP50-95 | 0.986 | **0.985** |
| Precision | 0.998 | **0.997** |
| Recall | 1.000 | **1.000** |
| 单轮耗时 | ~77 s | **~11.5 ~ 12.2 s（约 6.3 ~ 6.7 倍加速）** |

同一份数据、同一套超参、同一份随机种子，权重逐参数梯度与 CPU 对齐
（聚合相对误差 ~1e-5）。

**损失计算整段跑在显卡上**：分类 BCE、回归 IoU、DFL、标签匹配全在 A 卡上算，
没有一个算子回落到 CPU（只有标签索引用到的 `unique` 例外，它只处理索引，张量很小）。

> 修好之前，A 卡跑完 25 轮 mAP50 只有 **0.603**，cls_loss 反复震荡到 643。
> 根因见下面的「补丁 11」。

---

## CPU 与 A 卡对比

同一台机器、同一份数据、同一套超参、同一份随机种子。
配置：YOLOv8s @ 512px / batch 8 / 1 类，136 张训练图 + 33 张验证图，
显卡 = AMD Radeon Pro VII（Vega 20 / gfx906，DirectX 12）。

| 环节 | CPU | A 卡（DirectML） | 对比 |
| --- | --- | --- | --- |
| 损失前向（带梯度） | 23.8 ms | 51.6 ms | CPU 快 2.2× |
| 损失反向 | 12.9 ms | 21.6 ms | CPU 快 1.7× |
| 模型前向 | 1041 ms | 73 ms | **A 卡快 14×** |
| 模型反向 | 1562 ms | 311 ms | **A 卡快 5×** |
| 一个 iteration（前向 + 损失 + 反向） | 2.75 s | 0.233 s | **A 卡快 11.8×** |
| 整轮训练（17 iteration + 验证 + 数据管线） | 77.2 s | 11.5 ~ 12.2 s | **A 卡快 6.3 ~ 6.7×** |
| 25 轮训练总耗时 | 1930.9 s | 287.7 ~ 305.9 s | **A 卡快 6.3 ~ 6.7×** |
| 最终 mAP50 | 0.99288 | **0.99475** | |
| 最终 mAP50-95 | 0.98586 | 0.9855 | |
| 损失计算在哪算 | CPU | `privateuseone:0`（整段在显卡） | |
| 训练步骤里回落到 CPU 的算子 | — | **0 个** | |

几点说明：

* **损失本身在 CPU 上更快**（24 ms vs 52 ms）。DirectML 每个算子都有固定开销
  （实测 ~0.2 ~ 3 ms），而 ultralytics 的损失是「一堆小算子」堆出来的，
  小模型 + 小输入时 CPU 多线程算这种张量反而占优。
  模型本身则相反 —— 卷积这种大张量在 A 卡上是压倒性的优势（14×）。
* 端到端 6 倍多的加速，主要来自模型前向/反向，而不是损失。
  换句话说：**损失留在 A 卡的收益是「不折腾数据」，不是「更快」**；
  想让损失本身更快，得靠补丁 9/10/12 把那些小算子合并掉。
* 25 轮总耗时在 287.7 ~ 305.9 s 之间浮动是正常的（同一份配置的两次实测
  相差 6%），机器负载和驱动的分配策略都会影响。

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
>
> 调多少次都安全：所有补丁都是幂等的，重复调用不会把已生效的修复顶掉。

---

## 补丁清单

| # | 补丁 | DirectML 的问题 | 处理方式 |
| --- | --- | --- | --- |
| 1 | `patch_inference_mode` | BatchNorm 原地写缓冲区 + inference tensor 冲突，报 `Cannot set version_counter for inference tensor` | 降级成 `no_grad` |
| 2 | `patch_device` | `select_device` 不认 `privateuseone`，报 `Invalid CUDA device` | 拦下 privateuseone 直通 |
| 3 | `patch_unique` | `unique` 不支持 `return_counts` | 回 CPU 算再搬回（张量很小） |
| 4 | `patch_scatter` | `scatter_add_` 不支持，而且原生实现**不累加**（结果是错的） | dim=0 时改用 `index_put_(accumulate=True)`，**留在显卡上** |
| 5 | `patch_bool_ops` | `float32 * bool` 的结果 dtype 被判成 bool、数值被截断成全 False | 非浮点操作数转成对方 dtype |
| 6 | `patch_memory` | `PrivateUse1Module` 没有 `memory_reserved` 等方法，日志直接抛异常 | 补恒 0 实现 |
| 7 | `patch_cat` | `cat` 不能带 0 列张量（报 UnicodeDecodeError） | 跳过空张量 |
| 8 | `patch_gather` | `gather` 的反向把散射累加做成了「取平均」（64 份贡献摊成 1） | 反向在 CPU 上 `scatter_add_` |
| 9 | `patch_bce` | `BCEWithLogits` 内部 `log_sigmoid_forward` 不支持，回落 CPU | 自己用 softplus 公式**在显卡上算** |
| 10 | `patch_addmv` | `aten::addmv.out` 不支持（`matmul(N-D, 1-D)` 在 DML 上正好被拆成它） | 改写成 elementwise 形式 |
| 11 | `patch_slot_assign` | **视图的 `storage_offset` 恒为 0，原地赋值的反向把梯度写错槽位（致命）** | **槽位赋值改走 `index_put_`，视图上的原地算子改成 out-of-place** |
| 12 | `patch_flat_index` | 高级索引（`t[i0, i1, ...]`）带梯度时不是并行 kernel，而是一串 `copy_`（前向 175 ms，`no_grad` 下只要 47 ms） | **折算成扁平偏移 + `gather`（并行 kernel），反向用 `index_put_(accumulate=True)` 在显卡上散射累加** |

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

「往张量的某个位置写值」在 PyTorch 里其实是「写一个视图」，自动微分用
CopySlices 节点记录它。CopySlices 的反向靠「视图相对基张量的偏移量」
定位被覆盖的那一段（`torch/csrc/autograd/functions/tensor.cpp`）：

~~~cpp
auto offset = view.sym_storage_offset() - base.sym_storage_offset();
grad_slice = result.as_strided_symint(view.sym_sizes(), view.sym_strides(), offset);
~~~

**而 DirectML 的视图没有真正的 storage —— `storage_offset()` 恒为 0：**

~~~python
t = torch.arange(12.).reshape(3, 4).to("privateuseone:0")
t[1:2].storage_offset()    # CPU: 4     DirectML: 0
~~~

offset 算成 0，梯度就被写到「第一个」槽位，而不是被覆盖的那个：

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

**修法**：不去改后端算子，而是绕开 CopySlices：

1. 槽位赋值 `t[i] = v` 改走 `index_put_` —— 它的反向在 DirectML 上是对的，
   按 index 直接算梯度，位置不会错；
2. 视图上的原地算子 `t[i] *= v` 改成「out-of-place 计算 + no_grad 写回」——
   原地算子作用在视图上必然产生 CopySlices，那就干脆不产生它。

修完之后，损失计算可以整段留在显卡上，梯度与 CPU 对齐
（`python -m directml_amd_train doctor`）：

~~~
CPU  loss = [4.937076, 7.984281, 8.640423]
DML  loss = [4.937078, 7.984282, 8.640425]
损失计算设备 = privateuseone:0（整段留在显卡上）
参数梯度 183 个，分类分支 24 个（DirectML 上非零 24 个）
梯度范数合计 CPU 164923 / DML 164925，聚合相对误差 9.775e-06
单参数最大相对误差 6.556e-03（model.9.cv1.conv.weight）
~~~

> 早期版本是靠「把损失整段搬到 CPU」绕开这个 bug 的（`patch_loss_on_cpu`）。
> 现在是**真的修好了**，损失留在显卡上算；那块 CPU 路径作为可选开关保留（见下）。

### 补丁 12：让高级索引也吃上并行 kernel

DirectML 的 `aten::index`（多个索引张量的高级索引）并不是并行 kernel，
而是一串 `copy_` / `_to_copy` —— 实测损失前向里出现了 195 次 `copy_`、
162 次 `_to_copy`，`aten::index` 一项就占掉前向 60% 以上的 CPU 时间。
而 `gather` 在 DirectML 上是正经的并行 kernel。

连续张量是行主序的，所以多维索引可以折算成一个扁平偏移量：

~~~python
offset = Σ_j idx_j * stride_j        # 尾部固定维度选中的偏移也算进去
~~~

剩下的尾部维度如果还在，就等于「以 P = 尾部维度乘积为步长的一段连续内存」，
于是一次 `gather` 就能把整段取出来：

~~~python
P == 1 : gather(t.reshape(-1), 0, offset)
P >  1 : gather(t.reshape(-1), 0, (offset[:, None] * P + arange(P)).reshape(-1))
~~~

`gather` 的输出形状等于 **index** 的形状（不是输入的形状），
所以索引先摊成一维、结果再 `view` 回原来的形状。

反向不能直接用 DirectML 的 `gather` —— 它的反向是坏的（把多份贡献摊成 1 份，见补丁 8），
所以补丁 12 自己写反向：在显卡上用 `index_put_(accumulate=True)` 散射累加，
既避开坏掉的反向，也不用把梯度搬回 CPU。

ultralytics 的检测损失里正好有 9 处这样的高级索引
（`ultralytics/utils/tal.py` 与 `utils/loss.py`），每步训练都要走一遍。
单点实测（AMD Radeon Pro VII）：

| 调用点 | 形状 | 原生 `aten::index` | 改写成 `gather` | 提速 |
| --- | --- | --- | --- | --- |
| `tal.py` 匹配打分（3 个索引） | (8, 5376, 2) | 1.2 ms | 0.05 ms | 约 24× |
| `tal.py` 匹配框 | (8, 5376, 4) | 1.0 ms | 0.26 ms | 约 3.8× |
| `loss.py` 取目标框 | (8, 5376, 4) | 1.6 ms | 0.27 ms | 约 5.8× |
| `loss.py` 取 DFL 分布 | (8, 5376, 64) | 0.9 ms | 0.24 ms | 约 3.8× |

损失整体（YOLOv8s @ 512px / batch 8，**带梯度**，也就是训练场景）：

| 显卡命令队列状态 | 原生 `aten::index` | 补丁 12（展平 gather） |
| --- | --- | --- |
| 队列空（迭代之间已排空） | 48.7 ms | 47.9 ms（关越界检查）/ 51.6 ms（开） |
| 队列积压（连续迭代，没排空） | **175 ms** | **53 ms** |

> **补丁 12 的价值不是「单点更快」，而是「快慢稳定」**：
> 队列空的时候两条路差不多（48.7 ms vs 51.6 ms）；
> 队列一深，原生 `aten::index` 那串 `copy_` 就会被拖到 175 ms，
> 而改写成一次 `gather` 之后基本不受队列深度影响（53 ms）。
> 真实训练里队列是半满的，所以原生路径落在两者之间 —— 这才是端到端收益的来源。

> 越界检查（逐维 min/max + 标量回读）本身要花约 3.7 ms/次前向，
> 不需要这份严格性可以 `patch_flat_index(verify_bounds=False)` 关掉。

端到端（同一台机器、同一时段配对跑 25 轮，只差这个开关）：

| 配对 | 补丁 12 开启 | 补丁 12 关闭（原生 `aten::index`） | 提升 |
| --- | --- | --- | --- |
| 第 1 对 | 305.9 s（12.23 s/轮） | 319.5 s（12.78 s/轮） | **4.3%** |
| 第 2 对 | 287.7 s（11.51 s/轮） | 312.5 s（12.50 s/轮） | **7.9%** |

两对跑出来的最终指标**逐位相同**
（box 0.28391 / cls 0.25665 / dfl 0.77045 / mAP50 0.99475 / mAP50-95 0.9855）——
改写只换算法路径，不改数值。

> 端到端的收益只有几个百分点，是因为损失只占一个 iteration 的一部分：
> 512px / batch 8 下模型自身的前向+反向约 380 ms，损失约 75 ms。
> 但这一步之后，**训练步骤里已经没有任何 DirectML 算子回落到 CPU**
> （实测抓取回落告警：0 个）。

**越界语义**：原生 `aten::index` 越界会抛 `IndexError`，
但 DirectML 的 `gather` 越界**不报错，会直接读到野内存**
（实测取到 `[0.0, 9.0]` 这种随机值）。所以补丁 12 自己保留了逐维越界检查：
把索引张量拷回主机查 min/max，一次 DML→CPU 同步约 0.3 ms；
同一个索引张量重复使用时走缓存（张量被原地改过就自动失效）。
不需要这份严格性时可以关掉：

~~~python
from directml_amd_train import patch_flat_index

patch_flat_index(verify_bounds=False)     # 只省掉越界检查，取值语义不变
~~~

---

### 补丁 12 的一次翻车：0 列张量把训练打崩（已修）

这一条是在「悠悠机器人视觉数据集标注 Web UI」那个会话里发现的：
训练跑到第 1 轮验证时，被补丁库自己打崩了 ——
`ZeroDivisionError: integer modulo by zero`。

**根因**：ultralytics 的 NMS 在「没有 mask 系数 / 关键点通道」的时候，
会把预测张量拆成 `box, cls, mask = x.split((4, nc, extra), 1)`，
其中 `extra == 0` 分出来的 `mask` 是 **`(n, 0)` 的 0 列张量**。
`multi_label` 分支里的 `mask[i]` 就是对 0 列张量的高级索引，
而 `detect/val.py` 的验证阶段正好走这条分支（`multi_label=True`）：

~~~python
# ultralytics/utils/nms.py
box, cls, mask = x.split((4, nc, extra), 1)      # extra == 0 -> mask 是 (n, 0)
x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
                                                             ^^^^^^
~~~

补丁 12 折算扁平偏移时算出「尾部保留维度的乘积」`P = 0`，
紧接着去算 `s % P` —— 除零。

**修法**：这类索引根本不该交给改写后的路径，直接回退原生实现。
`gather` 表达不出「空输出」，原生 `aten::index` 的语义反而是完整的 ——
实测原生 DirectML 对 0 列张量返回 `(75, 0)`，与 CPU 完全一致。
顺带修掉两个同类边角：

* 源张量为空（`shape (0, 5)`）时展平后一个元素都没有，`gather` 下标必然越界
  → 回退原生（原生按 PyTorch 语义报 `IndexError`，与 CPU 一致）。
* 索引张量为空时输出必然是空的，而 DirectML 的 `gather` 处理空下标会抛
  `UnicodeDecodeError` → 回退原生。

**触发条件**：类别数 ≥ 2（`multi_label &= nc > 1`，单类别时这条分支根本不走）。
所以单类别数据集训练碰不到，多类别数据集第 1 轮验证必崩。

**验证**：修复前 ultralytics 真实的 `non_max_suppression` 直接除零崩溃，
修复后同一条路径正常返回 300 个检测框，训练 + 验证全程跑通。
回归测试：`tests/test_patches.py::test_ultralytics_nms_zero_column_mask_no_crash`。

---

---

## 损失计算在哪儿算

| 环节 | 在哪儿算 |
| --- | --- |
| 损失拼装（box / cls / dfl 的乘加） | **显卡**（补丁 11 修好了原地赋值的反向） |
| 分类 BCE（`BCEWithLogitsLoss`） | **显卡**（补丁 9 用 softplus 公式自己实现，绕开不支持的 `log_sigmoid_forward`） |
| `bbox_decode` 里的 `matmul` | **显卡**（补丁 10 改写成 elementwise，绕开 `addmv`） |
| 标签索引用的 `scatter_add_` | **显卡**（补丁 4 改用 `index_put_(accumulate=True)`） |
| 标签匹配里的高级索引 `t[i0, i1, ...]` | **显卡**（补丁 12 折算成扁平偏移 + `gather`，吃上并行 kernel） |
| 标签索引用的 `unique` | 回 CPU 算再搬回（补丁 3）—— 只处理索引，张量很小 |

> `amp`（混合精度）仍然关掉：DirectML 不支持 `autocast`。

### 想把损失放回 CPU？

DirectML 每个算子都有固定开销（实测 ~0.2 ~ 3 ms），而 ultralytics 的损失
是「很多小算子」堆出来的。**小模型 + 小输入**时，CPU 多线程算这种小张量
反而更快：

| | 损失前向 | 损失反向 | 每轮（17 iteration） |
| --- | --- | --- | --- |
| 损失在显卡（默认，含补丁 12） | 52 ms | 22 ms | ~11.5 ~ 12.2 s |
| 损失在 CPU | 24 ms | 13 ms | ~10.6 s（但还要额外搬预测张量） |

补丁 12 之前，这一栏是 175 ms（队列积压时）/ 23 ms / ~12.8 s —— 也就是说
**补丁 12 把「损失在显卡」和「损失在 CPU」的差距从 ~2.2 s/轮压到 ~1.2 s/轮**。
小模型 + 小输入时 CPU 多线程算这种一堆小算子仍然略快，但差距不大，
而且默认路径不用把预测张量搬来搬去。

所以保留了一个开关，小模型上可以换回 CPU：

~~~python
from directml_amd_train import enable_directml_amd, train_yolo

enable_directml_amd(loss_on_cpu=True)          # 或者
train_yolo("data.yaml", epochs=25, device="dml", loss_on_cpu=True)
~~~

大模型 / 大输入用默认的显卡路径更划算 —— 损失张量变大之后，
DirectML 的算子开销就不是瓶颈了。

---

## 回归测试

~~~powershell
pytest tests/ -v
~~~

* `tests/test_patches.py` —— 逐条补丁的语义对照（同一份输入，CPU 与 DirectML 必须一致）
* `tests/test_alignment.py` —— 端到端：逐参数梯度对齐，分类分支必须有梯度，损失必须在显卡上算

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
* `t[i:j] = v` 这种**切片**赋值在 DirectML 上仍然会错位（补丁 11 只修了标量槽位）。
  ultralytics 的损失拼装里没有这种写法，训练不受影响；自己写代码时避开即可。
* 更阴的一个坑：DirectML 上**对同一个张量连续做跨步原地写**
  （`t[:, k, :] = v` 写不同的 k），**只有第一次生效**，后面几次静默失效
  （强制同步也救不回来，不同张量则互不影响）。
  合成测试数据时很容易中招 —— 在显卡上 `pred[:, 4, :] = 0.9` 之后再写
  `pred[:, 5, 10:] = 0.8`，前者的值会没了。
  稳妥做法：数据在 CPU 上造好再 `.to(device)`，或者先写进新张量再整体搬过去。
* 标签索引用到的 `unique` 还是会回 CPU 算 —— 张量很小，不是瓶颈。
* DirectML 上「连续反复计算同一个损失」（中间没有任何 CPU 工作插进来）
  会把显卡命令队列卡死 —— 下一次前向的第一个算子永远不返回，
  堆栈停在 `make_anchors` 的 `meshgrid` 上。
  实测**原生 `aten::index` 路径也一样卡**，所以这是后端 / 损失本身的既有问题，
  不是本库引入的。正常训练循环不受影响：迭代之间有数据加载、优化器等
  CPU 工作，队列能排空。
* 补丁 12 只接管「连续的索引张量前缀 + 尾部整数」这一种形态
  （损失里的 9 处都属于这一类）。切片、bool 掩码、0 维索引张量、
  非连续张量会回退原生实现 —— **数值仍然正确**，只是没吃到并行加速。
* 补丁 12 遇到空张量（0 尺寸尾维）或空索引也会回退原生实现：
  尾部维度乘积为 0 时折算偏移会除零崩溃，而 `gather` 表达不出空输出
  （ultralytics 的 NMS 里就有 0 列的 mask 张量，详见上面的说明）。
* 补丁 12 的越界检查是一次 DML→CPU 同步（每个索引点约 0.3 ms）。
  嫌慢可以 `patch_flat_index(verify_bounds=False)` 关掉，
  但那样越界索引会读到野内存而不是报错。
* DirectML 每个算子有 ~0.2~3 ms 的固定开销，**算子很多、每个张量都很小**的计算
  （比如损失）在 CPU 上可能更快 —— 见上面的开关。
* 补丁只对 **torch-directml 0.2.5.dev240914** 验证过。换版本请先跑 `doctor`。
* 只覆盖 ultralytics 8.4.x 的检测任务。分割/姿态/OBB 的损失拼装方式不同，
  遇到类似问题请对照 `patch_slot_assign` 的思路自行扩展。

---

## 许可证

MIT，见 [LICENSE](LICENSE)。
