# Direct_AMD_Train

**让 PyTorch 在 AMD 显卡（DirectX 12）上真正训得起来 —— Windows 上的「ROCm 替身」。**

Windows 上没有 ROCm 的 AMD 卡（GCN5 / Vega 20 / gfx906 这类），PyTorch 能吃到的
只有 `torch-directml`。但 DirectML 不是 CUDA：算子支持是打折的，语义也和
安全完整的 CUDA 不一样。直接跑训练会出三类问题：

* **静默算错** —— 跑得完、loss 也往下走，但某个分支的梯度恒为 0，模型学不会分类；
* **直接崩** —— `torch.unique` 传二维张量，整个进程被原生层断言带走（0xC0000409）；
* **悄悄回落 CPU** —— 算子不支持时自动回落到 CPU，训练能跑，但速度掉一半。

这个库把这些差异一条条补上，并且：

* 用 **36 条可复现的回归测试**守住 —— 同一份输入，CPU 与 DirectML 必须一致；
* 自带 **算子能力探测**（`probe`）—— 换 torch-directml 版本时跑一遍，就知道有没有新的坑；
* 自带 **梯度数值对齐自检**（`doctor`）—— 训练能不能收敛，看它就知道；
* 补丁**分两层**：torch 层与框架无关（任何 PyTorch 项目都能用），ultralytics 层按需开启。

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
python -m Direct_AMD_Train doctor

# 算子能力探测：DirectML 上哪些算子能用、哪些被补丁救回来了
python -m Direct_AMD_Train probe
python -m Direct_AMD_Train probe --native      # 测原生能力（不打补丁）

# 补丁清单：每条补丁修的是什么问题、属于哪一层
python -m Direct_AMD_Train patches

# 训练
python -m Direct_AMD_Train train --data path/to/data.yaml --epochs 25 --device dml

# 导出 ONNX
python -m Direct_AMD_Train export --weights runs/exp/weights/best.pt --out out/
~~~

### 代码里用

~~~python
from Direct_AMD_Train import enable, train_yolo

enable()          # 必须在 import ultralytics 之前
train_yolo("data.yaml", epochs=25, device="dml")
~~~

或者自己接管：

~~~python
from Direct_AMD_Train import enable
enable()
from ultralytics import YOLO          # 补丁打完之后再 import

YOLO("yolov8s.pt").train(data="data.yaml", epochs=25,
                         device="privateuseone:0", amp=False)
~~~

> `enable()` 必须在 `import ultralytics` **之前**调用 ——
> ultralytics 的 `smart_inference_mode` 装饰器在 import 时就把
> `torch.inference_mode` 这个类抓在手里，之后再改就不生效了。
>
> 调多少次都安全：所有补丁都是幂等的，重复调用不会把已生效的修复顶掉。

### 纯 PyTorch 项目（不装 ultralytics）

库的 torch 层与框架无关，任何在 DirectML 上跑 PyTorch 的项目都能用：

~~~python
from Direct_AMD_Train import enable

enable(framework="torch")     # 只打 torch 层，绝不碰 ultralytics
~~~

| `framework` | 打哪些补丁 |
| --- | --- |
| `"auto"`（默认） | 装了 ultralytics 就打两层，没装就只打 torch 层 |
| `"torch"` | 只打 torch 层（纯 PyTorch 项目用这个） |
| `"ultralytics"` | 两层都打（会 import ultralytics） |
| `"none"` | 一条都不打（只做环境探测时用） |

---

## 能力矩阵

### 算子能力（`python -m Direct_AMD_Train probe`）

本机（torch-directml 0.2.5.dev240914）实测 95 项常用算子：

| | 通过 | 失败 |
| --- | --- | --- |
| 原生 DirectML | 88 | 3 失败 + 4 会炸/会崩（已跳过） |
| 打上本库的补丁后 | **94** | 1（`conv3d`） |

原生跑不通、打完补丁能跑的那些：

| 算子 | 原生表现 | 补丁怎么救 |
| --- | --- | --- |
| `unique`（二维以上） | **进程硬崩**（0xC0000409，原生层断言） | 回 CPU 算 |
| `unique`（bool） | 数值错（少一个 `False`） | 回 CPU 算 |
| `unique(return_counts=True)` | `RuntimeError` | 回 CPU 算 |
| `scatter_add_` | 不支持，且原生实现**不累加**（数值错） | 改走 `index_put_(accumulate=True)` |
| `argsort` / `sort` / `topk`（待排的值全相等） | `UnicodeDecodeError` | 先试原生，炸了回 CPU 重算 |
| `cat`（带 0 列张量） | `UnicodeDecodeError` | 跳过空张量 |
| `batch_probiou`（OBB 的旋转框 IoU） | `UnicodeDecodeError` | 回 CPU 算 |

以下算子**功能正常但会回落 CPU**（有 warning，不影响正确性，只是慢）：
`addmv`、`log_sigmoid_forward`（BCE 内部）、`std`、`roll`、`take`、`unique_consecutive`。

唯一真正用不了的：**`conv3d`**（DirectML 的卷积只接受 4 维输入），
所以 3D 卷积 / 视频类模型本库暂不支持。

### 任务支持（实测梯度对齐）

每个任务都用 ultralytics 官方模型 + 官方损失，跑一次真实前向 + 反向，
与 CPU 逐参数比对（`python -m Direct_AMD_Train doctor` 是 detect 任务的加强版）：

| 任务 | loss（CPU vs DirectML） | 梯度聚合相对误差 | 被清零的参数 |
| --- | --- | --- | --- |
| detect | 逐位相同 | 1.55e-06 | 0 |
| segment | 逐位相同 | 1.25e-06 | 0 |
| pose | 逐位相同 | 1.95e-04 | 0 |
| obb | 逐位相同 | 2.42e-06 | 0 |
| classify | 逐位相同 | 3.03e-07 | 0 |

「被清零的参数」这一列最要紧：DirectML 的原生实现会让某些分支的梯度恒为 0，
loss 曲线看着正常但模型就是学不会那一类 —— 这正是本库最初要解决的问题。

---

## 补丁体系

补丁分两层，这是这个库从「给 ultralytics 打补丁」升级成「通用 DirectML 兼容层」的关键：

| 层 | 改什么 | 谁需要 |
| --- | --- | --- |
| **torch 层** | torch / torch_directml 自己的算子语义 | 任何在 DirectML 上跑 PyTorch 的项目 |
| **ultralytics 层** | ultralytics 的内部行为（设备选择、显存日志、OBB 的 probiou、损失搬 CPU） | 只有用 ultralytics 时 |

完整清单用 `python -m Direct_AMD_Train patches` 打出来，库里也有一份注册表
（`Direct_AMD_train/registry.py`），每条补丁都记着：影响的算子、症状、修法、适用任务。

| # | 补丁 | 层 | DirectML 的问题 | 处理方式 |
| --- | --- | --- | --- | --- |
| 1 | `patch_inference_mode` | torch | BatchNorm 原地写缓冲区与 inference tensor 冲突 | 降级成 `no_grad` |
| 2 | `patch_unique` | torch | 多维输入**直接崩进程**；`return_counts` 不支持；bool 数值错 | 按情况回 CPU |
| 3 | `patch_sort` | torch | 待排的值全相等时 `UnicodeDecodeError` | 先试原生，炸了回 CPU 重算 |
| 4 | `patch_scatter` | torch | `scatter_add_` 不支持，且原生实现**不累加** | 改走 `index_put_(accumulate=True)` |
| 5 | `patch_bool_ops` | torch | `float32 * bool` 的 dtype 被判成 bool、数值被截断 | 非浮点操作数转成对方 dtype |
| 6 | `patch_memory` | torch | `PrivateUse1Module` 没有显存相关方法 | 补恒 0 实现 |
| 7 | `patch_cat` | torch | `cat` 不能带 0 列张量 | 跳过空张量 |
| 8 | `patch_gather` | torch | `gather` 的反向把散射累加做成「取平均」 | 反向在 CPU 上 `scatter_add_` |
| 9 | `patch_bce` | torch | `BCEWithLogits` 内部 `log_sigmoid_forward` 不支持 | 自己用 softplus 公式**在显卡上算** |
| 10 | `patch_addmv` | torch | `aten::addmv.out` 不支持 | 改写成 elementwise 形式 |
| 11 | `patch_slot_assign` | torch | **视图 `storage_offset` 恒为 0，原地赋值的反向把梯度写错槽位（致命）** | **槽位赋值改走 `index_put_`，视图上的原地算子改成 out-of-place** |
| 12 | `patch_flat_index` | torch | 高级索引不是并行 kernel，而是一串 `copy_` | **折算成扁平偏移 + `gather`** |
| 13 | `patch_device` | ultralytics | `select_device` 不认 `privateuseone` | 拦下 privateuseone 直通 |
| 14 | `patch_ultralytics_memory` | ultralytics | 进度条问显存，DirectML 没这个方法 | 恒返回 0 |
| 15 | `patch_probiou` | ultralytics | OBB 的旋转框 IoU 不支持 | 回 CPU 算 |
| 16 | `patch_loss_on_cpu` | ultralytics | 小模型上损失在 CPU 反而更快 | 可选开关，默认关 |

### 补丁 11：真正让训练不收敛的那一条

ultralytics 的损失是这么拼装的（`ultralytics/utils/loss.py`）：

~~~python
loss = torch.zeros(3, device=self.device)
loss[1] = bce_loss.sum() / target_scores_sum   # 分类
loss[0], loss[2] = self.bbox_loss(...)          # 回归 / DFL
loss[0] *= self.hyp.box
~~~

「往张量的某个位置写值」在 PyTorch 里是「写一个视图」，自动微分用 CopySlices 节点记录它。
CopySlices 的反向靠「视图相对基张量的偏移量」定位被覆盖的那一段，而 DirectML 的视图
根本没有真正的 storage —— 实测：

~~~python
t = torch.arange(12.).reshape(3, 4).to('privateuseone:0')
t[1:2].storage_offset()    CPU: 4     DirectML: 0    ← 永远是 0
~~~

于是梯度被写到「第一个」槽位，而不是被覆盖的那个：

~~~python
x = torch.zeros(3, requires_grad=True)
y = x.clone()
y[1] = 1.0
y.sum().backward()
CPU  -> x.grad = [1, 0, 1]   被覆盖的槽位 1 不该再往回传，正确
DML  -> x.grad = [0, 1, 1]   清零的成了槽位 0，槽位 1 反而留着
~~~

前向是对的（loss 数值两边一模一样），所以训练不会报错，只是梯度悄悄错位 ——
分类分支（Detect 的 cv3）拿到的梯度恒为 0，cls_loss 永远降不下来。

修法：槽位赋值改走 `index_put_`（反向不走 CopySlices，直接按 index 算梯度）；
视图上的原地算子（`t[i] *= v`）改成「out-of-place 计算 + no_grad 写回」，
干脆不产生 CopySlices 节点。

修完之后损失可以整段留在显卡上，梯度与 CPU 对齐（`python -m Direct_AMD_Train doctor`）：

~~~
CPU  loss = [4.937076, 7.984281, 8.640423]
DML  loss = [4.937078, 7.984282, 8.640425]
损失计算设备 = privateuseone:0（整段留在显卡上）
参数梯度 183 个，分类分支 24 个（DirectML 上非零 24 个）
梯度范数合计 CPU 164923 / DML 164925，聚合相对误差 9.775e-06
单参数最大相对误差 6.556e-03（model.9.cv1.conv.weight）
~~~

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
所以补丁 12 自己写反向：在显卡上用 `index_put_(accumulate=True)` 散射累加。

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

端到端（同一台机器、同一时段配对跑 25 轮，只差这个开关）：

| 配对 | 补丁 12 开启 | 补丁 12 关闭（原生 `aten::index`） | 提升 |
| --- | --- | --- | --- |
| 第 1 对 | 305.9 s（12.23 s/轮） | 319.5 s（12.78 s/轮） | **4.3%** |
| 第 2 对 | 287.7 s（11.51 s/轮） | 312.5 s（12.50 s/轮） | **7.9%** |

两对跑出来的最终指标**逐位相同**
（box 0.28391 / cls 0.25665 / dfl 0.77045 / mAP50 0.99475 / mAP50-95 0.9855）。

**越界语义**：原生 `aten::index` 越界会抛 `IndexError`，
但 DirectML 的 `gather` 越界**不报错，会直接读到野内存**。所以补丁 12 自己保留了逐维越界检查
（一次 DML→CPU 同步约 0.3 ms，同一个索引张量重复使用时走缓存）。
不需要这份严格性时可以关掉：

~~~python
from Direct_AMD_train import patch_flat_index

patch_flat_index(verify_bounds=False)     # 只省掉越界检查，取值语义不变
~~~

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

补丁 12 折算扁平偏移时算出「尾部保留维度的乘积」`P = 0`，紧接着去算 `s % P` —— 除零。

**修法**：这类索引回退原生实现。`gather` 表达不出空输出，
原生 `aten::index` 的语义反而是完整的 —— 实测 DirectML 对 0 列张量返回 `(75, 0)`，
与 CPU 完全一致。

**触发条件**：类别数 ≥ 2（`multi_label &= nc > 1`，单类别时这条分支根本不走）。
所以单类别数据集训练碰不到，多类别数据集第 1 轮验证必崩。

回归测试：`tests/test_patches.py::test_ultralytics_nms_zero_column_mask_no_crash`。

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
  换句话说：**损失留在 A 卡的收益是「不折腾数据」，不是「更快」**。
* 25 轮总耗时在 287.7 ~ 305.9 s 之间浮动是正常的（同一份配置的两次实测相差 6%）。

---

## 已知限制

* **`conv3d` 用不了**：DirectML 的卷积只接受 4 维输入，
  3D 卷积 / 视频类模型本库暂不支持。
* `t[i:j] = v` 这种**切片**赋值在 DirectML 上仍然会错位（补丁 11 只修了标量槽位）。
  ultralytics 的损失拼装里没有这种写法，训练不受影响；自己写代码时避开即可。
* 更阴的一个坑：DirectML 上**对同一个张量连续做跨步原地写**
  （`t[:, k, :] = v` 写不同的 k），**只有第一次生效**，后面几次静默失效
  （强制同步也救不回来，不同张量则互不影响）。
  合成测试数据时很容易中招 —— 在显卡上 `pred[:, 4, :] = 0.9` 之后再写
  `pred[:, 5, 10:] = 0.8`，前者的值会没了。
  稳妥做法：数据在 CPU 上造好再 `.to(device)`，或者先写进新张量再整体搬过去。
* 连续反复计算同一个损失会把显卡命令队列卡死
  （下一次前向的第一个算子永远不返回，堆栈停在 `make_anchors` 的 `meshgrid` 上）。
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
  （比如损失）在 CPU 上可能更快 —— 见下面的开关。
* 补丁只对 **torch-directml 0.2.5.dev240914** 验证过。换版本请先跑 `doctor` 和 `probe`。

### 小模型上想让损失回 CPU

保留了一个可选开关：

~~~python
from Direct_AMD_train import enable

enable(loss_on_cpu=True)     # 损失回 CPU 算，模型前后向仍在显卡
~~~

---

## 从 1.x 迁到 2.0

2.0 是**独立开发版**：包名、入口函数、补丁体系都变了。

| 1.x | 2.0 |
| --- | --- |
| `from directml_amd_train import enable_directml_amd` | `from Direct_AMD_train import enable` |
| `python -m directml_amd_train doctor` | `python -m Direct_AMD_train doctor` |
| `pip install -e .`（包名 `directml-amd-train`） | `pip install -e .`（包名 `Direct_AMD_Train`） |
| 补丁没有分层 | torch 层 / ultralytics 层，`enable(framework=...)` 选 |
| 只有 `doctor` | 多一个 `probe`（算子能力表）和 `patches`（补丁清单） |
| 12 条补丁 | 16 条（新增 unique 加固、sort 兜底、probiou、ultralytics 显存） |

**旧名不再保留**（没有兼容层），引用旧包名的项目要一起改。
函数名 `enable_directml_amd` 也换成了 `enable`。

---

## 开发

~~~powershell
# 回归测试（36 条）
pytest tests/ -q

# 环境 + 梯度对齐自检
python -m Direct_AMD_train doctor

# 算子能力表
python -m Direct_AMD_train probe
~~~

许可证：MIT。

