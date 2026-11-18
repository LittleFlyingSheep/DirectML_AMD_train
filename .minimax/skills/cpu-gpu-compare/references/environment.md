# 环境与后端事实（Direct_AMD_train）

## 跑起来需要什么

解释器不是 venv，是 `anaconda3\envs\maimai\python.exe`（Python 3.11），
torch / ultralytics 靠 `PYTHONPATH` 拼两个 site-packages 目录：

```powershell
$env:PYTHONPATH='<repo>\_debug\envs\yolo_dml_env;<repo>\_debug\envs\yolo_train_env'
& 'C:\Users\<user>\anaconda3\envs\maimai\python.exe' -u -m pytest tests/ -q
& 'C:\Users\<user>\anaconda3\envs\maimai\python.exe' -u -m Direct_AMD_train doctor
```

- `yolo_dml_env`：torch 2.4.1+cpu、torchvision 0.19.1、torch-directml 0.2.5.dev240914
- `yolo_train_env`：torch 2.15.0、ultralytics 8.4.213、pytest 9.2.0
- 数据集：`_debug/data/yolo_dataset`（136 train / 33 val，1 类 card），
  `data_local.yaml` 指向本机路径（原始 data.yaml 的绝对路径已失效）
- 权重：`yolov8s.pt` 在仓库根目录
- `_debug/`、runs/、*.log、*.pt、*.onnx 都在 .gitignore 里，换机器要重搭
- 显卡：AMD Radeon Pro VII（Vega 20 / gfx906），torch_directml 的设备名带尾随空格

## 补丁清单（16 条，分两层；改动前后语义不同）

补丁从 1.x 的 12 条扩到 16 条，并按作用范围分两层（`enable(framework=...)` 控制打哪层）：
**torch 层**改 torch / torch_directml 自己的算子语义，与框架无关；
**ultralytics 层**改 ultralytics 内部行为，按需开启。

torch 层：

| # | 补丁 | DirectML 的问题 | 处理方式 |
| --- | --- | --- | --- |
| 1 | `patch_inference_mode` | BatchNorm 原地写缓冲区 + inference tensor 冲突 | 降级成 `no_grad` |
| 2 | `patch_unique` | `unique` 不支持 `return_counts`；二维以上输入直接崩进程；bool 数值错 | 按情况回 CPU，一维数值类型留在显卡上 |
| 3 | `patch_sort` | 待排值全相等时 UnicodeDecodeError（`sort` / `topk` 一样） | 先试原生，炸了回 CPU 重算 |
| 4 | `patch_scatter` | `scatter_add_` 不累加（结果是错的） | dim=0 改用 `index_put_(accumulate=True)` |
| 5 | `patch_bool_ops` | `float32 * bool` 的 dtype 被判成 bool、数值被截断 | 非浮点操作数转 dtype |
| 6 | `patch_memory` | `PrivateUse1Module` 没有 `memory_reserved` | 补恒 0 实现 |
| 7 | `patch_cat` | `cat` 不能带 0 列张量 | 跳过空张量 |
| 8 | `patch_gather` | `gather` 反向把散射累加做成「取平均」 | 反向在 CPU 上 `scatter_add_` |
| 9 | `patch_bce` | `log_sigmoid_forward` 不支持，回落 CPU | softplus 公式在显卡上算 |
| 10 | `patch_addmv` | `aten::addmv.out` 不支持 | 改写成 elementwise |
| 11 | `patch_slot_assign` | **视图 storage_offset 恒为 0，原地赋值反向写错槽位（致命）** | 槽位赋值走 `index_put_` |
| 12 | `patch_flat_index` | 带梯度时高级索引是一串 `copy_`，队列一深被拖慢 | 折算成扁平偏移 + `gather` |

ultralytics 层：

| # | 补丁 | DirectML 的问题 | 处理方式 |
| --- | --- | --- | --- |
| 13 | `patch_device` | `select_device` 不认 privateuseone | 拦下 privateuseone 直通 |
| 14 | `patch_ultralytics_memory` | 训练进度条问显存，DirectML 没有这个方法 | 恒返回 0 |
| 15 | `patch_probiou` | OBB 的旋转框 IoU 在 DirectML 上不支持 | 回 CPU 算完再搬回显卡 |
| 16 | `patch_loss_on_cpu` | 小模型 / 小输入上损失在 CPU 反而更快 | 可选开关，把损失搬回 CPU |

完整表（带适用任务）看 `python -m Direct_AMD_train patches`。

补丁 11 是「跑得完但不收敛」的根因：CopySlices 反向用
`view.sym_storage_offset() - base.sym_storage_offset()` 定位被覆盖的区间，
而 DirectML 的视图 offset 恒为 0 → 梯度写到第一个槽位 → 分类分支梯度恒为 0。

所有补丁都用 `_once.py` 幂等化：`patch_bool_ops` 会替换 `Tensor.__imul__` 等 dunder，
不幂等的话第二次 `enable()` 会把 `patch_slot_assign` 的修复悄悄顶掉。

## DirectML 算子级事实（实测，别重复试错）

**语义错误（会造成静默错值）**

| 算子 | DirectML 行为 | CPU |
| --- | --- | --- |
| 视图 `storage_offset()` | 恒为 0 | 正确值 |
| `scatter_add_` | 不累加（[1,0,1,0]） | [2,0,1,0] |
| `gather` 反向 | 把多份贡献摊成 1 份 | 正确散射累加 |
| `index_put_`（含 accumulate） | 前向与反向都正确 | 正确 |
| `index_put_`（不带 accumulate） | 正确 | 正确 |

**性能特征**

- 每算子边际开销 ~25 µs；`nonzero` 1.9 ms、`masked_select` 0.8 ms、`unique` 0.7 ms（含同步）
- `aten::index`（多索引张量的高级索引）不是并行 kernel，是一串 `copy_`：
  损失前向里实测 195 次 `copy_` + 162 次 `_to_copy`
- `gather` 是并行 kernel；队列空时两者接近，队列积压时 `aten::index` 慢 3 倍以上
- 标量回读 ~0.2 ms；整块 D2H 拷贝（`t.to("cpu")`）也 ~0.3 ms 但要搬整条张量

**CPU 回落（会打断显卡流水线）**

`bincount`、`index_select`、`aten::addmv.out`、`aten::log_sigmoid_forward`、
`aten::index_add.out`、`aten::aminmax.out`。用
`warnings.catch_warnings(record=True)` 抓 "fall back to run on the CPU" 可以定位。

**原生可用**：sigmoid / exp / log1p / abs / clamp / softplus / masked_select / topk /
sort / masked_fill / where / maximum / logsumexp / silu / square / div / index / gather /
index_put_。

## 已知限制

- DirectML 不支持 `autocast`，`amp` 一律关掉
- `t[i:j] = v` 这种切片赋值在 DirectML 上仍然错位（补丁 11 只修了标量槽位）
- 标签索引用到的 `unique` 仍回 CPU（张量很小）
- **连续反复计算同一个损失**（中间没有任何 CPU 工作）会卡死显卡命令队列，
  表现为下一次前向的第一个算子永远不返回。原生 `aten::index` 路径同样卡，
  是后端/损失本身的既有问题。训练循环不受影响（迭代之间有数据加载等 CPU 工作）
- 小模型 + 小输入时，损失在 CPU 上反而更快（DirectML 每算子固定开销）
- 只对 torch-directml 0.2.5.dev240914 验证过；换版本先跑 `doctor`
