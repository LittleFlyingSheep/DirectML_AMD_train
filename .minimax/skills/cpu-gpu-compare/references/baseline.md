# 已测基线与解读陷阱

基线机器：AMD Radeon Pro VII（Vega 20 / gfx906），Windows，torch-directml 0.2.5.dev240914。
配置：YOLOv8s @ 512px / batch 8 / 1 类（nc=2），136 训练 + 33 验证，workers=0，
mosaic=0 fliplr=0 scale=0.3 hsv_s=0.4。

## CPU 与 A 卡对比（当前版本实测）

| 环节 | CPU | A 卡（DirectML） | 对比 |
| --- | --- | --- | --- |
| 损失前向（带梯度） | 23.8 ms | 51.6 ms | CPU 快 2.2× |
| 损失反向 | 12.9 ms | 21.6 ms | CPU 快 1.7× |
| 模型前向 | 1041 ms | 73 ms | **A 卡快 14×** |
| 模型反向 | 1562 ms | 311 ms | **A 卡快 5×** |
| 一个 iteration（前向+损失+反向） | 2.75 s | 0.233 s | **A 卡快 11.8×** |
| 整轮训练（17 iteration + 验证 + 数据管线） | 77.2 s | 11.5 ~ 12.2 s | **A 卡快 6.3 ~ 6.7×** |
| 25 轮总耗时 | 1930.9 s | 287.7 ~ 305.9 s | **A 卡快 6.3 ~ 6.7×** |
| 最终 mAP50 | 0.99288 | **0.99475** | |
| 最终 mAP50-95 | 0.98586 | 0.9855 | |
| 损失计算在哪算 | CPU | `privateuseone:0` | |
| 训练步骤里回落到 CPU 的算子 | — | **0 个** | |

**一句话结论**：端到端 6 倍多的加速几乎全部来自模型前向/反向（卷积这种大张量），
不是损失。损失留在 A 卡的收益是「不折腾数据」，不是「更快」。

## 补丁 12（高级索引 → 展平 gather）的 A/B

配对跑（只差 `flat_index` 开关），25 轮：

| 配对 | 开启 | 关闭（原生 `aten::index`） | 提升 |
| --- | --- | --- | --- |
| 第 1 对 | 305.9 s（12.23 s/轮） | 319.5 s（12.78 s/轮） | 4.3% |
| 第 2 对 | 287.7 s（11.51 s/轮） | 312.5 s（12.50 s/轮） | 7.9% |

四次训练的最终指标**逐位相同**（box 0.28391 / cls 0.25665 / dfl 0.77045 /
mAP50 0.99475 / mAP50-95 0.9855）—— 改写只换算法路径，不改数值。

## 陷阱一：队列积压会放大差距

同一个损失前向（YOLOv8s@512/batch8，带梯度）：

| 显卡命令队列状态 | 原生 `aten::index` | 补丁 12（展平 gather） |
| --- | --- | --- |
| 队列空（迭代之间已排空） | 48.7 ms | 51.6 ms（关越界检查 47.9 ms） |
| 队列积压（连续迭代、没排空） | **175 ms** | **53 ms** |

所以「高级索引快 3.3 倍」这个说法只在队列积压时成立。
真实训练队列是半满的，原生路径落在两者之间 —— 这才是端到端只有几个百分点的原因。
**报结论时必须同时给两栏**，否则等于没测。

越界检查（逐维 min/max + 标量回读）本身约 3.7 ms/次前向，
`patch_flat_index(verify_bounds=False)` 可关。

## 陷阱二：组件收益 ≫ 端到端收益

512px / batch8：模型自身前向+反向 ~380 ms，损失 ~75 ms。
组件上快 3 倍，折算到整轮只有 4~8%。写 README 时别把组件数字当端到端数字。

## 陷阱三：同配置两次实测能差 6%

`flat_index=True` 两次实测：305.9 s 与 287.7 s（差 6%）。
所以任何小于 ~6% 的端到端差异都不可信 —— 要配对跑，而且至少两对。

## 复现命令

```powershell
$env:PYTHONPATH='<repo>\_debug\envs\yolo_dml_env;<repo>\_debug\envs\yolo_train_env'
Set-Location '<repo>'

# 组件级（两个进程）
& 'C:\Users\<user>\anaconda3\envs\luna\python.exe' -u .minimax\skills\cpu-gpu-compare\scripts\bench_components.py cpu
& 'C:\Users\<user>\anaconda3\envs\luna\python.exe' -u .minimax\skills\cpu-gpu-compare\scripts\bench_components.py dml

# 端到端配对（~11 分钟）
& 'C:\Users\<user>\anaconda3\envs\luna\python.exe' -u .minimax\skills\cpu-gpu-compare\scripts\bench_ab.py --epochs 25

# 数值对齐
& 'C:\Users\<user>\anaconda3\envs\luna\python.exe' -u -m directml_amd_train doctor
```

训练产物在 `_debug/runs/<name>/results.csv`，`time` 列是累计秒数，
最后一行是总耗时。
