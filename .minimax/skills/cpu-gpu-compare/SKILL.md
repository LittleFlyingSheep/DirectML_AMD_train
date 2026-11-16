---
name: cpu-gpu-compare
description: |
  在 DirectML_AMD_train 仓库里做「CPU 计算 vs A 卡（DirectML）计算」的对比与瓶颈定位：
  组件级耗时、端到端配对 A/B、数值对齐验证，最后产出可交付的对比表。
  触发短语："CPU 和 A 卡对比"、"CPU 与 GPU 性能对比"、"这个补丁值多少"、"损失在哪算更快"、
  "DirectML 加速比"、"benchmark the patches"、"CPU vs GPU comparison"、"损失计算在哪儿算"。
  不要用于：与性能无关的功能开发或 bug 修复、只想跑一次训练看结果（那直接用 train_yolo / doctor）。
displayNames:
  zh-Hans: "CPU_A卡计算对比"
---

# CPU_A卡计算对比

在本仓库里回答两类问题：**「这段计算在 CPU 和 A 卡上各要多久」**，
**「某个补丁到底值多少」**。本仓库的 DirectML 后端有一堆语义差异和性能陷阱，
盲测出来的数字经常是假的，所以下面的步骤顺序本身就是重点。

## Inputs to collect

- 对比的两个配置是什么（例如「补丁 12 开/关」「CPU vs privateuseone:0」）
- 对比口径：只要组件级耗时，还是要端到端训练时长
- 训练类对比用的数据集路径与超参（仓库里现成的本地数据集在
  `_debug/data/yolo_dataset/data_local.yaml`，已被 .gitignore 掉，本机才有）

## Procedure

### 1. 先清场：把占显存的残留进程清掉

每个 DirectML 上下文都会占显存，而这张卡（AMD Radeon Pro VII）DirectML 只看得到 4 GB。
上一轮跑挂的脚本会留下 python 进程继续占着显存，让后续跑批报
`Could not allocate tensor ... not enough GPU video memory`，
或者表现为「算子永远不返回」的假卡死 —— 那不是代码问题。

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  ForEach-Object { Invoke-CimMethod -InputObject $_ -MethodName Terminate | Out-Null }
```

只清本仓库脚本起的进程；先看 `CommandLine` 确认，别误杀用户自己的任务。

### 2. 数值对齐先过关，否则耗时对比没有意义

```powershell
$env:PYTHONPATH='<repo>\_debug\envs\yolo_dml_env;<repo>\_debug\envs\yolo_train_env'
& '<anaconda>\envs\luna\python.exe' -u -m pytest tests/ -q
& '<anaconda>\envs\luna\python.exe' -u -m directml_amd_train doctor
```

doctor 要看到：损失设备 = `privateuseone:0`、分类分支梯度 24/24 非零、
聚合相对误差 ~1e-5。数值没对齐就停在这里修语义，别急着比速度 ——
本仓库一半的坑都是「跑得完但不收敛」。

### 3. 组件级计时：每个配置独立进程 + 迭代间排空队列

```powershell
& '<anaconda>\envs\luna\python.exe' -u <skill>\scripts\bench_components.py cpu
& '<anaconda>\envs\luna\python.exe' -u <skill>\scripts\bench_components.py dml
```

脚本已经处理掉两个必须处理的细节，别在临时脚本里漏掉：

- **分进程**：CPU 多线程会把同一进程里后面的 DML 计时拖慢（同进程测出来的
  A 卡损失前向能虚高到 279 ms），所以 CPU 和 DML 必须各跑一次进程。
- **迭代间排空队列**：DirectML 上「连续反复算同一个损失」会卡死命令队列
  （下一次前向的第一个算子永远不返回，堆栈停在 `make_anchors` 的 `meshgrid`）。
  实测**原生 `aten::index` 路径同样卡**，是后端/损失本身的既有问题，不是补丁引入的。
  脚本在每次迭代后做一次标量回读把队列排空，且不计入计时区间。

产出五个数：损失前向、损失反向、模型前向、模型反向、一个 iteration。

### 4. 端到端配对 A/B：同机同时段，只差一个开关

```powershell
& '<anaconda>\envs\luna\python.exe' -u <skill>\scripts\bench_ab.py --epochs 25
```

规则（不遵守就会得出错误结论）：

- **配对**：同一台机器、同一时段，只差被测开关，其它完全一致。
- **至少两对**：同一配置的两次实测能差 6%（287.7 s vs 305.9 s 都出现过）。
  一对看不出 4~8% 这种量级。
- **超参对齐基线**：`mosaic=0 fliplr=0 scale=0.3 hsv_s=0.4 workers=0`，
  否则 ultralytics 默认的 mosaic=1.0 会把数据管线变成混淆项。
- **对指标**：改写类补丁的最终指标应当**逐位相同**（改写只换算法路径不改数值）。
  不相同说明数值语义变了，先停下修。

### 5. 解读：三个必须避开的陷阱

详见 `references/baseline.md`。要点：

- **队列积压会放大差距**。「单点快 N 倍」常常只反映了队列状态：
  同一个高级索引，队列空时原生 48.7 ms / 改写 51.6 ms，队列积压时原生 175 ms / 改写 53 ms。
  报结论时**必须同时给「队列空」和「队列积压」两栏**，否则等于没测。
- **端到端收益 ≪ 组件收益**。损失只占一个 iteration 的一部分
  （512px/batch8：模型前后向 ~380 ms，损失 ~75 ms），组件快 3 倍，端到端只有几个百分点。
- **小张量上 CPU 更快**。DirectML 每算子固定开销 ~0.2~3 ms，
  ultralytics 的损失是一堆小算子堆出来的，CPU 多线程算这种张量反而占优。
  「损失留在 A 卡」的收益是「不折腾数据」，不是「更快」。

### 6. 产出对比表

列固定为 `环节 | CPU | A 卡（DirectML） | 对比`，行至少覆盖：
损失前向、损失反向、模型前向、模型反向、一个 iteration、整轮训练、
25 轮总耗时、最终 mAP50 / mAP50-95、损失计算在哪算、训练步骤里回落到 CPU 的算子数。
表下面写清测量条件与已知限制。

## Output contract

- 对比表（markdown），CPU 与 A 卡两列 + 差值/倍数
- 一句话结论：端到端快多少、瓶颈在哪
- 数值对齐结论（doctor 的关键行）
- 已知限制与复现命令

## Failure handling

- 报显存不足 / 算子不返回 → 先查残留进程（步骤 1）；若已清场仍复现，
  用原生路径（关掉被测补丁）再跑一次：两边都卡就是后端既有问题，不是这次改动引入的。
- doctor 数值不对齐 → 停，先修语义再谈速度，本仓库的语义补丁优先级高于一切性能优化。
- 两对 A/B 结论方向相反 → 如实报「收益在噪声区间内」，不要挑好看的数字写进 README。

## Examples

Input: 「补丁 12 到底值多少？给个 CPU 和 A 卡的对比表」
Output: 组件级五项耗时 + 两对配对 A/B（4.3% / 7.9%）+ 队列空/积压两栏 +
      「端到端几个百分点，瓶颈在模型前后向而不是损失」+ doctor 对齐结论。

## Windows (win32) platform notes

- 本仓库只在 Windows 上跑（torch-directml 只发 Windows 轮子），命令一律用 PowerShell。
- 解释器是 `C:\Users\<user>\anaconda3\envs\luna\python.exe`（Python 3.11），
  torch/ultralytics 不在 venv 里，靠 `PYTHONPATH` 拼两个 site-packages 目录。
- 训练环境与数据集在 `_debug/` 下（被 gitignore），换机器要按
  `references/environment.md` 重搭。
- 写中文文件一律用 UTF-8；PowerShell 5.1 读文件默认 GBK，会显示乱码
  （用 read 工具或 `-Encoding UTF8`）。
