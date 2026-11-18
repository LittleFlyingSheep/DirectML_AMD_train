---
name: cpu-gpu-compare
description: |
  在 Direct_AMD_train 仓库里做「CPU 计算 vs A 卡（DirectML）计算」的对比与瓶颈定位：
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

### 0. 先查证据台账：不要抄没有产物佐证的数字

`references/baseline.md` 的 §7 是一张**证据台账**：每个数字都标了它是「实测 / 投影 /
旧值无佐证」。

**引用本仓库的任何基线数字之前，先去台账里查那一条。**
1.x 版本的文档里有几个数字是「抄下来就信」的，实测发现其中三个站不住：

- CPU 整轮 77.2 s → 实测 52.81 s（**偏了 46%**，而且旧值没有任何落盘产物）
- 「损失本体 175 ms → 53 ms（3.3 倍）」→ 只在一种特定的队列口径下成立，
  换个口径能得出**相反**结论（见步骤 5）
- 「`aten::index` 占前向 60% 以上 CPU 时间」→ 在 ultralytics 8.4.213 上不成立

宁可自己重跑一遍，也不要引用台账里标着「旧值 / 无佐证」的数字。

### 1. 先清场：把占显存的残留进程清掉

每个 DirectML 上下文都会占显存，而这张卡（AMD Radeon Pro VII）DirectML 只看得到 4 GB。
上一轮跑挂的脚本会留下 python 进程继续占着显存，让后续跑批报
`Could not allocate tensor ... not enough GPU video memory`，
或者表现为「算子永远不返回」的假卡死 —— 那不是代码问题。

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'bench_components|bench_ab|train_yolo|DirectML_AMD_train' } |
  ForEach-Object { Invoke-CimMethod -InputObject $_ -MethodName Terminate | Out-Null }
```

**必须按 `CommandLine` 过滤，不要无差别杀掉所有 `python.exe`** —— 这台机器上可能还开着
别的会话的任务（别的项目、别的解释器），无差别清场会把它们一起带走。
先 `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine`
看清楚再杀，只杀本仓库脚本起的那些。

### 2. 数值对齐先过关，否则耗时对比没有意义

```powershell
$env:PYTHONPATH='<repo>\_debug\envs\yolo_dml_env;<repo>\_debug\envs\yolo_train_env'
& '<anaconda>\envs\maimai\python.exe' -u -m pytest tests/ -q
& '<anaconda>\envs\maimai\python.exe' -u -m Direct_AMD_train doctor
```

doctor 要看到：损失设备 = `privateuseone:0`、分类分支梯度 24/24 非零、
聚合相对误差 ~1e-5。数值没对齐就停在这里修语义，别急着比速度 ——
本仓库一半的坑都是「跑得完但不收敛」。

还要**逐位**证明「改写类补丁不改数值」：

```powershell
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_align.py
```

它在一个进程里跑三遍（CPU / A卡补丁开 / A卡补丁关），逐参数比对梯度。
正常结果：补丁开 vs 关 **183/183 参数逐位相同**、differing_elements = 0；
CPU vs A 卡是浮点噪声量级（聚合相对误差 ~1e-6）。

> 脚本里有个顺序陷阱，改的时候别踩：原生 `Tensor.__getitem__` 必须在 `enable()`
> **之前**捕获。补丁 12 是靠替换 `Tensor.__getitem__` 生效的，
> enable 之后捕获就等于拿打过补丁的版本去「还原」，
> 第 3 遍其实还是打了补丁，比对结果必然「逐位相同」—— 自己跟自己比。

### 3. 组件级计时：每个配置独立进程 + 迭代间排空队列

```powershell
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_components.py cpu
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_components.py dml
```

脚本已经处理掉两个必须处理的细节，别在临时脚本里漏掉：

- **分进程**：CPU 多线程会把同一进程里后面的 DML 计时拖慢（同进程测出来的
  A 卡损失前向能虚高到 279 ms），所以 CPU 和 DML 必须各跑一次进程。
- **迭代间排空队列**：DirectML 上「连续反复算同一个损失」会卡死命令队列
  （下一次前向的第一个算子永远不返回，堆栈停在 `make_anchors` 的 `meshgrid`）。
  实测**原生 `aten::index` 路径同样卡**，是后端/损失本身的既有问题，不是补丁引入的。
  脚本在每次迭代后做一次标量回读把队列排空，且不计入计时区间。

产出五个数：损失前向、损失反向、模型前向、模型反向、一个 iteration。

**但这个脚本用的是合成 batch（nc=2，每图 2 个框），损失耗时系统性偏乐观**：
实测合成 batch 的损失前向 71.0 ms，真实训练循环里是 96.3 ms。
所以**损失那一行必须用真实训练循环的数字**，用 `bench_step.py`：

```powershell
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_step.py dml_flat 3
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_step.py dml_noflat 3
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_step.py cpu 2
```

它挂在 ultralytics 的 `on_train_batch_start/end` 回调上量**真实训练步**，
并包一层 `v8DetectionLoss.loss` 量损失前向。模型前/反向的结论不受 batch 影响，
损失那一行受影响。

### 3.5. 算子级证据：补丁到底改了什么

耗时数字会骗人，算子直方图不会。跑两个模式各一遍，做差：

```powershell
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_opdiff.py dml_flat
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_opdiff.py dml_noflat
```

预期（ultralytics 8.4.213）：补丁把损失前向里的 **13 个 `aten::index` 改写成
12 个 `gather`**，模型前向一个算子都没碰（331 vs 331），
代价是每次损失前向多 ~200 个小算子和 16 次 `_local_scalar_dense`（D2H 标量回读 = 越界检查）。

这解释了补丁的收益**为什么不是常数**：收益 ∝「原生 `aten::index` 的元素数 ÷ 补丁固定开销」。
索引张量小的时候补丁反而更慢。

### 4. 端到端配对 A/B：同机同时段，只差一个开关

```powershell
& '<anaconda>\envs\maimai\python.exe' -u <skill>\scripts\bench_ab.py --epochs 25 --timeout 1800
```

规则（不遵守就会得出错误结论）：

- **配对**：同一台机器、同一时段，只差被测开关，其它完全一致。
- **至少两对**：同一配置的两次实测能差 6%（287.7 s vs 305.9 s 都出现过）。
  一对看不出 4~8% 这种量级。
- **超参对齐基线**：`mosaic=0 fliplr=0 scale=0.3 hsv_s=0.4 workers=0`，
  否则 ultralytics 默认的 mosaic=1.0 会把数据管线变成混淆项。
- **对指标**：改写类补丁的最终指标应当**逐位相同**（改写只换算法路径不改数值）。
  不相同说明数值语义变了，先停下修。
- **必须带超时**：DirectML 的命令队列在某些情况下会卡死，训练进程会无限挂起。
  `bench_ab.py --timeout` 到点会终止子进程并如实报「超时」。
- **别读旧产物**：中途被杀掉的那一臂，`results.csv` 还留在盘上。
  直接读会拿到上一轮的数当成新的 —— `bench_ab.py` 会比对修改时间，
  只认本次运行产生的 CSV，否则报「没有本次运行的产物」。
- **中途被污染就作废那一臂**：如果跑的过程中另一个会话的任务上了同一块卡，
  表现是**每轮耗时递增**（补丁的代价应该是恒定的），那一臂的数据不能要。
  只杀自己的进程，别动别人的任务。

### 5. 解读：三个必须避开的陷阱

详见 `references/baseline.md`。要点：

- **队列状态会放大差距，而且不同口径能得出相反结论**。
  同一件事有三档口径，引用时**必须带口径名**（详见 `references/baseline.md` §4）：

  | 口径 | 定义 | 原生 `aten::index` | 展平 `gather` |
  | --- | --- | --- | --- |
  | B1 排空 | 计时前排空命令队列 | 48.7 ~ 55.0 ms | 51.6 ~ 54.0 ms |
  | B2 半满 | 先跑模型前向不排空（真实训练状态） | 55.0 ms | 57.8 ms |
  | B3 积压 | 连续反复算同一个损失，中间无任何 CPU 工作 | 175 ms | 53 ms |

  **旧文档里的「快 3.3 倍」来自 B3，而 B3 是个会卡死队列、复现不出来的状态。**
  在能复现的 B1/B2 上补丁**并不更快**（甚至慢 5~10%）。
  真实训练队列在 B1 与 B2 之间，所以**可信的量级只有端到端的 4~11%**。
  报结论时必须同时给两栏**并写明口径**，否则等于没测。
- **端到端收益 ≪ 组件收益**。损失只占一个 iteration 的一部分。
  512px/batch8 实测：A 卡单步 323 ms 里模型前 63 ms + 模型反 146 ms + 损失 96 ms。
  补丁把损失前向压掉 48.7 ms（组件快 1.50×），折算到整轮只有 10.9%。
  写 README 时别把组件数字当端到端数字。
- **小张量上 CPU 更快**。DirectML 每算子固定开销 ~0.2~3 ms，
  ultralytics 的损失是一堆小算子堆出来的，CPU 多线程算这种张量反而占优。
  实测 A 卡上的损失前向（96.3 ms）仍比 CPU（55.2 ms）慢 1.74 倍。
  「损失留在 A 卡」的收益是「不折腾数据」，不是「更快」。
- **组件级五项不可相加**。分项之和可能超过实测 iteration，
  做「瓶颈占比」推断要用实测 iteration，别拿分项求和。

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
Output: bench_align 证明补丁开/关梯度 183/183 逐位相同 +
      真实训练循环的分步耗时 + 端到端配对 A/B（区间 4~11%）+
      B1/B2 两栏（带口径名）+ 算子直方图（13 个 index → 12 个 gather）+
      「端到端 4~11%，瓶颈在模型前后向；A 卡上损失占 30%，是次大瓶颈」+
      doctor 对齐结论 + 每个数字的证据等级。

## Windows (win32) platform notes

- 本仓库只在 Windows 上跑（torch-directml 只发 Windows 轮子），命令一律用 PowerShell。
- 解释器是 `C:\Users\<user>\anaconda3\envs\maimai\python.exe`（Python 3.11），
  torch/ultralytics 不在 venv 里，靠 `PYTHONPATH` 拼两个 site-packages 目录。
- **`PYTHONPATH` 的顺序不能颠倒**：`yolo_dml_env` 必须在前。
  两个目录里都有 torch，但只有 `yolo_dml_env` 里的 `torch_directml`
  是按 torch 2.4.1 编译的；顺序颠倒会让 torch 2.15 配上
  `torch_directml_native` → `DLL load failed ... 找不到指定的程序`。
- 仓库根目录要自己加进 `sys.path`（或 `Set-Location` 到仓库根）才能
  `import Direct_AMD_train`。
- 训练环境与数据集在 `_debug/` 下（被 gitignore），换机器要按
  `references/environment.md` 重搭。
- 写中文文件一律用 UTF-8；PowerShell 5.1 读文件默认 GBK，会显示乱码
  （用 read 工具或 `-Encoding UTF8`）。
