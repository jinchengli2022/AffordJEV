# Valen 前景选择实验

## 五种 Valen 的完整对照

入口：`scripts/run_umd_valen_point_union_full.sh`；配置：`configs/umd_valen_point_union_full.json`。先检查固定九图，再自动运行 **2,661 张／5,232 个功能对**，结果保存在独立目录。

统一使用内部中心点／互斥区域并集前景；`foreground_selector.minimum_yes_probability` 当前为 **0.96**。原始 Valen 判断前景，重新生成两阶段 PartGLEE 候选；五组共享每个功能对的新候选，分别运行填色、描边、质心、定义调整及条件 SFT。后续功能判断仍为 `P(yes)>P(no)`。SFT 只加载脚本指定的已有 LoRA＋判断头，不重新训练。

`full/comparison/index.html` 包含全部样本及各阶段实际输入；`full/evaluation/metrics.json` 保存五组指标。完成后自动在 `docs/research/实验结果.md` 新增对应分表，保留旧实验。小时监控只读取状态并通知本对话，不恢复训练、不重启实验。

## 内部中心点＋互斥区域并集

入口：`scripts/run_umd_valen_point_union.sh`；配置：`configs/umd_valen_point_union.json`。只运行固定 **9 张样例／18 个功能对**，不自动启动全量。沿用外部 Conda 环境和原始权重。

**object-level 候选 → 互斥分配 → 连通块拆分／碎片合并 → 内部中心点 → Valen yes/no → 前景并集。** 重叠像素按质量分除以面积平方根归属；不足 512 像素的碎片并入相邻区域，覆盖像素不减少。没有候选覆盖的位置记为 0，不强行补齐。

中心点取最大内接圆圆心，同分取最接近算术质心的位置。前景 Valen 每次只读取完整 RGB 和单点 RGB，判断该点是否属于指定物体，接受该物体的任意可见部位；操作名称不用于限制前景到功能部位。

前景阈值直接修改 `configs/umd_valen_point_union.json` 的 `foreground_selector.minimum_yes_probability`，当前为 **0.96**：只有 **`P(yes)>0.96`** 的预处理区域取并集，等于阈值也不接收。原始模型选择与阈值接收分别保留；不接收不等于模型高置信判断为背景。后续部件分割、功能 prompt 和判断规则均不变，功能判断仍为 `P(yes)>P(no)`（二分类约等于 `P(yes)>0.5`）。

新增输出：`objects/samples/…/preprocessed/` 保存互斥标签、区域、合并记录和点图；`foreground/samples/…/queries/<操作>/region_*/` 保存实际输入、英文请求、中文翻译及分数。`smoke/comparison/index.html` 展示所有样例的四个阶段。原始 mask 色块总览允许重叠；预处理色块图逐像素互斥。两者均是人工检查图，不是前景 Valen 输入。

## 历史描边＋最高分单候选

入口：`scripts/run_umd_valen_foreground.sh`。数据、配置、权重、输出目录、外部 Conda Python 和 GPU 编号均在脚本顶部。默认先检查 9 张图，再跑 2,661 张图／5,232 个固定功能对；`--phase smoke` 只运行样例。相同配置与输出目录可接续已完成阶段，不重复已记录的失败；换参数需使用新目录。

**RGB → PartGLEE 物体候选 → Valen yes/no → 单个前景 → PartGLEE 部件 → Valen 质心 yes/no → 功能 mask。** 两阶段使用原始 Valen，无训练。PartGLEE 只接收 `object / part`；物体名和操作提供给 Valen。

物体候选按原质量／面积门槛过滤，仅去除逐像素完全相同的 mask，不做位置排序或近似 NMS。每 6 个候选组成一组描边输入。所有组判断后，选择 `P(yes)>0.5` 的最高分；同分按模型质量和原编号决定，全部 no 则输出空预测。

每个操作独立判断前景。同一 RGB 下，相同前景 mask 共享部件分割；不同操作不共用前景判断。部件阶段保持原裁剪、互斥整理、功能定义与质心计算方式；质心不强制移入 mask。

| 输出 | 用途 |
|---|---|
| `protocol.json`、`config.json`、`models.json` | 固定数据、参数和模型来源 |
| `status.json` | 当前阶段与进度 |
| `smoke/`、`full/` | 样例检查和正式实验，分别保存 |
| `*/objects/samples/` | 全部原始物体 mask、类别与层级、过滤记录、分组可视化 |
| `*/foreground/samples/…/queries/<操作>/` | 前景实际输入、prompt、全部分数、选择结果与计时 |
| `*/parts/samples/…/<mask SHA256>/` | 前景裁剪、原始部件与最终互斥候选 |
| `*/methods/centroid/samples/…/queries/<操作>/` | 功能判断三张实际输入、分数、最终 mask 和分步页面 |
| `*/evaluation/`、`*/comparison/index.html` | 指标、逐功能对诊断和新旧对照界面 |

模型 worker 不读取 GT。报告器单独读取原生 dominant 标签；背景误报计入、void 忽略，拒识与运行失败均保留为空预测，分别统计。前景功能像素覆盖率不是完整物体的 IoU。

时间记录为离线分阶段计算成本：同图物体分割、相同 mask 的部件分割按复用次数摊销；Valen 包含预处理与输入记录保存。另记整个阶段墙钟时间，不将它们称为实时单次延迟。

前景 yes/no 概率是模型分数；跨候选／跨组分数没有额外校准，不保证高分就定位正确。实验将如实记录错误前景与最终分割指标。
