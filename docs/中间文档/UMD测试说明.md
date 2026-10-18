# UMD 基础测试

更新：2026-10-16。范围为 **原始 tools 单物体 RGB、人工标注帧、主要功能标签**。原始 PartGLEE＋Valen-4B，不微调。用户已确认：UMD 测试关闭“禁止整物体”限制。

## 操作类型

| 原始 ID | 功能 | 用于判别的含义 |
|---|---|---|
| 0 | background | 原始背景；模型未匹配或未覆盖的像素也输出 0 |
| 1 | grasp | 用于抓握、操控的专门握持部位 |
| 2 | cut | 刀刃等切割工作部位 |
| 3 | scoop | 用于舀取、收集材料的凹形工作面 |
| 4 | contain | 用于容纳内容物的内部腔体表面 |
| 5 | pound | 用于敲击的工作头或撞击面 |
| 6 | support | 从下方承托材料的较平坦工作面 |
| 7 | wrap-grasp | 可用手指和手掌环抱抓握的外部主体表面 |

ID 与名称沿用 [UMD 官方 README](https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/README.txt)。完整英文判别定义是本项目的通用提示词，见 [umd_affordances.json](../configs/umd_affordances.json)，不作为官方定义的逐字引文。

Valen 为每个候选做 **七种功能＋none 的八选一**。`none` 是拒选项，不是新增的 UMD 功能；它与未覆盖区域、场景背景分别保存，生成原生类别图时映射为 0。每个候选只属于一个类别，同类区域合并；不同类别的 mask 互不重叠。

原来的 `stick / open / shovel` 不再使用。尤其不把 `shovel` 一律替换成 `scoop`：UMD 分开定义了 `scoop` 和 `support`。

## 数据和划分

当前目录：

```text
data/UMD/
├── part-affordance-dataset/      # 单物体数据和官方划分
│   ├── README.txt
│   ├── category_split.txt
│   ├── category_10_fold.txt
│   ├── novel_split.txt
│   └── tools/<物体实例>/
│       ├── <实例>_<帧号>_rgb.jpg
│       ├── <实例>_<帧号>_label.mat
│       ├── <实例>_<帧号>_label_rank.mat
│       └── <实例>_<帧号>_depth.png
├── part-affordance-clutter/      # 多物体场景，当前未评测
├── manifests/                   # 当前测试清单及元数据
├── part-affordance-dataset-tools.tar.gz
└── part-affordance-dataset-clutter.tar.gz
```

- 直接读取 RGB 原像素网格，不改写原始数据。推理中的裁剪在输出时还原到原图坐标。
- `*_label.mat/gt_label` 是主要功能类别图。本版不使用 `*_label_rank.mat` 的多功能排序。
- 根据 .mat 文件中的 **gt_type 字符串**筛选 `manual`，排除 `automatic`；字段缺失会报错。已检查的 tools 帧和 clutter 帧起始编号不同，因此不能用统一的“帧号除以 3”代替此字段。
- `novel_split.txt` 和 `category_split.txt` 中 1 为 train、2 为 test；十折文件中第指定列的 1 为 test、0 为 train。划分按实体工具，不重新随机打散图像。
- 默认选择 `novel/test`。这个名字表示原数据集的未见类别划分，不保证这些类别从未出现在基础模型预训练中。
- 清单包含 RGB/标注路径、尺寸、文件哈希、实例和人工标注来源；配套 `.meta.json` 记录划分文件哈希、排除数量和是否限量。
- 正常建清单时，所选划分缺少物体目录或 RGB 对应标签会报错。`--allow-incomplete` 仅供开发，使用情况写入元数据。

本地 tools 数据已完成下载、逐文件校验及完整 novel/test 评测，见 [UMD 实验结果](UMD实验结果.md)。已解压的 clutter 目录含多个物体，当前推理的“单个中央目标物体”假设不适用；代码会拒绝将 clutter 清单送入这套单物体基线。

## 入口与配置

在项目根目录依次运行：

```bash
bash scripts/prepare_umd.sh
bash scripts/infer_umd.sh
bash scripts/eval_umd.sh
```

直接编辑脚本开头的路径和参数。推理使用 `CUDA_VISIBLE_DEVICES=0`；环境为目录外已有的 `partglee`、`valen-4b` Conda 环境。官方模型源码和权重目前复用 `/home/ljc/code/Affordance/PartGLEE`、`/home/ljc/code/Affordance/Valen-4B`。

`infer_umd.sh/LIMIT=0` 表示完整清单，正整数为开发子集。每次输出选新的空目录。评测读取本次推理保存的 `evaluation_manifest.jsonl`，防止限量推理却按完整数据集统计。

`umd_pipeline.json` 的部件候选默认最多 6 个，按既有规则分配重叠像素，`whole_object_fraction=null` 同时关闭候选阶段和合并阶段的整物体过滤。PartGLEE 使用统一的 17 类物体词表和通用部件词表，所有图完全相同，不向模型提供当前图的真实类别。

两个模型分别在批处理进程内加载一次。当前为可审查的离线基线，阶段间使用中间文件；尚未实现两个常驻模型的实时内存传输或预热速度基准。

## 输出和指标

推理输出根目录：

| 路径 | 内容 |
|---|---|
| `index.html` | 全部样本的预测预览入口 |
| `evaluation_manifest.jsonl` | 本次实际处理的样本，供独立评测使用 |
| `model_jobs.jsonl` | 传给模型进程的任务；无标注、深度路径 |
| `config_snapshot.json`、`affordances_snapshot.json` | 本次参数和功能定义 |
| `source_sha256.json` | 源码指纹 |
| `partglee_load.json`、`valen_load.json` | 批次内各模型的单次加载记录 |
| `run.json`、`logs/` | 样本状态、失败计数、批次耗时和模型日志 |
| `samples/<实例>/<帧号>/index.html` | 单图从定位到功能预测的全部中间可视化 |

单图目录保留 `raw/` 原始候选、`foreground/` 预测物体及裁剪、`regions/` 互斥候选、`valen_request.json`、`valen_response.json`、`valen_compiled/` 实际 prompt/输入张量、`decisions.json`、`timings.json`。最终结果在 `result/class_map.png` 和 `result/masks/<功能>.png`，均与原始 RGB 对齐。

评测输出：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 总体、各类别、各实体指标，失败数量，混淆矩阵和完整统计口径 |
| `per_class.csv` | 0–7 类的像素计数、IoU、Dice |
| `per_image.jsonl`、`per_query.jsonl` | 逐 RGB、逐 RGB＋功能的成绩及失败原因 |
| `index.html`、`visualizations/` | RGB → 原始 GT → 预测对照 |

主要指标：

- **foreground mIoU**：先累计全图混淆矩阵，再对七种前景功能求类别平均 IoU；背景上的误预测、GT 中缺席类别的误预测会计入。全局 GT 与预测均为空的类不以 1 分充数。
- **gIoU**：对 GT 中存在的每个“图像＋功能”计算 IoU，再平均；同时报告平均 Dice 和 `P@50`（IoU ≥ 0.5）。
- **foreground pixel cIoU**：七类累计交集除以累计并集。它与仅对 GT 存在的查询累计并集的口径不同，不能混用。
- 原始 GT 的 `-1` 为 void，不计分；0 背景仍参与错误统计。不会用 GT 前景裁剪预测来提高分数。
- 已记录的推理失败按全背景预测计分，功能查询 IoU 为 0。未运行/缺失预测默认报错；只有显式 `--allow-missing` 才按失败计分，并报告数量。
- 当前实现的是上述 IoU/Dice 指标，**尚未复现原论文的排序标签指标或其他工作的 weighted F-measure**。使用原生标签和划分，不等于可以直接与任何 UMD 论文表格比较。
- 保存模块调用耗时、批次总耗时；含首次调用、可视化和落盘成本，不宣称实时 FPS。

## 基础检查

```bash
PYTHONPATH=src /home/ljc/miniconda3/envs/partglee/bin/python -m unittest discover -s tests -v
```

检查人工标注来源、官方划分、缺失数据、标签映射、整物体保留、区域互斥、void 忽略和失败计分。真实模型流程检查使用官方 novel **train** 实例的两张 RGB，仅验证代码与中间输出，不作为正式 UMD 测试成绩。

本次检查已完成：

- 11 项基础检查通过。
- `scissors_02/00000001`、`scissors_02/00000004` 两张人工标注帧完成真实推理和独立评测；两张图的模型像素输入指纹不同，均读取了各自的 RGB。
- 两张输出均为原始 640×480，各类 mask 无重叠，未使用 GT 前景定位。
- 批次总耗时 49.16 秒，含两个模型各一次加载及完整中间量保存；不是实时延迟基准。
- [逐模块预览](../outputs/umd_smoke_20261015/index.html) · [RGB／GT／预测对照](../outputs/umd_smoke_20261015/evaluation/index.html) · [开发检查指标与协议](../outputs/umd_smoke_20261015/evaluation/metrics.json)。
