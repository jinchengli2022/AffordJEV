# AffordanceJEV

**PartGLEE 部件分割＋Valen-4B 功能判断 → 像素对齐的功能 mask。** 支持原始 Valen、普通 SFT/LoRA，以及 A/B/C 实验版本。

源码、模型配置、tokenizer、功能定义和训练入口均在本仓库内。**Conda 环境留在仓库外；Git 不包含大权重、数据集和实验输出。** 克隆后补齐权重及要处理的 RGB／数据集即可运行。

[整体框架](docs/research/整体框架.md) · [环境安装](docs/环境安装.md) · [权重清单](models/README.md) · [训练复现](docs/训练复现.md) · [实验结果](docs/research/实验结果.md) · [迁移验证](docs/迁移验证.md)

## 目录

```text
AffordanceJEV/
├── scripts/                 # 推理、训练、数据准备和检查入口
├── src/affordance_jev/       # 分割、判断、合并、训练、评测与可视化
├── third_party/PartGLEE/     # 固定版本的 PartGLEE 源码及原始网络配置
├── configs/                 # 功能定义、候选参数、76 张选择及训练参数
├── requirements/            # 两个外部 Conda 环境的包版本与约束
├── models/
│   ├── partglee/            # PartGLEE_SwinL.pth
│   ├── clip_vit_base_patch32/ # PartGLEE 所需 CLIP，不能遗漏
│   ├── valen/               # 官方模型代码、配置、tokenizer、三片权重
│   └── manifest.json        # 文件大小、SHA256、来源和微调权重清单
├── checkpoints/             # 本地 SFT/LoRA 与 A/B/C 权重
├── data/UMD/                # 官方 UMD 数据；清单在新机器重新生成
├── outputs/                 # 中间量、预测、指标和浏览界面
└── docs/                    # 方法与实验说明
```

## 克隆后运行

需要 Linux、NVIDIA CUDA GPU、Conda；已验证 RTX 4090。PartGLEE 的 CUDA 扩展需在目标机器编译，不能直接复用别的机器的 `.so`。

```bash
git clone git@github.com:jinchengli2022/AffordJEV.git
cd AffordJEV

# 已有环境可以跳过安装，直接做 check。
bash scripts/setup.sh all
bash scripts/setup.sh check

# 按 models/README.md 放入五个基础权重文件，然后校验。
conda activate partglee
PYTHONPATH=src python -m affordance_jev.deployment.check_assets --sha256

# 单张 RGB，也可以输入图片文件夹；输出目录必须为空。
bash scripts/infer.sh --input /path/to/image.jpg --output outputs/demo
```

默认环境为 `partglee` 与 `valen-4b-sft`，默认 Conda 安装位置 `$HOME/miniconda3`。路径、配置、输出、权重和 `CUDA_VISIBLE_DEVICES` 都在脚本顶部，可直接修改。脚本由自身位置定位仓库，不依赖原机器的 `/home/ljc/code/Affordance`。

若不提供具体物体／部件词表，使用通用配置：

```bash
bash scripts/infer.sh --input /path/to/image.jpg --output outputs/demo_generic \
  --config configs/umd_pipeline_generic.json
```

该配置只向 PartGLEE 提供 `object` 和 `part`，不指定物体种类或部件名称；保留原有两次分割、50 个原始物体候选／50 个部件候选及最多 6 个互斥候选的设置。仍使用文本编码器，不能称为完全无文本输入。Valen 的功能类别与定义照常提供。

历史结果使用 `configs/umd_pipeline.json` 中的 17 个物体名称和 18 个部件名称；通用配置会重新生成候选，旧表指标不能直接沿用。UMD 批量入口同样接受 `--config configs/umd_pipeline_generic.json`，同时需通过 `--output` 指定新目录。

通用词表的给定操作全量实验已完成，结果与前景误选诊断见[实验结果](docs/research/实验结果.md)。通用配置仍为可选项；部分样例会将底板选作前景。[早期分步检查](outputs/partglee_generic_smoke_20261018/index.html)。

原始模型使用 `configs/umd_affordances.json`。加载 UMD 微调版本时，额外传入完整 checkpoint，并使用训练时的 dominant 功能定义，例如：

```bash
bash scripts/infer.sh --input /path/to/image.jpg --output outputs/demo_b0 \
  --adapter checkpoints/valen_76_abc_20261017/b0/final \
  --definitions configs/umd_affordances_sft.json
```

请在仓库根目录执行上述含相对 CLI 参数的示例。B0 是本地微调权重，需另行复制其 `final/` 目录；仅下载官方 Valen 权重不会得到 B0 的结果。

## UMD 评测

将官方 tools 数据解压为 `data/UMD/part-affordance-dataset/`，其中应有 `tools/`、`novel_split.txt` 等原始文件。

```bash
bash scripts/prepare_umd.sh   # 生成 novel/test 人工帧清单
bash scripts/infer_umd.sh     # RGB → 两次 PartGLEE → Valen → 功能 mask
bash scripts/eval_umd.sh      # 原始 dominant GT 评测和对照图
```

默认评测 **2,661 张人工帧、5,232 个图像＋功能对**。使用七类原始 ID，忽略 void（−1），背景误报计入；关闭整物体限制。GT 只参与清单筛选、训练监督和评测，不进入正常推理输入。

已有固定候选时，`scripts/infer_valen_umd.sh` 可只替换 Valen checkpoint 重跑分类及评测，默认演示 B0。首次运行应先通过 `infer_umd.sh` 生成候选缓存。

**Valen 选择前景的新实验：** 修改脚本顶部路径与 GPU 后执行 `bash scripts/run_umd_valen_foreground.sh`。先检查 9 张图，再全量测试；原始 Valen 根据物体名＋操作逐候选判断，选出一个完整目标后重新分割部件，最终采用质心点 yes/no 判断。全部拒绝时输出空结果，不回退几何排序。此入口的固定功能对与对照来自此前本地实验，迁移时需提供对应清单和对照目录。[参数、输出与复现说明](src/affordance_jev/experiments/valen_foreground/README.md)

## 训练与可视化

| 入口 | 用途 |
|---|---|
| `scripts/prepare_valen_umd.sh` | 校验固定 76 张训练图，生成候选、监督 records 和 A/B 变体 |
| `scripts/train_valen_umd.sh` | 决策头预热＋普通 SFT/LoRA；支持仅预热 |
| `scripts/train_valen_abc.sh` | 使用同一预热权重，单独训练 S0R／A／B／C |
| `scripts/view_gt_majority.sh` | 固定候选并按 GT 多数类别重新赋类，生成全量离线界面 |

训练数据、模型、配置和输出路径均在入口顶部，完整步骤见 [训练复现](docs/训练复现.md)。固定 76 张选择已收入配置，新机器不需要外部 OOAL 仓库或历史 `outputs` 来重建训练输入。普通 SFT 和 A/B/C 均为本项目实验，非 Valen 官方训练复现。

每次推理保存：目标物体 mask、原始部件候选、整理后候选、Valen 三张实际输入与请求、类别概率、最终 mask 和 `index.html`。浏览器打开输出目录的 `index.html` 查看。

GT 多数类别界面默认输出 `outputs/umd_gt_majority_viewer/index.html`，可查看全部 2,661 个样本。**81.95% mIoU / 75.49% gIoU 是使用 GT 的诊断值，不是模型推理成绩。**

## 版本与迁移约定

- 模型与上游来源见 [third_party/sources.json](third_party/sources.json)，本地适配说明见 [third_party/README.md](third_party/README.md)。
- 计算参数、功能定义、训练样本和评测协议保持原实验设置；环境版本固定。跨 GPU／软件版本的逐位一致性不作保证。
- 默认不覆盖任何已有训练或推理目录；重复实验修改输出路径。
- 不复制 Conda 环境。`.gitignore` 排除权重、数据、编译产物、日志和历史本机定时器配置；模型代码、配置、tokenizer 和渲染字体保留在 Git。
- `baselines/ooal` 和 `baselines/affordance_r1` 用于对比实验，需另行部署。Affordance-R1 入口目前引用本机 `2d_affordance` 中的模型权重及 SAM2 源码；这两个 baseline 不属于 PartGLEE＋Valen 自包含运行路径。
- 仓库：[jinchengli2022/AffordJEV](https://github.com/jinchengli2022/AffordJEV)，上传范围以本目录为根。
