# OOAL：UMD 统一协议复现

本轮复现 OOAL 方法并与 PartGLEE＋Valen-4B 统一评测，**不宣称精确复现论文的 59.7%**。作者未公开原 UMD MMSegmentation 代码及完整配置。[作者说明](https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2408711599)

| 项目 | 本轮设置 |
|---|---|
| 网络 | 官方 OOAL `b65857c`；DINOv2 ViT-B/14＋CLIP ViT-B/16 文本编码器；8 个学习提示、末 3 层特征融合、2 层解码器 |
| 初始化 | DINOv2／CLIP 使用基础预训练权重并冻结；功能预测模块重新初始化，未加载 AGD20K 任务 checkpoint |
| 环境 | 项目外 Conda `ooal-umd`；Python 3.11、PyTorch 2.7.1＋CUDA 12.6；GPU 0，RTX 4090 |
| 训练数据 | 作者 `train_novel_one_pick` 的 76 张，每实例 1 张；32 张人工标注、44 张自动标注。RGB 和标签均与原始 UMD 逐像素核对一致 |
| 数据隔离 | 官方 novel/train 的 8 类与 novel/test 的 9 类没有交集；训练图像、实例与测试集均无重叠 |
| 训练配置 | 固定 seed=321、20,000 步、batch=1；SGD lr=0.01、momentum=0.9、weight decay=0.0005、Nesterov、余弦衰减 |
| 预处理 | 沿用公开 OOAL 尺寸：RGB 缩放 256×256、同步随机裁剪 224×224、水平翻转；类别图用最近邻插值；ImageNet 归一化 |
| UMD 适配 | 暴露原网络的 sigmoid 前 logits；8 类交叉熵，包含背景，忽略 void=-1；测试 logits 上采样至原始 640×480 后 argmax |
| 模型选择 | 只使用第 20,000 步权重；训练期间不评测 test，不按测试结果选阈值或 checkpoint |
| 测试协议 | 同一份 2,661 张人工帧清单、原始 dominant 标签、原生类别编号；保留 ladle 的 contain，不做论文额外重映射 |
| 指标 | 直接调用现有 `affordance_jev.evaluation.evaluate`；背景误预测计入，7 类前景 mIoU 为主指标 |

训练使用的 76 张中没有 ladle，作者这部分 PNG 与原始 dominant 标签完全一致。[划分与标签核查](../../data/UMD/manifests/ooal_novel_train_76.meta.json)

运行入口：[训练](../../scripts/train_ooal_umd.sh)、[推理与评测](../../scripts/infer_ooal_umd.sh)。全部路径和参数见 [配置](../../configs/ooal_umd.json)；源码在 `src/affordance_jev/baselines/ooal/`。OOAL 仓库仅增加本地权重加载与 logits 返回接口，原 AGD20K 默认行为保留。

记录目录：[训练日志与源代码快照](../../outputs/ooal_umd_novel_20261016/train/)、[权重](../../checkpoints/ooal_umd_novel_20261016/)、[统一实验汇总](../research/实验结果.md)。训练状态和损失分别保存于 `train/status.json`、`train/loss.jsonl`。

计时口径：OOAL 为预热后的单张 RGB→全部功能类别图，包含读取、预处理与输出上采样，排除模型加载和文件保存；此前 PartGLEE＋Valen 是分阶段离线计时，**不据此宣称严格端到端加速比**。

## 与论文分数的差距复核

2026-10-16。[论文表 2](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_One-Shot_Open_Affordance_Learning_with_Foundation_Models_CVPR_2024_paper.pdf)的 Unseen mIoU 为 **59.70%**，本地原始标签口径为 **39.47%**，相差 **20.23 个百分点**。以下只重算已保存预测的混淆矩阵，没有重训、修改预测或覆盖原始指标。

| 计分条件（两模型相同） | OOAL mIoU | PartGLEE＋Valen mIoU |
|---|---:|---:|
| 原始标签；背景误检计入 | 39.47% | 41.64% |
| 仅将 ladle 的 GT：contain → scoop | 45.11% | 44.12% |
| 上项＋忽略所有 GT 背景像素（诊断假设） | 53.28% | 52.06% |

- **已确认的标签差异**：作者对测试集 ladle 做了 contain → scoop 重映射；我们原来没有做。这一项使 OOAL 增加 **5.64 个百分点**，并使两模型的排序反转。[作者确认](https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2614175357)
- **尚未确认的背景处理**：“不把背景类别纳入平均”与“忽略所有 GT 背景像素”不同。后者不惩罚预测落在背景上的误检。作者只说明排除背景，公开证据不足以确定原代码采用哪种方式；**53.28% 只是敏感性分析，不能当作论文协议复现成绩**。[作者说明](https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2668062477)
- **尚未对齐的训练／预处理细节**：本地采用公开 AGD20K 的 256→224 预处理、batch=1、8 类 CE；论文对 UMD 只说明 MMSegmentation 默认训练设置，原 UMD 配置无法获取。分辨率、增强及背景学习方式可能影响结果，当前无法量化各自影响。
- **尚未确认的测试帧筛选**：本地仅测 2,661 张人工帧；作者是否测全部 7,969 帧仍无明确回复，不能擅自认定相同。[待答问题](https://github.com/Reagan1311/OOAL/issues/15)

76 张训练样本已逐像素对齐作者发布的数据，20,000 步已完整训练。已确认的标签差异只能解释部分差距；**目前不足以把本地差距归因为 OOAL 本身能力，也不能用原表宣称超过论文 OOAL**。

[计分复核明细与输入哈希](../../outputs/ooal_umd_novel_20261016/comparison/protocol_audit_20261016.json)。

## 标签修改可视化

[打开完整界面](../../outputs/umd_gt_label_viewer/index.html)：RGB、原始 GT、按 OOAL 规则修改的 GT、变化区域同步查看；覆盖全部 2,661 张测试图及 76 张训练示例，支持类别／实例筛选、透明度、放大、像素检查和下载。

修改只替换类别 ID：`物体为 ladle 且 gt_label == 4` 的像素设为 `3`。几何边界、手柄、背景以及其他物体不变。500 张 ladle 测试图中 **412 张发生修改、88 张无 contain 像素**，累计修改 **937,808 像素**；76 张训练示例均不变。新标签是按作者规则生成的可视化副本，未改写原始数据或评测结果。
