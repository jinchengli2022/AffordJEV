# UMD benchmark 与 baseline 选择

更新：2026-10-16。当前只选 **UMD**；研究目标为 **部件分割＋快速判别模型，实现可泛化的功能区域分割**。下文区分论文描述、公开代码和本项目协议；没有核实的处理细节不作推断。

## 1. 建议选哪些 baseline

优先安排 **Affordance-R1、A4-Agent、M2F-AFF**；补充 **DINO 线性探针、OOAL**。零样本迁移、少样本训练、全监督训练分组报告，不能混称为 train-free。

| 工作与发表情况 | 比较价值 | 是否使用 UMD 训练标注 | 开源与复现条件 |
|---|---|---|---|
| **Affordance-R1，AAAI 2026** · [论文](https://ojs.aaai.org/index.php/AAAI/article/view/37937) | 大模型预测框点＋SAM2；核心精度／速度对照 | 不使用；模型在 ReasonAff 上训练，再迁移到 UMD | [代码与权重入口](https://github.com/hq-King/Affordance-R1)已公开。应选官方权重，单独标明我们此前的本地 SFT 版本 |
| **A4-Agent，ECCV 2026 已接收** · [论文](https://arxiv.org/html/2512.14442v1) | 无任务训练的推理、图像编辑、定位、分割流程；比较复杂推理是否值得其耗时 | 不使用 | [官方仓库](https://github.com/EnVision-Research/A4-Agent)标注接收信息并提供代码。论文主结果依赖 GPT-4o；仓库另支持本地 Qwen2.5-VL，须分开报告 |
| **M2F-AFF，ECCV 2024 ACVR Workshop** · [论文](https://arxiv.org/html/2409.01814v1) | Mask2Former 功能分割；传统监督模型的精度／速度参照 | 使用完整训练划分 | [代码](https://github.com/apicis/aff-seg)、[UMD 权重](https://apicis.github.io/aff-seg/)均公开，复现条件最完整之一 |
| **Probing & Bridging 的 DINO 线性探针，CVPR 2026** · [论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Probing_and_Bridging_Geometry-Interaction_Cues_for_Affordance_Reasoning_in_Vision_CVPR_2026_paper.pdf) | 冻结视觉骨干＋轻量分类头；检验简单判别方法的能力 | 使用像素标注训练分类头 | [代码](https://github.com/QZhang2111/Probing_Bridging_Affordance)含 UMD 数据读取、训练和评测；需要训练对应划分的 head |
| **OOAL，CVPR 2024** · [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_One-Shot_Open_Affordance_Learning_with_Foundation_Models_CVPR_2024_paper.pdf) | 基础模型＋少量标注，对比本方法少量微调后的数据效率 | 每个训练物体**实例**取 1 张；seen／unseen 分别 54／76 张 | [官方仓库](https://github.com/Reagan1311/OOAL)原始入口只有 AGD20K；作者在 [issue #3](https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2395459749)另提供 UMD one-shot 数据。原 UMD 代码未提供；本地适配及统一重测已完成，见下文 |

还应加入两项**本项目自建对照**：同一批 PartGLEE 候选分别交给 **CLIP／SigLIP 相似度**和 **Qwen 生成式判断**，与 Valen 比较。它们不是已有论文的 UMD 成绩，但能直接检验快速判别模块的贡献。

## 2. UMD 的标签究竟是什么

UMD 原始数据提供两种标注，并非只有一种“热力图”。[官方数据说明](https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/README.txt)

| 原始文件 | 含义 | 转成评测标签的方法 |
|---|---|---|
| `*_label.mat` → `gt_label[H,W]` | 每像素最可能的功能类别，即 dominant／主类别标签；0 为背景 | 对功能 c 取 `gt_label == c`，得到二值 mask；不同功能互斥 |
| `*_label_rank.mat` → `gt_label[H,W,7]` | 每像素对 7 种功能的排序；1 最高，允许并列；背景各通道均为 0 | 必须另外指定 rank 如何变成分数或 mask；不同功能可能重叠 |

**GT 的格式、模型的输出格式、指标的计算方式需要分别说明。** 模型输出热力图，不代表 GT 本来就是热力图；使用 IoU，也不能证明读取的是主类别标签。

## 3. 各论文实际怎样使用 UMD

“是否允许单个像素对应多个功能标签”按**评测 GT**判断：同一 RGB 的不同功能 GT 是否允许在同一像素重叠，不是指模型是否输出多通道概率。“不确定”表示公开证据不足。

| 工作 | UMD GT 的处理 | 是否允许单个像素对应多个功能标签 | 模型输出与指标 | 划分与关键限制 |
|---|---|---|---|---|
| **Affordance-R1** | **排序标签 → 连续分数 → 二值化**；不是直接读取 dominant | 是 | 框点→SAM2 二值 mask；gIoU、cIoU、P@50、P@50:95 | 论文称覆盖全部类别并抽取 1/10 人工帧，共 1,922 张；公开代码按文件帧号 `%30==1` 抽样，每类固定查询 2 种操作 |
| **A4-Agent** | 公开代码取 **`rank == 1`** 得到二值 GT；没有执行 R1 的分数阈值转换 | 是 | 推理／定位→SAM2 mask；gIoU、cIoU、P@50、P@50:95 | 论文称沿用 R1、1,922 张；公开预处理按排序后文件列表下标 `%30==2` 抽样，每类固定 2 种操作 |
| **M2F-AFF** | **主类别标签**；代码明确读取 `*_label.mat['gt_label']` 并保存类别图 | 否 | 类别分割；各功能 Jaccard／IoU 及其平均 | `category_split`，14,823 train／14,020 test；训练部分再划验证集。公开转换脚本未筛选 `gt_type=manual` |
| **DINO 线性探针** | **主类别标签**；读取 `*_label.mat['gt_label']`，训练时另生成 patch 标签 | 否 | 分类 logits 上采样、argmax；mIoU | 发布清单为 category 划分：11,800 train／3,023 val／14,020 test；不是本项目 novel/manual 测试集 |
| **OOAL** | **主类别标签＋ladle 重映射**；作者发布的训练 PNG 与原始 `*_label.mat` 逐像素匹配，仅将 ladle 的 `contain` 改为 `scoop`；作者确认测试集也重映射 | 否 | MMSegmentation 分割评测；seen／unseen mIoU 和调和平均 hIoU | 使用 category／novel 两种划分；已核实 one-shot 训练样本包含自动标注帧，测试是否仅用人工帧尚未确认 |
| **INTRA，ECCV 2024** · [论文附录 A.4](https://arxiv.org/html/2409.06210v1) | 先把输入标注中 **大于 0 的值置为 1**；未明确原始 MAT 来源 | 不确定 | 预测为连续热力图；GT 与预测缩放、归一化后算 KLD／SIM；NSS 另对 GT 使用 0.1 阈值 | AGD20K Seen 训练后跨数据集测试；去掉训练中没有的操作。论文称修改后 UMD 有 39,846 个 images，未提供可核对清单，不能视为同等数量独立 RGB |
| **OVAL-Prompt，ICRA 2024 VLMNM Workshop** · [论文](https://arxiv.org/html/2404.11000v2) | 描述为像素级功能 GT；**具体 MAT、rank 转换及阈值未交代清楚** | 不确定 | LLM 将功能映射到部位，再用 VLPart 分割；weighted F-measure，β=1 | 零样本；测试清单／帧筛选细节未核实，不能直接归入 dominant 或 R1 协议 |
| **MaskPrompt，AAAI 2025** · [论文](https://ojs.aaai.org/index.php/AAAI/article/download/32200/34355) | UMD 上做闭集功能分割；**GT 的 MAT 来源和转换未公开说明到可复现程度** | 不确定 | mask proposals＋功能分类；UMD 表 1 报告 F1-Score | 确实有 UMD 定量结果，但具体划分、UMD 训练细节及 F1 汇总方式不充分；文中 0.7 是检测阈值，不能当作 GT 二值化阈值 |

以上前五项的标签／配置依据：[R1 `zero_shot.py`](https://github.com/hq-King/Affordance-R1/blob/9471404158d3baa0e6722c10c7a8cdcca5022efa/zero_shot.py)、[A4 预处理](https://github.com/EnVision-Research/A4-Agent/blob/cb83d39f8dbe30896b2e42eb27debf724bf8a7f1/dataset/preprocess_umd_new.py)、[M2F-AFF 转换代码](https://github.com/apicis/aff-seg/blob/2abf8988237026580b446e68cced98e9d039d630/src/utils/split_UMD.py)、[线性探针数据读取](https://github.com/QZhang2111/Probing_Bridging_Affordance/blob/69a442165e719ae5e05b7e6f81ab70e42707e470/geometry_probing/umd_linear_probing/src/data/dataset.py)与[配置](https://github.com/QZhang2111/Probing_Bridging_Affordance/blob/69a442165e719ae5e05b7e6f81ab70e42707e470/geometry_probing/umd_linear_probing/configs/dinov2.yaml)、[OOAL 作者对类别图的说明](https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2397055763)与[测试集重映射确认](https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2614175357)。

### OOAL 源码补查

官方仓库已克隆到 [`/home/ljc/code/Affordance/OOAL`](../../OOAL)，版本 `b65857c`。官方原始分支及提交历史未提供 UMD 数据加载器；本地新增适配位于 AffordanceJEV。

- **公开代码的 AGD20K 训练**：读取 `one-shot-seen／one-shot-unseen/<图名>.npy`，RGB 和标签同步缩放、裁剪、翻转，随后用 `BCE(pred, label / 255)`；标签是按功能分通道的软标签。[加载代码](../../OOAL/data/agd20k_ego.py)、[损失代码](../../OOAL/models/ooal.py)
- **公开代码的 AGD20K 测试**：读取 `Seen／Unseen/testset/GT/<操作>/<物体>/<图名>.png` 灰度图，缓存为 `.t7`，除以 255、缩放后计算 KLD／SIM／NSS；NSS 内的 0.1 阈值不是 UMD 的标签转换规则。[读取与缓存](../../OOAL/utils/util.py)、[测试入口](../../OOAL/test.py)
- **UMD 标签已做实物核查**：下载作者提供的 one-shot 包，54＋76 条 PNG 均为 `480×640`、单通道、类别值 0–7；其中 127 条与原始 `*_label.mat` 完全一致，另外 3 条在 ladle 的 `4→3` 重映射后完全一致。这确认了训练数据使用主类别图；不是 R1 的 rank 分数二值化。跨两划分的 130 条记录中，50 条源自人工标注、80 条源自自动标注，重复帧分别计数。[逐样本核查记录](../outputs/ooal_label_audit_20261016/umd_one_shot_comparison.json)
- **UMD 代码仍不完整**：作者用 MMSegmentation 训练和测试，但原代码所在服务器已无法访问；测试集的逐帧清单及背景处理实现仍未核实。不能将公开 AGD20K 加载器当作 UMD 实现。[作者说明](https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2408711599)

**本地统一重测已完成（2026-10-16）**：补充 UMD 单标签训练／推理入口，使用作者 76 张 novel one-shot 样本训练 20,000 步；在现有 2,661 张人工测试图上，OOAL 本地适配的 mIoU 为 **39.47%**，PartGLEE＋Valen 为 **41.64%**。这是统一协议下的本地对比，不替代原论文的 unseen 59.7%。[完整结果](实验结果.md)、[适配配置](others/OOAL复现说明.md)

### 两处最容易混淆的地方

**Affordance-R1 的阈值作用在 GT 转换后的分数上。** 对当前图排序张量的最大值 R，某操作在某像素的 rank 为 r：

```text
s = 0                 当 r = 0
s = 1 - r / R         当 r > 0
GT = (uint8(255*s) / 255 > 0.5)
```

本地核查的 UMD 帧中 R=9，因此保留 rank 1–4；A4 的公开代码只保留 rank 1。由于排序允许并列，**rank=1 也不能直接等同于 dominant 类别图**。两种代码的抽帧规则也不同，所以“沿用同一 benchmark”不代表标签与样本完全一致。

**论文描述与发布代码要分别记录。** 在本地完整 tools 数据上，仅复刻 R1 的文件名筛选规则得到 **1,013 张 RGB／2,026 个查询**，与论文写的 1,922 张尚未对应上；这是数据清单核查，不是新模型评测。最终论文用的清单仍需作者确认。R1 和 A4 当前固定操作映射均未实际查询 `wrap-grasp`，也不能代替我们完整 7 类测试。

## 4. 相关但暂不列为首批复现的工作

| 工作 | 与本课题的关系 | 暂缓原因 |
|---|---|---|
| **INTRA** | 弱监督训练后跨数据集泛化，可提供热力图方法对照 | [项目页](https://jeeit17.github.io/INTRA/)与[项目仓库](https://github.com/jeeit17/INTRA)可访问，但本次未核实可运行的训练／推理代码与权重；仓库主要为网站文件 |
| **OVAL-Prompt** | **LLM 选部位＋部件分割**，是需要讨论的接近工作 | [项目页](https://ekjt.github.io/OVAL-Prompt/)未提供已核实的完整复现代码／权重；且为 workshop 工作 |
| **MaskPrompt** | SAM 形状候选＋区域特征／文本分类，与候选判别路线相关 | 本次未找到可核实的官方训练／评测代码和权重；先列相关工作，不承诺复现论文成绩 |
| **AffordAnything／AffordAnything+，2026 预印本** · [论文](https://arxiv.org/html/2608.08929v1) | 分割候选＋语义相似度／融合，是很接近的近期方向 | 未核实正式发表和代码发布。其 **UMD-NAT** 按功能类型重划，测试仅 contain，共 760 个图像＋功能对；GT 为二值图，IoU@0.4 的 0.4 阈值作用于**预测热力图**，不是 R1 的 GT 转换；原始 MAT 转换未核实 |

筛选时排除了“只提到 UMD、没有实际 UMD 实验”的论文。例如 [WSAG-PLSP（ICLR 2025）](https://arxiv.org/html/2505.24103v1)与部件伪标签思路相关，但本次查到的正文、补充材料和官方发布围绕 AGD20K，不能当成已有 UMD benchmark 结果。

另须区分 **Probing & Bridging 的两个实验**：UMD 定量结果来自有监督的线性探针；DINO＋FLUX 的零样本融合在 UMD 仅做定性展示，不能把融合方法写成已有 UMD 零样本定量 baseline。[论文 §3.1、§4.2](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Probing_and_Bridging_Geometry-Interaction_Cues_for_Affordance_Reasoning_in_Vision_CVPR_2026_paper.pdf)

## 5. 本项目建议采用的统一比较协议

**主实验沿用已完成的 UMD tools／novel-test／manual／dominant 设置**，不在本次调研中更改数据或重跑实验。

| 项目 | 固定口径 |
|---|---|
| 测试数据 | [现有清单](../data/UMD/manifests/umd_novel_test.jsonl)：2,661 张 RGB、29 个实例、9 个物体类别；按 `gt_type=manual` 筛选 |
| 功能集合 | `grasp / cut / scoop / contain / pound / support / wrap-grasp`；关闭整物体限制 |
| 模型输入 | RGB＋统一功能名称／定义；不提供 GT mask、GT 框或 GT 裁剪。原方法若输入真实物体类别，须单独标明，不能隐含加入 |
| GT 与输出 | 使用 dominant 原生类别图；所有模型输出还原到原始像素网格。多操作独立预测可能重叠，转换成互斥类别图的规则须事先固定，不用 GT 消歧 |
| 主指标 | 七类前景 mIoU＋各类 IoU；背景误预测照常计入，void 忽略 |
| 补充指标 | 沿用现有 5,232 个 GT 存在的图像＋功能对，报告 gIoU、P@50、Dice；该查询口径不等同于 R1 的固定两操作查询 |
| 训练分组 | ① 不使用 UMD 训练标注；② 少样本；③ 全监督。训练／验证只能来自 official novel/train，不能把 test 用于选阈值、prompt 或 checkpoint |
| 耗时 | 同一 GPU、精度与输入尺寸口径；预热后 batch=1，报告单次 RGB＋操作延迟和整张图全部 7 类的总延迟；候选分割与判别模块分别计时。API 方案单独报告 |

复现前必须处理的三项差异：

1. **划分不同的权重不能直接拿来比较泛化。** M2F-AFF 的 category 训练集可能包含当前 novel/test 的物体，需按统一训练清单重训；DINO 探针同理。OOAL 已通过本地 UMD 入口重新训练，未复用 AGD20K 任务权重；须保留本地适配的标注。
2. **少样本数量要对齐。** OOAL 的 novel 设置是 76 个训练实例各 1 张；若本方法少量微调，应使用同一批样本，并固定种子。线性探针优先选 RGB 特征版；若加入估计深度／法线，额外模型及耗时也要计入。
3. **原论文数字与统一重测分表。** 如要复现 R1／A4 原协议，另存样本清单、rank 转换、查询集合和空 GT 规则。其论文值不能直接与本项目当前 **mIoU 41.64%／gIoU 54.31%** 作高低结论。[本地完整结果与口径](others/UMD实验结果.md)

这一设置能检验 **UMD 未见类别上的迁移与速度**；仅凭 UMD 固定 7 类，不能证明任意新功能的开放词表能力，也不能保证测试物体从未出现在基础模型预训练数据中。
