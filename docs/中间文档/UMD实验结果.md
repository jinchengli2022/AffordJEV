# UMD 实验结果

实验日期：2026-10-15。**PartGLEE＋Valen-4B 原始权重，未在 UMD 上微调。**

UMD tools 官方 novel/test：**29 个实例、2,661 张人工标注 RGB，全部完成，0 失败**。输入为 RGB＋通用功能定义；最多 6 个互斥候选，关闭整物体限制。

**总体指标**

| 前景 mIoU | 前景累计 cIoU | 平均 IoU（gIoU） | P@50 | 平均 Dice |
|---:|---:|---:|---:|---:|
| **41.64%** | 45.97% | **54.31%** | 62.69% | 63.94% |

mIoU 按 7 种功能类别平均；gIoU、P@50、Dice 按 5,232 个 GT 中存在的“图像＋功能”查询统计。

**各功能 IoU**

| grasp | cut | scoop | contain | pound | support | wrap-grasp |
|---:|---:|---:|---:|---:|---:|---:|
| 47.09% | **81.76%** | 36.47% | 45.27% | 39.88% | **6.93%** | 34.09% |

**耗时（单卡 RTX 4090）**

| 统计范围 | 平均耗时 |
|---|---:|
| PartGLEE 两次调用：物体定位＋部件分割 | 0.308 秒/图 |
| Valen 判断该图全部候选 | 0.213 秒/图 |
| 离线推理全流程，含中间结果保存、不含评测 | 4.58 秒/图 |

推理＋评测总计约 **3 小时 25 分钟**。以上为离线实测，实时端到端延迟尚未测量。

[全部预测与中间结果](../outputs/umd_novel_test/index.html) · [RGB／GT／预测对照（前 50 张）](../outputs/umd_novel_test/evaluation/index.html) · [完整指标](../outputs/umd_novel_test/evaluation/metrics.json) · [评测口径](UMD测试说明.md)
