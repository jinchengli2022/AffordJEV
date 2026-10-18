# Affordance-R1 UMD 基线

原始 Affordance-R1 → 框／点 → SAM2。使用官方 prompt，提供当前物体类别与操作名称；不加载本地 LoRA。

入口：`scripts/infer_affordance_r1_umd.sh`。在文件顶部填写 Conda Python、模型／SAM2 权重、SAM2 源码、数据清单、配置、GPU 和输出位置。本机复用 `affordance-r1-portable` 环境以及已有 `2d_affordance` 下的 SAM2 源码和权重。

```bash
bash scripts/infer_affordance_r1_umd.sh
```

- 每张图独立查询七类，保存逐操作 prompt、原始回复、框点、SAM2 分数与二值 mask。
- `raw_masks/` 为各操作的独立 mask；`result/class_map.png` 为按 SAM2 像素 logit 消解重叠后的七类单标签结果。
- `sam_logits.npz` 仅为 float16 检查副本；合并使用 float32，保存的 class map 与二值 mask 是实际评分输出。
- 解析／分割失败的操作输出空 mask，不删除测试样本。
- 原始全图评分自动保存到 `evaluation/`。批处理耗时是吞吐成本，不能当作单请求延迟。

两种口径的对照和全量可视化：

```bash
PYTHONPATH=src /home/ljc/miniconda3/envs/affordance-r1-portable/bin/python \
  -m affordance_jev.baselines.affordance_r1.compare \
  --root outputs/affordance_r1_umd_20261017/test \
  --output outputs/affordance_r1_umd_20261017/comparison --gallery
```

“给定操作”从相同推理中提取 5,232 个指定功能 mask；其 mIoU 仅累计这些查询中的交／并集，不惩罚未查询类别。OOAL／Valen 使用原有预测提取目标类别，按相同查询重评分。两种口径分表；known-category 输入条件在结果中保留。
