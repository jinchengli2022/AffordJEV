# UMD Part Affordance

| 路径 | 内容 |
|---|---|
| `part-affordance-dataset/` | 单物体 tools 数据、官方划分及原始说明 |
| `part-affordance-clutter/` | 多物体 clutter 场景，当前未评测 |
| `manifests/umd_novel_test.jsonl` | 当前测试清单：29 个实例、2,661 张人工标注 RGB |
| `manifests/umd_novel_test.meta.json` | 划分、筛选数量、标签类别及文件哈希 |
| `part-affordance-dataset-tools.tar.gz` | 单物体数据原始压缩包 |
| `part-affordance-dataset-clutter.tar.gz` | 多物体数据原始压缩包 |

当前实验使用官方 `novel/test` 划分，以及 `*_label.mat` 中的主类别标注。原图、深度和标注保持原样。

数据文件不随 Git 上传。新机器将官方 tools 压缩包解压到上表目录后，在仓库根目录运行 `bash scripts/prepare_umd.sh` 重建路径清单；训练清单由 `bash scripts/prepare_valen_umd.sh` 从固定 76 张选择重建。

清单记录本机真实路径，不直接复制旧机器的 JSONL。数据和 GT 原文件保持不变，重建时按 SHA256 核验。

[运行步骤](../../README.md) · [训练复现](../../docs/训练复现.md) · [实验结果](../../docs/实验结果汇总.md)
