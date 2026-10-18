#!/usr/bin/env bash
set -euo pipefail

# 手动填写：使用下载完成并解压后的官方 tools 数据；不自动解压正在下载的文件。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="$HOME/miniconda3"  # Conda 安装位置；环境留在仓库外
PYTHON="$CONDA_ROOT/envs/partglee/bin/python"
DATASET_DIR="$PROJECT_DIR/data/UMD/part-affordance-dataset"
MANIFEST="$PROJECT_DIR/data/UMD/manifests/umd_novel_test.jsonl"
SPLIT="novel"       # novel：新类别；category：新实例；fold：官方十折
PARTITION="test"    # train / test / all
FOLD=1             # 仅 SPLIT=fold 时生效，范围 1..10

export PYTHONPATH="$PROJECT_DIR/src"
"$PYTHON" -m affordance_jev.datasets.umd \
  --dataset-dir "$DATASET_DIR" \
  --output "$MANIFEST" \
  --subset tools \
  --split "$SPLIT" \
  --partition "$PARTITION" \
  --fold "$FOLD" "$@"
