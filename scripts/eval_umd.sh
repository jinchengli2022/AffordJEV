#!/usr/bin/env bash
set -euo pipefail

# 手动填写：使用本次推理保存的清单，保证限量检查时不会混入未推理样本。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="$HOME/miniconda3"  # Conda 安装位置；环境留在仓库外
PYTHON="$CONDA_ROOT/envs/partglee/bin/python"
PREDICTIONS_DIR="$PROJECT_DIR/outputs/umd_novel_test"
EVAL_MANIFEST="$PROJECT_DIR/outputs/umd_novel_test/evaluation_manifest.jsonl"
OUTPUT_DIR="$PROJECT_DIR/outputs/umd_novel_test/evaluation"
VIS_LIMIT=50       # 保存前多少张 RGB / GT / 预测对照图；不影响指标统计范围

export PYTHONPATH="$PROJECT_DIR/src"
"$PYTHON" -m affordance_jev.evaluation.evaluate \
  --manifest "$EVAL_MANIFEST" \
  --predictions "$PREDICTIONS_DIR" \
  --output "$OUTPUT_DIR" \
  --vis-limit "$VIS_LIMIT" "$@"
