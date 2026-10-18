#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 按本机 Conda 安装位置修改；此任务只使用 CPU，不加载模型。
PYTHON="$HOME/miniconda3/envs/partglee/bin/python"
MANIFEST="$ROOT/data/UMD/manifests/umd_novel_test.jsonl"
DATASET_ROOT="$ROOT/data/UMD/part-affordance-dataset"
PREDICTIONS="$ROOT/outputs/umd_novel_test"
ANALYSIS="$ROOT/outputs/umd_error_analysis_20261016"
OUTPUT="$ROOT/outputs/umd_gt_majority_viewer"
WORKERS=4

if [[ -f "$OUTPUT/index.html" && $# -eq 0 ]]; then
  printf '浏览器打开：%s/index.html\n已有全量结果；只刷新界面可追加 --web-only。\n' "$OUTPUT"
  exit 0
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m affordance_jev.evaluation.majority_viewer \
  --manifest "$MANIFEST" --dataset-root "$DATASET_ROOT" \
  --predictions "$PREDICTIONS" --analysis "$ANALYSIS" \
  --output "$OUTPUT" --workers "$WORKERS" "$@"
# 完成后直接用浏览器打开 OUTPUT/index.html；更新界面可追加 --web-only。
