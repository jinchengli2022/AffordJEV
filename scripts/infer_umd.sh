#!/usr/bin/env bash
set -euo pipefail

# 手动填写：MANIFEST 由 prepare_umd.sh 生成；OUTPUT_DIR 使用新的空目录。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="$HOME/miniconda3"  # Conda 安装位置；环境留在仓库外
MANIFEST="$PROJECT_DIR/data/UMD/manifests/umd_novel_test.jsonl"
PIPELINE_CONFIG="$PROJECT_DIR/configs/umd_pipeline.json"
AFFORDANCE_CONFIG="$PROJECT_DIR/configs/umd_affordances.json"
OUTPUT_DIR="$PROJECT_DIR/outputs/umd_novel_test"

# 源码与模型文件均在本仓库内；Conda 环境独立存放。
PARTGLEE_ROOT="$PROJECT_DIR/third_party/PartGLEE"
PARTGLEE_WEIGHTS="$PROJECT_DIR/models/partglee/PartGLEE_SwinL.pth"
VALEN_MODEL="$PROJECT_DIR/models/valen"
PARTGLEE_PYTHON="$CONDA_ROOT/envs/partglee/bin/python"
VALEN_PYTHON="$CONDA_ROOT/envs/valen-4b-sft/bin/python"
ADAPTER_DIR=""    # 留空=原始 Valen；微调时填完整 final 目录，并使用 umd_affordances_sft.json
LIMIT=0            # 0=完整清单；例如 2=只做两张图的开发检查
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="$PROJECT_DIR/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ADAPTER_ARGS=()
if [[ -n "$ADAPTER_DIR" ]]; then ADAPTER_ARGS=(--adapter "$ADAPTER_DIR"); fi
"$PARTGLEE_PYTHON" -m affordance_jev.pipeline.run \
  --manifest "$MANIFEST" \
  --output "$OUTPUT_DIR" \
  --config "$PIPELINE_CONFIG" \
  --definitions "$AFFORDANCE_CONFIG" \
  --partglee-root "$PARTGLEE_ROOT" \
  --partglee-weights "$PARTGLEE_WEIGHTS" \
  --valen-model "$VALEN_MODEL" \
  --partglee-python "$PARTGLEE_PYTHON" \
  --valen-python "$VALEN_PYTHON" \
  --limit "$LIMIT" "${ADAPTER_ARGS[@]}" "$@"
