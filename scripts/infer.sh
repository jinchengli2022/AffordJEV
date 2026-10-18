#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="$HOME/miniconda3"

# 手动填写：单张 RGB 或图片文件夹；输出必须是新的空目录。
RGB_INPUT="$PROJECT_DIR/data/input.png"
OUTPUT_DIR="$PROJECT_DIR/outputs/rgb_inference"
PIPELINE_CONFIG="$PROJECT_DIR/configs/umd_pipeline.json"
AFFORDANCE_CONFIG="$PROJECT_DIR/configs/umd_affordances.json"
PARTGLEE_ROOT="$PROJECT_DIR/third_party/PartGLEE"
PARTGLEE_WEIGHTS="$PROJECT_DIR/models/partglee/PartGLEE_SwinL.pth"
VALEN_MODEL="$PROJECT_DIR/models/valen"
PARTGLEE_PYTHON="$CONDA_ROOT/envs/partglee/bin/python"
VALEN_PYTHON="$CONDA_ROOT/envs/valen-4b-sft/bin/python"
ADAPTER_DIR=""  # 留空=原始模型；微调时填 final 目录并改用 umd_affordances_sft.json
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="$PROJECT_DIR/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ADAPTER_ARGS=()
if [[ -n "$ADAPTER_DIR" ]]; then ADAPTER_ARGS=(--adapter "$ADAPTER_DIR"); fi

"$PARTGLEE_PYTHON" -m affordance_jev.pipeline.images \
  --input "$RGB_INPUT" --output "$OUTPUT_DIR" \
  --config "$PIPELINE_CONFIG" --definitions "$AFFORDANCE_CONFIG" \
  --partglee-root "$PARTGLEE_ROOT" --partglee-weights "$PARTGLEE_WEIGHTS" \
  --valen-model "$VALEN_MODEL" \
  --partglee-python "$PARTGLEE_PYTHON" --valen-python "$VALEN_PYTHON" \
  "${ADAPTER_ARGS[@]}" "$@"
