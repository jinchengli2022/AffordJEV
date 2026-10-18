#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="$HOME/miniconda3"  # Conda 安装位置；环境留在仓库外
TEST_MANIFEST="$PROJECT_DIR/data/UMD/manifests/umd_novel_test.jsonl"
CANDIDATE_CACHE="$PROJECT_DIR/outputs/umd_novel_test"
ADAPTER_DIR="$PROJECT_DIR/checkpoints/valen_76_abc_20261017/b0/final"
OUTPUT_DIR="$PROJECT_DIR/outputs/valen_b0_test"
PYTHON="$CONDA_ROOT/envs/valen-4b-sft/bin/python"
GPU_IDS="0"  # 多卡可写 0,1；每张图仍是 batch=1

export PYTHONPATH="$PROJECT_DIR/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1

# 复用固定候选；使用与 SFT 相同的填色输入和 dominant 功能定义。
# OUTPUT_DIR 必须为空。各 GPU 分担互不重复的图片，均为 batch=1。
"$PYTHON" -m affordance_jev.experiments.umd_valen.data variant \
  --source "$CANDIDATE_CACHE" --output "$OUTPUT_DIR" \
  --manifest "$TEST_MANIFEST" --style fill --protocol dominant

"$PYTHON" -m affordance_jev.experiments.umd_valen.parallel_infer \
  --root "$OUTPUT_DIR" --adapter "$ADAPTER_DIR" --gpus "$GPU_IDS"

"$PYTHON" -m affordance_jev.evaluation.evaluate \
  --manifest "$OUTPUT_DIR/evaluation_manifest.jsonl" \
  --predictions "$OUTPUT_DIR" --output "$OUTPUT_DIR/evaluation" --vis-limit 0
