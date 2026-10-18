#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/ljc/code/Affordance/AffordanceJEV
PYTHON=/home/ljc/miniconda3/envs/ooal-umd/bin/python
CONFIG=/home/ljc/code/Affordance/AffordanceJEV/configs/ooal_umd.json
TEST_MANIFEST=/home/ljc/code/Affordance/AffordanceJEV/data/UMD/manifests/umd_novel_test.jsonl
CHECKPOINT=/home/ljc/code/Affordance/AffordanceJEV/checkpoints/ooal_umd_novel_20261016/iteration_020000.pth
OUTPUT_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/ooal_umd_novel_20261016/test
REFERENCE_RESULTS=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_novel_test
COMPARISON_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/ooal_umd_novel_20261016/comparison
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1

# 全部 2,661 张图；RGB 进入模型，GT 只在下一个独立评测步骤读取。
"$PYTHON" -m affordance_jev.baselines.ooal infer \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --manifest "$TEST_MANIFEST" --output "$OUTPUT_DIR"
"$PYTHON" -m affordance_jev.evaluation.evaluate \
  --manifest "$OUTPUT_DIR/evaluation_manifest.jsonl" \
  --predictions "$OUTPUT_DIR" --output "$OUTPUT_DIR/evaluation" --vis-limit 50
"$PYTHON" -m affordance_jev.baselines.ooal.compare \
  --ooal "$OUTPUT_DIR" --ours "$REFERENCE_RESULTS" --output "$COMPARISON_DIR"
