#!/usr/bin/env bash
set -euo pipefail

# 2,661 张 RGB／5,232 个功能对，五种 Valen；先做九图检查，再自动全量运行。
PROJECT_DIR=/home/ljc/code/Affordance/AffordanceJEV
OUTPUT_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_valen_point_suite_p096_20261019
CONFIG=/home/ljc/code/Affordance/AffordanceJEV/configs/umd_valen_point_union_full.json
MANIFEST=/home/ljc/code/Affordance/AffordanceJEV/data/UMD/manifests/umd_novel_test.jsonl
QUERIES=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_given_operation_20261018/test_queries.jsonl
DEFINITIONS=/home/ljc/code/Affordance/AffordanceJEV/configs/umd_affordances.json
BASELINE_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_given_operation_20261018
SFT_CHECKPOINT=/home/ljc/code/Affordance/AffordanceJEV/checkpoints/umd_given_operation_20261018/final
PARTGLEE_ROOT=/home/ljc/code/Affordance/AffordanceJEV/third_party/PartGLEE
PARTGLEE_WEIGHTS=/home/ljc/code/Affordance/AffordanceJEV/models/partglee/PartGLEE_SwinL.pth
CLIP_MODEL=/home/ljc/code/Affordance/AffordanceJEV/models/clip_vit_base_patch32
VALEN_MODEL=/home/ljc/code/Affordance/AffordanceJEV/models/valen
PARTGLEE_PYTHON=/home/ljc/miniconda3/envs/partglee/bin/python
VALEN_PYTHON=/home/ljc/miniconda3/envs/valen-4b-sft/bin/python
GPUS=0,1,2,3,4,5,6

# 前景门槛在 CONFIG 的 foreground_selector.minimum_yes_probability 中直接修改。
# 五组：fill / outline / centroid / definition / sft。后续功能判断规则不变。
# SFT 只加载已有 LoRA＋判断头，不重训；OOAL 与 Affordance-R1 沿用原对照。
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1
"$PARTGLEE_PYTHON" -m affordance_jev.experiments.valen_foreground.suite \
  --root "$OUTPUT_DIR" --config "$CONFIG" --manifest "$MANIFEST" --queries "$QUERIES" \
  --definitions "$DEFINITIONS" --baseline "$BASELINE_DIR" --adapter "$SFT_CHECKPOINT" \
  --partglee-root "$PARTGLEE_ROOT" --partglee-weights "$PARTGLEE_WEIGHTS" \
  --clip-model "$CLIP_MODEL" --valen-model "$VALEN_MODEL" \
  --partglee-python "$PARTGLEE_PYTHON" --valen-python "$VALEN_PYTHON" --gpus "$GPUS" "$@"
