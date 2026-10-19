#!/usr/bin/env bash
set -euo pipefail

# 本轮只运行固定 9 张样例／18 个功能对。手动填写路径与 GPU。
PROJECT_DIR=/home/ljc/code/Affordance/AffordanceJEV
MANIFEST=/home/ljc/code/Affordance/AffordanceJEV/data/UMD/manifests/umd_novel_test.jsonl
QUERIES=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_given_operation_20261018/test_queries.jsonl
CONFIG=/home/ljc/code/Affordance/AffordanceJEV/configs/umd_valen_point_union.json
# 前景严格要求 P(yes) > CONFIG 中的 foreground_selector.minimum_yes_probability。
# 当前为 0.96；后续部件功能判断仍使用原 yes/no 规则。换配置时使用新的输出目录。
DEFINITIONS=/home/ljc/code/Affordance/AffordanceJEV/configs/umd_affordances.json
PARTGLEE_ROOT=/home/ljc/code/Affordance/AffordanceJEV/third_party/PartGLEE
PARTGLEE_WEIGHTS=/home/ljc/code/Affordance/AffordanceJEV/models/partglee/PartGLEE_SwinL.pth
CLIP_MODEL=/home/ljc/code/Affordance/AffordanceJEV/models/clip_vit_base_patch32
VALEN_MODEL=/home/ljc/code/Affordance/AffordanceJEV/models/valen
BASELINE_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_given_operation_20261018
OUTPUT_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_valen_point_union_p096_20261019
PARTGLEE_PYTHON=/home/ljc/miniconda3/envs/partglee/bin/python
VALEN_PYTHON=/home/ljc/miniconda3/envs/valen-4b-sft/bin/python
GPU=0

export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1
"$PARTGLEE_PYTHON" -m affordance_jev.experiments.valen_foreground.run \
  --phase smoke --output "$OUTPUT_DIR" --config "$CONFIG" \
  --manifest "$MANIFEST" --queries "$QUERIES" --definitions "$DEFINITIONS" \
  --baseline "$BASELINE_DIR" --partglee-root "$PARTGLEE_ROOT" \
  --partglee-weights "$PARTGLEE_WEIGHTS" --clip-model "$CLIP_MODEL" \
  --valen-model "$VALEN_MODEL" --partglee-python "$PARTGLEE_PYTHON" \
  --valen-python "$VALEN_PYTHON" --gpus "$GPU"
