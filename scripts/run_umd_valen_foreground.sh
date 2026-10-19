#!/usr/bin/env bash
set -euo pipefail

# 手动修改以下路径与 GPU 编号。新实验使用新的输出目录；相同目录接续已完成阶段。
PROJECT_DIR=/home/ljc/code/Affordance/AffordanceJEV
MANIFEST=/home/ljc/code/Affordance/AffordanceJEV/data/UMD/manifests/umd_novel_test.jsonl
QUERIES=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_given_operation_20261018/test_queries.jsonl
CONFIG=/home/ljc/code/Affordance/AffordanceJEV/configs/umd_valen_foreground.json
DEFINITIONS=/home/ljc/code/Affordance/AffordanceJEV/configs/umd_affordances.json
PARTGLEE_ROOT=/home/ljc/code/Affordance/AffordanceJEV/third_party/PartGLEE
PARTGLEE_WEIGHTS=/home/ljc/code/Affordance/AffordanceJEV/models/partglee/PartGLEE_SwinL.pth
CLIP_MODEL=/home/ljc/code/Affordance/AffordanceJEV/models/clip_vit_base_patch32
VALEN_MODEL=/home/ljc/code/Affordance/AffordanceJEV/models/valen
BASELINE_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_given_operation_20261018
OUTPUT_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_valen_foreground_20261018
PARTGLEE_PYTHON=/home/ljc/miniconda3/envs/partglee/bin/python
VALEN_PYTHON=/home/ljc/miniconda3/envs/valen-4b-sft/bin/python
GPUS=0,1,2,3,4,5,6

# 每个 worker 由入口设置 CUDA_VISIBLE_DEVICES 为一个 GPU；无训练、无 SSH。
# 默认先检查 9 张图，再跑完整测试；传入 --phase smoke 可只检查样例。
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1
"$PARTGLEE_PYTHON" -m affordance_jev.experiments.valen_foreground.run \
  --output "$OUTPUT_DIR" --config "$CONFIG" --manifest "$MANIFEST" --queries "$QUERIES" \
  --definitions "$DEFINITIONS" --baseline "$BASELINE_DIR" \
  --partglee-root "$PARTGLEE_ROOT" --partglee-weights "$PARTGLEE_WEIGHTS" \
  --clip-model "$CLIP_MODEL" --valen-model "$VALEN_MODEL" \
  --partglee-python "$PARTGLEE_PYTHON" --valen-python "$VALEN_PYTHON" --gpus "$GPUS" "$@"
