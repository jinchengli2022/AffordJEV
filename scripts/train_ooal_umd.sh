#!/usr/bin/env bash
set -euo pipefail

# 修改下面这些路径即可复现实验；不会使用测试集挑选 checkpoint。
PROJECT_DIR=/home/ljc/code/Affordance/AffordanceJEV
PYTHON=/home/ljc/miniconda3/envs/ooal-umd/bin/python
CONFIG=/home/ljc/code/Affordance/AffordanceJEV/configs/ooal_umd.json
OUTPUT_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/ooal_umd_novel_20261016/train
CHECKPOINT_DIR=/home/ljc/code/Affordance/AffordanceJEV/checkpoints/ooal_umd_novel_20261016
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1

# CONFIG 中明确指定：76 张训练清单、基础权重、输入尺寸、20,000 步及优化参数。
"$PYTHON" -m affordance_jev.baselines.ooal prepare --config "$CONFIG"
"$PYTHON" -m affordance_jev.baselines.ooal train \
  --config "$CONFIG" --output "$OUTPUT_DIR" --checkpoints "$CHECKPOINT_DIR"
