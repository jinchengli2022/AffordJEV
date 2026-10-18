#!/usr/bin/env bash
set -euo pipefail

# Original checkpoint: no UMD training and no local LoRA adapter.
PROJECT_DIR="/home/ljc/code/Affordance/AffordanceJEV"
R1_SOURCE="/home/ljc/code/Affordance/2d_affordance/src"
PYTHON="/home/ljc/miniconda3/envs/affordance-r1-portable/bin/python"
MODEL="/home/ljc/code/Affordance/2d_affordance/model/pretrained/affordance-r1/huggingface"
SAM_CHECKPOINT="/home/ljc/code/Affordance/2d_affordance/model/pretrained/sam2-hiera-large/sam2_hiera_large.pt"
MANIFEST="$PROJECT_DIR/data/UMD/manifests/umd_novel_test.jsonl"
CONFIG="$PROJECT_DIR/configs/affordance_r1_umd.json"
DEFINITIONS="$PROJECT_DIR/configs/umd_affordances.json"
OUTPUT="$PROJECT_DIR/outputs/affordance_r1_umd_20261017/test"
GPUS="0,1,2,3,4,5,6"

export PYTHONPATH="$PROJECT_DIR/src:$R1_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$PYTHON" -m affordance_jev.baselines.affordance_r1 run \
  --manifest "$MANIFEST" --config "$CONFIG" --definitions "$DEFINITIONS" \
  --model "$MODEL" --sam-checkpoint "$SAM_CHECKPOINT" \
  --output "$OUTPUT" --gpus "$GPUS" "$@"
