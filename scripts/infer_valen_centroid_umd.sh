#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$HOME/miniconda3/envs/valen-4b-sft/bin/python"
SOURCE="$PROJECT_DIR/outputs/umd_visual_prompt_ablation/fill"
MANIFEST="$PROJECT_DIR/data/UMD/manifests/umd_novel_test.jsonl"
OUTPUT="$PROJECT_DIR/outputs/umd_centroid_20261017/test"  # 改为新的空目录
GPUS=(0 1 2 3 4 5)  # 每张卡处理不同图片；每个候选独立判断
PHASE="${1:-all}"  # all / prepare / infer / evaluate

export PYTHONPATH="$PROJECT_DIR/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
MODULE=affordance_jev.experiments.umd_valen.centroid
if [[ ! "$PHASE" =~ ^(all|prepare|infer|evaluate)$ ]]; then
  printf '阶段须为 all / prepare / infer / evaluate\n' >&2
  exit 2
fi

if [[ "$PHASE" == all || "$PHASE" == prepare ]]; then
  "$PYTHON" -m "$MODULE" prepare --root "$OUTPUT" --source "$SOURCE" --manifest "$MANIFEST"
fi

if [[ "$PHASE" == all || "$PHASE" == infer ]]; then
  mkdir -p "$OUTPUT/logs"
  PIDS=()
  for index in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "$PYTHON" -m "$MODULE" infer \
      --root "$OUTPUT" --shard-index "$index" --shard-count "${#GPUS[@]}" \
      > "$OUTPUT/logs/worker_$index.log" 2>&1 &
    PIDS+=("$!")
  done
  FAILED=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
  if [[ "$FAILED" != 0 ]]; then
    printf '有 worker 异常退出，请检查 %s/logs\n' "$OUTPUT" >&2
    exit 1
  fi
  "$PYTHON" -m "$MODULE" finalize --root "$OUTPUT"
fi

if [[ "$PHASE" == all || "$PHASE" == evaluate ]]; then
  "$PYTHON" -m affordance_jev.evaluation.evaluate \
    --manifest "$OUTPUT/evaluation_manifest.jsonl" --predictions "$OUTPUT" \
    --output "$OUTPUT/evaluation" --vis-limit 50
fi
