#!/usr/bin/env bash
set -euo pipefail

# 手动配置区。Conda 环境留在项目外；本脚本不创建或复制环境。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$HOME/miniconda3/envs/partglee/bin/python"
DATASET_DIR="$PROJECT_DIR/data/UMD/part-affordance-dataset"
SELECTION_CONFIG="$PROJECT_DIR/configs/umd_train_76.json"
TRAIN_MANIFEST="$PROJECT_DIR/data/UMD/manifests/ooal_novel_train_76.jsonl"
TEST_MANIFEST="$PROJECT_DIR/data/UMD/manifests/umd_novel_test.jsonl"
PIPELINE_CONFIG="$PROJECT_DIR/configs/umd_pipeline.json"
AFFORDANCE_CONFIG="$PROJECT_DIR/configs/umd_affordances.json"
CANDIDATES_DIR="$PROJECT_DIR/outputs/valen_76/train_candidates"
RECORDS_DIR="$PROJECT_DIR/outputs/valen_76/train_fill_dominant"
ABC_VIEWS_DIR="$PROJECT_DIR/outputs/valen_76_abc/training_views"
PREPARE_ABC_VIEWS=true  # 普通 SFT 可改 false；A/B 训练需要这些变体。
export CUDA_VISIBLE_DEVICES=0
# PartGLEE 和 CLIP 权重分别固定读取 models/partglee/、models/clip_vit_base_patch32/。

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# 先检查目标目录，避免覆盖已有候选、监督记录或训练变体。
"$PYTHON" - "$TEST_MANIFEST" "$CANDIDATES_DIR" "$RECORDS_DIR" "$ABC_VIEWS_DIR" "$PREPARE_ABC_VIEWS" <<'PY'
import sys
from pathlib import Path
test, candidates, records, views, make_views = sys.argv[1:]
if not Path(test).is_file():
    raise FileNotFoundError(f'请先运行 scripts/prepare_umd.sh 生成完整测试清单：{test}')
for value in [candidates, records] + ([views] if make_views == 'true' else []):
    path = Path(value)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'输出非空，请为新实验修改脚本顶部目录：{path}')
PY

# 1. 从原始 UMD 重建固定 76 张训练图；已有清单只校验，不改写。
VERIFY=()
if [[ -e "$TRAIN_MANIFEST" ]]; then VERIFY=(--verify-only); fi
"$PYTHON" -m affordance_jev.datasets.umd_train \
  --dataset-dir "$DATASET_DIR" --selection-config "$SELECTION_CONFIG" \
  --test-manifest "$TEST_MANIFEST" --output "$TRAIN_MANIFEST" "${VERIFY[@]}"

# 2. 只给 PartGLEE RGB，生成前景和互不重叠的候选 mask；GT 不进入模型。
"$PYTHON" -m affordance_jev.experiments.umd_valen.data train-candidates \
  --manifest "$TRAIN_MANIFEST" --test-manifest "$TEST_MANIFEST" \
  --config "$PIPELINE_CONFIG" --definitions "$AFFORDANCE_CONFIG" \
  --output "$CANDIDATES_DIR"
"$PYTHON" -m affordance_jev.segmentation.partglee \
  --jobs "$CANDIDATES_DIR/model_jobs.jsonl"

# 3. 固定候选，按原始 GT 多数类别生成每区域监督；保留原始 GT 不做重映射。
"$PYTHON" -m affordance_jev.experiments.umd_valen.data variant \
  --source "$CANDIDATES_DIR" --output "$RECORDS_DIR" \
  --manifest "$TRAIN_MANIFEST" --style fill --protocol dominant --training \
  --config "$PIPELINE_CONFIG" --definitions "$AFFORDANCE_CONFIG"

# 4. 仅从这 76 张训练图生成 A/B 变体；C 和普通 SFT 直接用原始记录。
if [[ "$PREPARE_ABC_VIEWS" == "true" ]]; then
  "$PYTHON" -m affordance_jev.experiments.umd_valen.abc.data \
    --records "$RECORDS_DIR/records.jsonl" --manifest "$TRAIN_MANIFEST" \
    --definitions "$AFFORDANCE_CONFIG" --output "$ABC_VIEWS_DIR"
fi
printf '训练输入已生成：%s\n' "$RECORDS_DIR/records.jsonl"
