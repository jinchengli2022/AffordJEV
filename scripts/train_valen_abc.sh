#!/usr/bin/env bash
set -euo pipefail

# 手动配置区。每次训练一个方案；s0r / a0 / a1 / b0 / b1 / c0 / c1。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$HOME/miniconda3/envs/valen-4b-sft/bin/python"
TRAIN_CONFIG="$PROJECT_DIR/configs/valen_76_abc/b0.json"
TRAIN_RECORDS="$PROJECT_DIR/outputs/valen_76/train_fill_dominant/records.jsonl"
TRAIN_VIEWS_DIR="$PROJECT_DIR/outputs/valen_76_abc/training_views" # A 用 a_records；B 用 b_records；C/S0R 不用。
BASE_MODEL="$PROJECT_DIR/models/valen"
WARMUP_HEAD="$PROJECT_DIR/checkpoints/valen_76/s0/321/warmup/head.safetensors"
OUTPUT_DIR="$PROJECT_DIR/outputs/valen_76_abc/b0"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/valen_76_abc/b0"
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# 只替换输入与输出路径；损失权重、seed、步数等直接沿用所选配置。
RUN_CONFIG="$OUTPUT_DIR/launch.json"
"$PYTHON" - "$TRAIN_CONFIG" "$TRAIN_RECORDS" "$TRAIN_VIEWS_DIR" "$BASE_MODEL" "$WARMUP_HEAD" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$RUN_CONFIG" <<'PY'
import json, sys
from pathlib import Path
from affordance_jev.paths import training_config
source, records, views, model, warmup, output, checkpoints, run_config = sys.argv[1:]
config = training_config(source)
for key, value in dict(records=records, model_path=model, warmup_head=warmup,
                       output=output, checkpoint_dir=checkpoints).items():
    config[key] = str(Path(value).resolve())
if config.get('views'):
    config['views'] = str((Path(views) / Path(config['views']).name).resolve())
for key in ['records', 'warmup_head'] + (['views'] if config.get('views') else []):
    if not Path(config[key]).is_file():
        raise FileNotFoundError(f'缺少 {key}：{config[key]}；先准备训练数据并运行 SFT warmup')
if not (Path(model) / 'config.json').is_file():
    raise FileNotFoundError(f'请先补全 Valen 权重：{model}')
for value in [output, checkpoints]:
    path = Path(value)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'输出非空，请修改脚本顶部输出目录：{path}')
Path(output).mkdir(parents=True, exist_ok=True)
with Path(run_config).open('x', encoding='utf-8') as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(f'方案：{config["method"]}；固定步数：{config["joint_steps"]}；配置：{run_config}')
PY

"$PYTHON" -m affordance_jev.experiments.umd_valen.abc.train \
  --config "$RUN_CONFIG" 2>&1 | tee "$OUTPUT_DIR/train.log"
