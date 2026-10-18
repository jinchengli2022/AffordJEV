#!/usr/bin/env bash
set -euo pipefail

# 手动配置区。默认单卡，4 张源图通过梯度累积形成一个有效 batch。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$HOME/miniconda3/envs/valen-4b-sft/bin/python"
TRAIN_CONFIG="$PROJECT_DIR/configs/valen_76_sft.json"
TRAIN_RECORDS="$PROJECT_DIR/outputs/valen_76/train_fill_dominant/records.jsonl"
BASE_MODEL="$PROJECT_DIR/models/valen"
OUTPUT_DIR="$PROJECT_DIR/outputs/valen_76/s0/321"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/valen_76/s0/321"
STAGE="${1:-all}"  # all / warmup / joint；ABC 只需先执行 warmup。
NUM_PROCESSES=1    # 恢复历史四卡 S0 时改为 4，并把下一行改为 0,1,2,3。
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# 保存实际使用的配置；只替换路径，学习率、步数、seed 等保留 TRAIN_CONFIG 的值。
RUN_CONFIG="$OUTPUT_DIR/launch_${STAGE}.json"
"$PYTHON" - "$TRAIN_CONFIG" "$TRAIN_RECORDS" "$BASE_MODEL" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$STAGE" "$RUN_CONFIG" "$NUM_PROCESSES" <<'PY'
import json,sys
from pathlib import Path
from affordance_jev.paths import training_config
source, records, model, output, checkpoints, stage, run_config, processes = sys.argv[1:]
if stage not in {'all', 'warmup', 'joint'}:
    raise ValueError('训练阶段只能为 all / warmup / joint')
config = training_config(source)
processes = int(processes)
if processes < 1 or config['effective_images_per_step'] % processes:
    raise ValueError('NUM_PROCESSES 必须为正整数，并能整除有效 batch')
for key, value in dict(records=records, model_path=model, output=output, checkpoint_dir=checkpoints).items():
    config[key] = str(Path(value).resolve())
if not Path(records).is_file():
    raise FileNotFoundError(f'请先运行 scripts/prepare_valen_umd.sh：{records}')
if not (Path(model) / 'config.json').is_file():
    raise FileNotFoundError(f'请先补全 Valen 权重：{model}')
out, ckpt = Path(output), Path(checkpoints)
if stage == 'joint':
    if not (ckpt / 'warmup/head.safetensors').is_file():
        raise FileNotFoundError('请先执行 bash scripts/train_valen_umd.sh warmup')
    if any(p.name != 'warmup' for p in ckpt.iterdir()):
        raise FileExistsError(f'权重目录已有联合训练输出：{ckpt}')
    protected = [out / 'joint', out / 'joint.log', Path(run_config)]
else:
    protected = [out, ckpt]
for path in protected:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'输出非空，请修改脚本顶部输出目录：{path}')
out.mkdir(parents=True, exist_ok=True)
with Path(run_config).open('x', encoding='utf-8') as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(f'阶段：{stage}；有效 batch：{config["effective_images_per_step"]}；配置：{run_config}')
PY

if [[ "$STAGE" == "all" || "$STAGE" == "warmup" ]]; then
  "$PYTHON" -m affordance_jev.experiments.umd_valen.train \
    --config "$RUN_CONFIG" --stage warmup 2>&1 | tee "$OUTPUT_DIR/warmup.log"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "joint" ]]; then
  if (( NUM_PROCESSES > 1 )); then
    # 历史 4090 四卡设置：禁用 P2P/IB，使用 NCCL 主机通信；其它主机可按硬件调整。
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_PROCESSES" \
      -m affordance_jev.experiments.umd_valen.train \
      --config "$RUN_CONFIG" --stage joint 2>&1 | tee "$OUTPUT_DIR/joint.log"
  else
    "$PYTHON" -m affordance_jev.experiments.umd_valen.train \
      --config "$RUN_CONFIG" --stage joint 2>&1 | tee "$OUTPUT_DIR/joint.log"
  fi
fi
