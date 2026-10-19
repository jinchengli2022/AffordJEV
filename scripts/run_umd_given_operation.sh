#!/usr/bin/env bash
set -euo pipefail

# 新实验请手动改成空目录；--continue 只用于接续同一次实验。
PROJECT_DIR=/home/ljc/code/Affordance/AffordanceJEV
OUTPUT_DIR=/home/ljc/code/Affordance/AffordanceJEV/outputs/umd_given_operation_20261018
PYTHON=/home/ljc/miniconda3/envs/partglee/bin/python

# 数据、模型和普通 SFT 参数分别来自以下现有文件，启动时会保存快照：
# data/UMD/manifests/{ooal_novel_train_76,umd_novel_test}.jsonl
# configs/umd_pipeline_generic.json、configs/umd_affordances.json
# configs/valen_76_sft.json、configs/affordance_r1_umd.json、configs/ooal_umd.json
# 使用外部 Conda 环境。GPU 1/2/3 跑候选，1/2 同时运行 Valen，4/5 跑 R1；
# GPU 0/6 在 OOAL 完成后进行双卡 SFT，随后跑 SFT 推理。
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1
"$PYTHON" -m affordance_jev.experiments.conditional_umd.suite --root "$OUTPUT_DIR" "$@"
