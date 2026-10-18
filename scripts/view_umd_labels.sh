#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$HOME/miniconda3/envs/valen-4b-sft/bin/python"
CONFIG="$PROJECT_DIR/configs/umd_label_viewer.json"
export PYTHONPATH="$PROJECT_DIR/src"

# 生成可离线打开的标签界面；输出目录必须为空，不覆盖原始数据或已有界面。
"$PYTHON" -m affordance_jev.label_viewer.build_umd --config "$CONFIG" "$@"
