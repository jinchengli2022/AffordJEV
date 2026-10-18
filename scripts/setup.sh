#!/usr/bin/env bash
# 只在新主机显式运行：环境由 Conda 管理，存放在仓库外。
set -euo pipefail

PARTGLEE_ENV="${PARTGLEE_ENV:-partglee}"
VALEN_ENV="${VALEN_ENV:-valen-4b-sft}"
MAX_JOBS="${MAX_JOBS:-4}"  # PartGLEE CUDA 扩展的并行编译进程数

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:---help}"

usage() {
  cat <<'EOF'
用法：bash scripts/setup.sh {all|partglee|valen|check}
  all       新建并安装两个外部 Conda 环境
  partglee  仅安装 PartGLEE 环境及 CUDA 扩展
  valen     仅安装 Valen 推理/训练环境
  check     检查现有两个环境，不安装或修改包

已存在的同名环境不会被覆盖。可用 PARTGLEE_ENV / VALEN_ENV 改名。
CUDA Toolkit 需事先安装；必要时先设置 CUDA_HOME。
EOF
}

case "$ACTION" in
  -h|--help) usage; exit 0 ;;
  all|partglee|valen|check) ;;
  *) usage >&2; exit 2 ;;
esac

command -v conda >/dev/null || { echo "请先安装并初始化 Conda。" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

check_environment() {
  conda run --no-capture-output -n "$1" python -m affordance_jev.deployment.check_environment --component "$2"
}

if [[ "$ACTION" == check ]]; then
  failed=0
  check_environment "$PARTGLEE_ENV" partglee || failed=1
  check_environment "$VALEN_ENV" valen || failed=1
  exit "$failed"
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo "这套版本清单针对 Linux x86_64 + NVIDIA CUDA。" >&2; exit 1;
}
command -v nvidia-smi >/dev/null || { echo "请先安装 NVIDIA 驱动。" >&2; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

# 先检查所有待创建名称，避免安装一半后才发现已有环境。
EXISTING_ENVS="$(conda env list --json)"
require_new_name() {
  if "$CONDA_BASE/bin/python" -c 'import json,os,sys; sys.exit(0 if any(os.path.basename(p)==sys.argv[1] for p in json.loads(sys.argv[2])["envs"]) else 1)' "$1" "$EXISTING_ENVS"; then
    echo "Conda 环境 '$1' 已存在，未做任何安装。请运行 check，或修改环境名称后重新运行。" >&2
    exit 1
  fi
}
if [[ "$ACTION" == all || "$ACTION" == partglee ]]; then require_new_name "$PARTGLEE_ENV"; fi
if [[ "$ACTION" == all || "$ACTION" == valen ]]; then require_new_name "$VALEN_ENV"; fi
if [[ "$ACTION" == all && "$PARTGLEE_ENV" == "$VALEN_ENV" ]]; then
  echo "PartGLEE 与 Valen 依赖不同，必须使用不同环境名称。" >&2; exit 1
fi

if [[ "$ACTION" == all || "$ACTION" == partglee ]]; then
  if [[ -z "${CUDA_HOME:-}" ]]; then
    command -v nvcc >/dev/null || { echo "未找到 nvcc，请安装 CUDA Toolkit 并设置 CUDA_HOME。" >&2; exit 1; }
    CUDA_HOME="$(dirname -- "$(dirname -- "$(command -v nvcc)")")"
  fi
  [[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "CUDA_HOME/bin/nvcc 不存在。" >&2; exit 1; }
  "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12\.' || {
    echo "PartGLEE 的 PyTorch cu121 扩展需使用 CUDA 12.x Toolkit 编译。" >&2; exit 1;
  }
  command -v gcc >/dev/null && command -v g++ >/dev/null || {
    echo "请先安装 gcc / g++（例如系统的 build-essential）。" >&2; exit 1;
  }
  export CUDA_HOME MAX_JOBS
fi

install_partglee() {
  conda create --yes --override-channels -c conda-forge -n "$PARTGLEE_ENV" python=3.10.22 pip
  conda run --no-capture-output -n "$PARTGLEE_ENV" python -m pip install \
    pip==26.2.1 setuptools==80.10.2 wheel==0.48.0 ninja==1.13.2
  conda run --no-capture-output -n "$PARTGLEE_ENV" python -m pip install \
    torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -c "$PROJECT_DIR/requirements/constraints-partglee.txt"
  conda run --no-capture-output -n "$PARTGLEE_ENV" python -m pip install \
    -r "$PROJECT_DIR/requirements/partglee.txt"

  # 用已安装的 PyTorch 编译本仓库源码，禁止构建隔离另装 PyTorch。
  # 不用 --user；不自动安装 Detectron2 声明但本推理路径不用的格式化工具。
  conda run --no-capture-output -n "$PARTGLEE_ENV" python -c \
    'import torch; assert torch.cuda.is_available(), "新环境无法访问 CUDA，请先修复驱动/GPU 可见性。"'
  FORCE_CUDA=1 conda run --no-capture-output -n "$PARTGLEE_ENV" python -m pip install \
    --no-build-isolation --no-deps -e "$PROJECT_DIR/third_party/PartGLEE"
  FORCE_CUDA=1 conda run --no-capture-output -n "$PARTGLEE_ENV" python -m pip install \
    --no-build-isolation --no-deps \
    "$PROJECT_DIR/third_party/PartGLEE/projects/PartGLEE/partglee/models/pixel_decoder/ops"
  check_environment "$PARTGLEE_ENV" partglee
}

install_valen() {
  conda create --yes --override-channels -c conda-forge -n "$VALEN_ENV" python=3.11.16 pip
  conda run --no-capture-output -n "$VALEN_ENV" python -m pip install \
    pip==25.1.1 setuptools==75.1.0 wheel==0.48.0
  conda run --no-capture-output -n "$VALEN_ENV" python -m pip install \
    torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -c "$PROJECT_DIR/requirements/constraints-valen.txt"
  conda run --no-capture-output -n "$VALEN_ENV" python -m pip install \
    -r "$PROJECT_DIR/requirements/valen.txt"
  check_environment "$VALEN_ENV" valen
}

if [[ "$ACTION" == all || "$ACTION" == partglee ]]; then install_partglee; fi
if [[ "$ACTION" == all || "$ACTION" == valen ]]; then install_valen; fi
echo "环境安装完成。接下来按 models/manifest.json 补齐权重，再运行推理入口。"
