#!/bin/bash
# OpenS2S Modal Conflict Experiment - 一键运行脚本

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "OpenS2S Modal Conflict Experiment"
echo "========================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# 检查依赖
echo "[1/3] Checking dependencies..."
pip install -q -r requirements.txt

# 创建输出目录
echo "[2/3] Setting up directories..."
mkdir -p outputs/hidden_states outputs/results

# 运行实验
echo "[3/3] Running experiment..."
python scripts/run_experiment.py \
    --config configs/experiment_config.yaml \
    "$@"

echo ""
echo "Done!"
