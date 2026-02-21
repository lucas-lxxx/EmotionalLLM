#!/bin/bash
# testN10 White-box Attack Experiment Runner
# This script activates the OpenS2S environment and runs the attack with GPU 6

set -e

echo "=========================================="
echo "testN10 White-box Attack Experiment"
echo "=========================================="

# Activate OpenS2S virtual environment
echo "Activating OpenS2S environment..."
source /data1/lixiang/Opens2s/OpenS2S/venv/bin/activate

# Set GPU to cuda:6
export CUDA_VISIBLE_DEVICES=6
echo "Using GPU: cuda:6 (mapped to cuda:0)"

# Navigate to testN10 directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Run the attack
echo ""
echo "Starting attack experiment..."
python run_testN10.py "$@"

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
