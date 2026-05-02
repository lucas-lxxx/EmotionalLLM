#!/bin/bash
# Q3 实验：STAA-Net SER 攻击迁移到 Voxtral
# 在服务器上按顺序执行所有步骤
#
# 使用方法：
#   cd /data1/lixiang/OPUS/Q3/code
#   bash run_all.sh [--skip-surrogate] [--skip-generator] [--skip-generate] [--skip-eval]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Q3: STAA-Net SER → Voxtral 迁移实验"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "时间: $(date)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU info"
echo ""

# 解析参数
SKIP_SURROGATE=0
SKIP_GENERATOR=0
SKIP_GENERATE=0
SKIP_EVAL=0
for arg in "$@"; do
    case $arg in
        --skip-surrogate) SKIP_SURROGATE=1 ;;
        --skip-generator) SKIP_GENERATOR=1 ;;
        --skip-generate)  SKIP_GENERATE=1 ;;
        --skip-eval)      SKIP_EVAL=1 ;;
    esac
done

# Step 1: 训练 Surrogate SER
if [ "$SKIP_SURROGATE" -eq 0 ]; then
    echo "=========================================="
    echo "Step 1/4: 训练 Surrogate SER"
    echo "=========================================="
    python train_surrogate.py
    echo ""
else
    echo "[跳过] Step 1: Surrogate SER"
fi

# Step 2: 训练 STAA-Net Generator
if [ "$SKIP_GENERATOR" -eq 0 ]; then
    echo "=========================================="
    echo "Step 2/4: 训练 STAA-Net Generator"
    echo "=========================================="
    python train_generator.py
    echo ""
else
    echo "[跳过] Step 2: STAA-Net Generator"
fi

# Step 3: 生成对抗样本
if [ "$SKIP_GENERATE" -eq 0 ]; then
    echo "=========================================="
    echo "Step 3/4: 生成对抗样本"
    echo "=========================================="
    python generate_adv.py
    echo ""
else
    echo "[跳过] Step 3: 生成对抗样本"
fi

# Step 4: Voxtral 评估
if [ "$SKIP_EVAL" -eq 0 ]; then
    echo "=========================================="
    echo "Step 4/4: Voxtral 评估"
    echo "=========================================="
    python eval_voxtral.py
    echo ""
else
    echo "[跳过] Step 4: Voxtral 评估"
fi

echo "=========================================="
echo "全部完成！$(date)"
echo "结果目录: /data1/lixiang/OPUS/Q3/results/"
echo "=========================================="
