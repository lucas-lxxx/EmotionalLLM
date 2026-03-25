#!/bin/bash
# Q3 实验：REN Atrous CNN 生成器攻击迁移到 Voxtral
# 使用方法：
#   cd /data1/lixiang/OPUS/Q3_REN/code
#   bash run_all.sh [--skip-train] [--skip-generate] [--skip-eval]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Q3: REN Atrous CNN → Voxtral 迁移实验"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "时间: $(date)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU info"
echo ""

# 解析参数
SKIP_TRAIN=0
SKIP_GENERATE=0
SKIP_EVAL=0
for arg in "$@"; do
    case $arg in
        --skip-train)    SKIP_TRAIN=1 ;;
        --skip-generate) SKIP_GENERATE=1 ;;
        --skip-eval)     SKIP_EVAL=1 ;;
    esac
done

# Step 1: 训练 Atrous CNN Generator（复用 STAA 的 surrogate SER）
if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "=========================================="
    echo "Step 1/3: 训练 Atrous CNN Generator"
    echo "=========================================="
    python train_generator.py
    echo ""
else
    echo "[跳过] Step 1: 训练 Generator"
fi

# Step 2: 生成对抗样本
if [ "$SKIP_GENERATE" -eq 0 ]; then
    echo "=========================================="
    echo "Step 2/3: 生成对抗样本"
    echo "=========================================="
    python generate_adv.py
    echo ""
else
    echo "[跳过] Step 2: 生成对抗样本"
fi

# Step 3: Voxtral 评估
if [ "$SKIP_EVAL" -eq 0 ]; then
    echo "=========================================="
    echo "Step 3/3: Voxtral 评估"
    echo "=========================================="
    python eval_voxtral.py
    echo ""
else
    echo "[跳过] Step 3: Voxtral 评估"
fi

echo "=========================================="
echo "全部完成！$(date)"
echo "结果目录: /data1/lixiang/OPUS/Q3_REN/results/"
echo "=========================================="
