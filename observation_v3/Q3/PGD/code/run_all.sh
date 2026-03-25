#!/bin/bash
# Q3 实验：PGD SER 攻击迁移到 Voxtral
# 使用方法：
#   cd /data1/lixiang/OPUS/Q3_PGD/code
#   bash run_all.sh [--skip-attack] [--skip-eval]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Q3: PGD SER → Voxtral 迁移实验"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "时间: $(date)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU info"
echo ""

# 解析参数
SKIP_ATTACK=0
SKIP_EVAL=0
for arg in "$@"; do
    case $arg in
        --skip-attack) SKIP_ATTACK=1 ;;
        --skip-eval)   SKIP_EVAL=1 ;;
    esac
done

# Step 1: PGD 攻击（复用 STAA 的 surrogate SER checkpoint）
if [ "$SKIP_ATTACK" -eq 0 ]; then
    echo "=========================================="
    echo "Step 1/2: PGD 攻击生成对抗样本"
    echo "=========================================="
    python pgd_attack.py
    echo ""
else
    echo "[跳过] Step 1: PGD 攻击"
fi

# Step 2: Voxtral 评估
if [ "$SKIP_EVAL" -eq 0 ]; then
    echo "=========================================="
    echo "Step 2/2: Voxtral 评估"
    echo "=========================================="
    python eval_voxtral.py
    echo ""
else
    echo "[跳过] Step 2: Voxtral 评估"
fi

echo "=========================================="
echo "全部完成！$(date)"
echo "结果目录: /data1/lixiang/OPUS/Q3_PGD/results/"
echo "=========================================="
