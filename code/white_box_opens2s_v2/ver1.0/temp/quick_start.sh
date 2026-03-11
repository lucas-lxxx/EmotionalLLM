#!/bin/bash
# Quick Start Script for OpenS2S White-box Attack
# Usage: ./quick_start.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}OpenS2S 白盒攻击快速启动脚本${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Step 1: Check Python
echo -e "${YELLOW}[1/7] 检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未找到${NC}"
    exit 1
fi
python_version=$(python3 --version)
echo -e "${GREEN}✅ $python_version${NC}"
echo ""

# Step 2: Check CUDA
echo -e "${YELLOW}[2/7] 检查 CUDA 环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ CUDA 可用${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠️  CUDA 不可用，将使用 CPU（速度较慢）${NC}"
fi
echo ""

# Step 3: Check OpenS2S model
echo -e "${YELLOW}[3/7] 检查 OpenS2S 模型...${NC}"
MODEL_PATH="/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S"
if [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✅ 模型路径存在: $MODEL_PATH${NC}"
    if [ -f "$MODEL_PATH/config.json" ]; then
        echo -e "${GREEN}✅ config.json 存在${NC}"
    else
        echo -e "${RED}❌ config.json 不存在${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ 模型路径不存在: $MODEL_PATH${NC}"
    exit 1
fi
echo ""

# Step 4: Check sample list
echo -e "${YELLOW}[4/7] 检查样本列表...${NC}"
if [ ! -f "sample_list.txt" ]; then
    echo -e "${RED}❌ sample_list.txt 不存在${NC}"
    exit 1
fi

sample_count=$(grep -v "^#" sample_list.txt | grep -v "^$" | wc -l)
echo -e "${GREEN}✅ 样本列表包含 $sample_count 个样本${NC}"

# Verify first 3 samples exist
echo "验证前 3 个样本文件..."
head -3 sample_list.txt | while read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    path=$(echo "$line" | awk '{print $1}')
    if [ -f "$path" ]; then
        echo -e "${GREEN}  ✅ $path${NC}"
    else
        echo -e "${RED}  ❌ $path 不存在${NC}"
    fi
done
echo ""

# Step 5: Check dependencies
echo -e "${YELLOW}[5/7] 检查 Python 依赖...${NC}"
dependencies=("torch" "torchaudio" "transformers" "numpy" "soundfile")
missing_deps=()

for dep in "${dependencies[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo -e "${GREEN}  ✅ $dep${NC}"
    else
        echo -e "${RED}  ❌ $dep 未安装${NC}"
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -gt 0 ]; then
    echo -e "${RED}缺少依赖包: ${missing_deps[*]}${NC}"
    echo -e "${YELLOW}请运行: pip install ${missing_deps[*]}${NC}"
    exit 1
fi
echo ""

# Step 6: Create results directory
echo -e "${YELLOW}[6/7] 创建结果目录...${NC}"
mkdir -p results
echo -e "${GREEN}✅ 结果目录: $(pwd)/results${NC}"
echo ""

# Step 7: Run options
echo -e "${YELLOW}[7/7] 准备运行实验...${NC}"
echo ""
echo -e "${BLUE}请选择运行模式：${NC}"
echo "  1) 测试运行（仅处理前 2 个样本）"
echo "  2) 完整运行（处理所有样本）"
echo "  3) 自定义范围"
echo "  4) 仅检查环境（不运行）"
echo ""
read -p "请输入选项 [1-4]: " option

case $option in
    1)
        echo -e "${GREEN}开始测试运行（前 2 个样本）...${NC}"
        export CUDA_VISIBLE_DEVICES=0
        python3 run_attack.py --start_idx 0 --end_idx 2
        ;;
    2)
        echo -e "${GREEN}开始完整运行（$sample_count 个样本）...${NC}"
        echo -e "${YELLOW}预计时间: $(($sample_count * 2)) 分钟（假设每样本 2 分钟）${NC}"
        read -p "确认继续？[y/N]: " confirm
        if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
            export CUDA_VISIBLE_DEVICES=0
            python3 run_attack.py
        else
            echo "已取消"
            exit 0
        fi
        ;;
    3)
        read -p "起始索引 (start_idx): " start_idx
        read -p "结束索引 (end_idx): " end_idx
        echo -e "${GREEN}运行样本 $start_idx 到 $end_idx...${NC}"
        export CUDA_VISIBLE_DEVICES=0
        python3 run_attack.py --start_idx $start_idx --end_idx $end_idx
        ;;
    4)
        echo -e "${GREEN}✅ 环境检查完成！${NC}"
        echo ""
        echo "手动运行命令："
        echo "  export CUDA_VISIBLE_DEVICES=0"
        echo "  python3 run_attack.py"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

# Check results
echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}✅ 实验完成！${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "查看结果："
echo "  ls -lh results/"
echo "  cat results/summary.json"
echo ""
echo "查看详细指南："
echo "  cat RUN_GUIDE.md"
