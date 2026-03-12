#!/bin/bash
# Voxtral 白盒攻击批量实验启动脚本
# CN: 1000 samples (25 per emotion × 4 emotions × 10 speakers), 2 shards on GPU 1,3
# EN: 1000 samples (25 per emotion × 4 emotions × 10 speakers), 2 shards on GPU 4,5
# 使用 nohup 确保 SSH 断开后继续运行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Conda 初始化
eval "$(/data1/lixiang/miniconda3/bin/conda shell.bash hook)"
conda activate opens2s

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

ESD_CN="/data1/lixiang/OpenS2S_dataset/ESD/CN"
ESD_EN="/data1/lixiang/OpenS2S_dataset/ESD/EN"
RESULT_CN="$SCRIPT_DIR/result/Voxtral_CN"
RESULT_EN="$SCRIPT_DIR/result/Voxtral_EN"

mkdir -p "$RESULT_CN" "$RESULT_EN"

echo "=========================================="
echo "Starting batch attack experiments"
echo "CN: $ESD_CN -> $RESULT_CN (2 shards, CUDA 0,3 = nvidia-smi GPU 1,4)"
echo "EN: $ESD_EN -> $RESULT_EN (2 shards, CUDA 4,2 = nvidia-smi GPU 5,3)"
echo "Logs: $LOG_DIR"
echo "=========================================="

# CN shard 0 on CUDA 0 (= nvidia-smi GPU 1)
nohup bash -c "
eval \"\$(/data1/lixiang/miniconda3/bin/conda shell.bash hook)\" && conda activate opens2s && \
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python $SCRIPT_DIR/run_attack.py \
  --esd_root $ESD_CN \
  --results_dir $RESULT_CN \
  --shard_id 0 --num_shards 2
" > "$LOG_DIR/cn_shard0_cuda0.log" 2>&1 &
PID_CN0=$!
echo "CN shard 0 (CUDA 0 = nvidia-smi GPU 1): PID=$PID_CN0"

# CN shard 1 on CUDA 3 (= nvidia-smi GPU 4)
nohup bash -c "
eval \"\$(/data1/lixiang/miniconda3/bin/conda shell.bash hook)\" && conda activate opens2s && \
CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python $SCRIPT_DIR/run_attack.py \
  --esd_root $ESD_CN \
  --results_dir $RESULT_CN \
  --shard_id 1 --num_shards 2
" > "$LOG_DIR/cn_shard1_cuda3.log" 2>&1 &
PID_CN1=$!
echo "CN shard 1 (CUDA 3 = nvidia-smi GPU 4): PID=$PID_CN1"

# EN shard 0 on CUDA 4 (= nvidia-smi GPU 5)
nohup bash -c "
eval \"\$(/data1/lixiang/miniconda3/bin/conda shell.bash hook)\" && conda activate opens2s && \
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python $SCRIPT_DIR/run_attack.py \
  --esd_root $ESD_EN \
  --results_dir $RESULT_EN \
  --shard_id 0 --num_shards 2
" > "$LOG_DIR/en_shard0_cuda4.log" 2>&1 &
PID_EN0=$!
echo "EN shard 0 (CUDA 4 = nvidia-smi GPU 5): PID=$PID_EN0"

# EN shard 1 on CUDA 2 (= nvidia-smi GPU 3)
nohup bash -c "
eval \"\$(/data1/lixiang/miniconda3/bin/conda shell.bash hook)\" && conda activate opens2s && \
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python $SCRIPT_DIR/run_attack.py \
  --esd_root $ESD_EN \
  --results_dir $RESULT_EN \
  --shard_id 1 --num_shards 2
" > "$LOG_DIR/en_shard1_cuda2.log" 2>&1 &
PID_EN1=$!
echo "EN shard 1 (CUDA 2 = nvidia-smi GPU 3): PID=$PID_EN1"

echo ""
echo "=========================================="
echo "All 4 processes launched:"
echo "  CN shard 0 (CUDA 0 = nvidia-smi GPU 1): PID=$PID_CN0  log: $LOG_DIR/cn_shard0_cuda0.log"
echo "  CN shard 1 (CUDA 3 = nvidia-smi GPU 4): PID=$PID_CN1  log: $LOG_DIR/cn_shard1_cuda3.log"
echo "  EN shard 0 (CUDA 4 = nvidia-smi GPU 5): PID=$PID_EN0  log: $LOG_DIR/en_shard0_cuda4.log"
echo "  EN shard 1 (CUDA 2 = nvidia-smi GPU 3): PID=$PID_EN1  log: $LOG_DIR/en_shard1_cuda2.log"
echo ""
echo "Monitor: tail -f $LOG_DIR/*.log"
echo "Check progress: ls $RESULT_CN/*/*.json 2>/dev/null | wc -l && ls $RESULT_EN/*/*.json 2>/dev/null | wc -l"
echo "Aggregate after done: python run_attack.py --aggregate_only --results_dir $RESULT_CN"
echo "                      python run_attack.py --aggregate_only --results_dir $RESULT_EN"
echo "=========================================="

# 保存 PID 方便后续查看
echo "$PID_CN0 cn_shard0_gpu1" > "$LOG_DIR/pids.txt"
echo "$PID_CN1 cn_shard1_gpu3" >> "$LOG_DIR/pids.txt"
echo "$PID_EN0 en_shard0_gpu4" >> "$LOG_DIR/pids.txt"
echo "$PID_EN1 en_shard1_gpu5" >> "$LOG_DIR/pids.txt"
