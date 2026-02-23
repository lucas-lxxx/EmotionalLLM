#!/bin/bash
# 自动监控空闲GPU并重启挂掉的shard 0和shard 1
# 用法: nohup bash auto_restart.sh &

PYTHON=/data1/lixiang/miniconda3/envs/opens2s/bin/python
SCRIPT=/data1/lixiang/EmotionalLLM/code/white_box_v2/experiment/run_attack.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SHARDS_TO_RUN=(0 1)
STARTED=()

while [ ${#SHARDS_TO_RUN[@]} -gt 0 ]; do
    for gpu in 0 1 2 3 4; do
        # 跳过已用的GPU
        free=$(CUDA_VISIBLE_DEVICES=$gpu $PYTHON -c "
import torch
try:
    f,_=torch.cuda.mem_get_info(0)
    print(f//1024//1024)
except:
    print(0)
" 2>/dev/null)
        if [ "$free" -gt 40000 ] 2>/dev/null; then
            shard=${SHARDS_TO_RUN[0]}
            SHARDS_TO_RUN=("${SHARDS_TO_RUN[@]:1}")
            echo "$(date): Starting shard $shard on CUDA $gpu (${free}MiB free)"
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT --mode esd --shard_id $shard --num_shards 5 \
                > /tmp/attack_shard${shard}.log 2>&1 &
            echo "PID: $!"
            STARTED+=($shard)
            sleep 30  # 等模型加载完再检查下一张卡
            break
        fi
    done
    [ ${#SHARDS_TO_RUN[@]} -gt 0 ] && sleep 120  # 每2分钟检查一次
done

echo "$(date): All shards started: ${STARTED[*]}"
