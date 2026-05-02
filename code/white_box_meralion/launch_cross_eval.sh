#!/bin/bash
# Launch cross-model transferability evaluation in parallel on GPUs 3, 4, 5.
PY=/data1/lixiang/miniconda3/envs/opens2s/bin/python

V_ROOT=/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result
OS_ROOT=/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result
M_ROOT=/data1/lixiang/EmotionalLLM/code/white_box_meralion/result

LOG_DIR=/data1/lixiang/EmotionalLLM/code/white_box_meralion/logs
mkdir -p $LOG_DIR

# ============================================================
# Target = MERaLiON (evaluates Voxtral adv + OpenS2S adv) -- GPU 3
# ============================================================
M_PAIRS="${V_ROOT}/Voxtral_IEMOCAP:V2M_IEMOCAP,${V_ROOT}/Voxtral_RAVDESS:V2M_RAVDESS,${V_ROOT}/Voxtral_EN:V2M_EN,${V_ROOT}/Voxtral_CN:V2M_CN,${OS_ROOT}/IEMOCAP:OS2M_IEMOCAP,${OS_ROOT}/RAVDESS:OS2M_RAVDESS,${OS_ROOT}/blackbox/EN:OS2M_EN,${OS_ROOT}/blackbox/CN:OS2M_CN"

cd /data1/lixiang/EmotionalLLM/code/white_box_meralion
CUDA_VISIBLE_DEVICES=3 nohup $PY cross_eval.py --pairs "$M_PAIRS" --max_per_dataset 60 > $LOG_DIR/cross_eval_meralion.log 2>&1 &
echo "MERaLiON evaluator PID=$!"

# ============================================================
# Target = Voxtral (evaluates OpenS2S adv + MERaLiON adv) -- GPU 4
# ============================================================
V_PAIRS="${OS_ROOT}/IEMOCAP:OS2V_IEMOCAP,${OS_ROOT}/RAVDESS:OS2V_RAVDESS,${OS_ROOT}/blackbox/EN:OS2V_EN,${OS_ROOT}/blackbox/CN:OS2V_CN,${M_ROOT}/MERaLiON_IEMOCAP:M2V_IEMOCAP,${M_ROOT}/MERaLiON_RAVDESS:M2V_RAVDESS,${M_ROOT}/MERaLiON_EN:M2V_EN,${M_ROOT}/MERaLiON_CN:M2V_CN"

cd /data1/lixiang/EmotionalLLM/code/white_box_voxtral
CUDA_VISIBLE_DEVICES=4 nohup $PY cross_eval.py --pairs "$V_PAIRS" --max_per_dataset 60 > $LOG_DIR/cross_eval_voxtral.log 2>&1 &
echo "Voxtral evaluator PID=$!"

# ============================================================
# Target = OpenS2S (evaluates Voxtral adv + MERaLiON adv) -- GPU 5
# ============================================================
OS_PAIRS="${V_ROOT}/Voxtral_IEMOCAP:V2OS_IEMOCAP,${V_ROOT}/Voxtral_RAVDESS:V2OS_RAVDESS,${V_ROOT}/Voxtral_EN:V2OS_EN,${V_ROOT}/Voxtral_CN:V2OS_CN,${M_ROOT}/MERaLiON_IEMOCAP:M2OS_IEMOCAP,${M_ROOT}/MERaLiON_RAVDESS:M2OS_RAVDESS,${M_ROOT}/MERaLiON_EN:M2OS_EN,${M_ROOT}/MERaLiON_CN:M2OS_CN"

cd /data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/ver2.0
CUDA_VISIBLE_DEVICES=5 nohup $PY cross_eval.py --pairs "$OS_PAIRS" --max_per_dataset 60 > $LOG_DIR/cross_eval_opens2s.log 2>&1 &
echo "OpenS2S evaluator PID=$!"

echo "All 3 cross-eval jobs launched. Logs in $LOG_DIR/"
