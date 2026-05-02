#!/bin/bash
# Run hallucination evaluation for all model-dataset combinations.
# Usage: bash run_all.sh [GPU_ID]
#
# Must run inside the opens2s conda environment:
#   source /data1/lixiang/miniconda3/bin/activate opens2s

set -e
GPU=${1:-2}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/data1/lixiang/miniconda3/envs/opens2s/bin/python"

echo "=== Hallucination Evaluation Pipeline ==="
echo "GPU: $GPU"
echo "Script dir: $SCRIPT_DIR"
echo ""

# Voxtral (4 datasets)
for ds in iemocap ravdess esd_en esd_cn; do
    echo ">>> Voxtral / $ds"
    $PYTHON "$SCRIPT_DIR/run_eval.py" --model voxtral --dataset $ds --gpu $GPU
    echo ""
done

# MERaLiON (4 datasets)
for ds in iemocap ravdess esd_en esd_cn; do
    echo ">>> MERaLiON / $ds"
    $PYTHON "$SCRIPT_DIR/run_eval.py" --model meralion --dataset $ds --gpu $GPU
    echo ""
done

# OpenS2S (IEMOCAP + RAVDESS only; ESD has no WAV files)
for ds in iemocap ravdess; do
    echo ">>> OpenS2S / $ds"
    $PYTHON "$SCRIPT_DIR/run_eval.py" --model opens2s --dataset $ds --gpu $GPU
    echo ""
done

echo "=== Computing metrics ==="
$PYTHON "$SCRIPT_DIR/compute_metrics.py" --results_dir "$SCRIPT_DIR/results"

echo "=== All done ==="
