# testN10 White-box Attack Experiment

## Overview

This directory contains a complete white-box adversarial attack experiment on 10 ESD/Angry audio samples using the OpenS2S model. The attack manipulates emotion classification from "angry" to "happy" while attempting to preserve semantic content.

## Directory Structure

```
testN10/
├── config_testN10.py          # Experiment configuration
├── run_testN10.py              # Main experiment script
├── run_experiment.sh           # Shell script to run experiment
├── sample_list.txt             # List of 10 ESD/Angry audio files
├── experiment.log              # Full execution log
├── EXPERIMENT_REPORT.md        # Detailed experiment report
├── README.md                   # This file
└── results/                    # Experiment results
    ├── 00000_0001_000351.json  # Per-sample detailed results
    ├── 00000_0001_000351.wav   # Adversarial audio
    ├── ...                     # (10 samples total)
    ├── summary.json            # Aggregate statistics
    └── summary.csv             # Summary in CSV format
```

## Quick Start

### Run the Experiment

```bash
cd testN10
bash run_experiment.sh
```

This will:
1. Activate the OpenS2S virtual environment
2. Set GPU to cuda:6 (via CUDA_VISIBLE_DEVICES=6)
3. Run the attack on all 10 samples
4. Generate results in the `results/` directory

### View Results

```bash
# View summary statistics
cat results/summary.json
cat results/summary.csv

# View detailed report
cat EXPERIMENT_REPORT.md

# View per-sample results
cat results/00000_0001_000351.json

# Listen to adversarial audio
# (Use any audio player)
```

## Methodology

The experiment implements a white-box adversarial attack with:

1. **Two-Stage Optimization**:
   - Stage A (20 steps): Focus on emotion manipulation
   - Stage B (40 steps): Balance emotion + semantic preservation

2. **Multi-Objective Loss**:
   - Emotion loss: Cross-entropy on target emotion tokens
   - ASR loss: Self-consistency to preserve transcript
   - Perceptual loss: Multi-resolution STFT for audio quality

3. **Expectation over Transformation (EoT)**:
   - Time shift, gain, and noise augmentation
   - Improves robustness of adversarial examples

4. **Constraints**:
   - L_inf ≤ 0.008 (imperceptible perturbations)
   - Gradient-based optimization with Adam

## Results Summary

- **Emotion Success Rate**: 80% (8/10 samples)
- **Average SNR**: 18.06 dB
- **Average WER**: 1.2
- **Joint Success**: 0% (emotion + low WER)

See `EXPERIMENT_REPORT.md` for detailed analysis.

## Configuration

Key parameters in `config_testN10.py`:

```python
# Attack parameters
epsilon = 0.008              # L_inf bound
total_steps = 60             # Total optimization steps
stage_a_steps = 20           # Stage A steps
lr = 0.003                   # Learning rate

# Loss weights
lambda_emo = 1.0             # Emotion loss weight
lambda_asr_stage_a = 1e-4    # ASR loss (Stage A)
lambda_asr_stage_b = 1e-2    # ASR loss (Stage B)
lambda_per_stage_b = 1e-5    # Perceptual loss (Stage B)

# Target
target_emotion = "happy"     # Target emotion
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- OpenS2S model and environment
- torchaudio or soundfile
- jiwer (for WER computation)

## Files Description

### Core Scripts

- **config_testN10.py**: Configuration dataclass with all hyperparameters
- **run_testN10.py**: Main experiment script that:
  - Loads OpenS2S model
  - Processes each audio sample
  - Runs adversarial attack
  - Evaluates results
  - Saves outputs

- **run_experiment.sh**: Convenience script to:
  - Activate virtual environment
  - Set GPU device
  - Run experiment with logging

### Input

- **sample_list.txt**: List of 10 ESD/Angry audio file paths

### Output

- **results/*.json**: Per-sample results with:
  - Clean and adversarial emotion predictions
  - Clean and adversarial transcripts
  - Success metrics (emotion, WER)
  - Signal metrics (L_inf, L2, SNR)
  - Loss and gradient traces

- **results/*.wav**: Adversarial audio files

- **results/summary.json**: Aggregate statistics

- **experiment.log**: Full execution log

- **EXPERIMENT_REPORT.md**: Detailed analysis and recommendations

## Usage Examples

### Run on Different Samples

Edit `sample_list.txt` to include your audio files:

```
/path/to/audio1.wav
/path/to/audio2.wav
...
```

### Modify Attack Parameters

Edit `config_testN10.py`:

```python
# Stronger attack
epsilon = 0.01
total_steps = 100

# Different target emotion
target_emotion = "sad"

# Stronger semantic preservation
lambda_asr_stage_b = 0.1
```

### Use Different GPU

Edit `run_experiment.sh`:

```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 instead of 6
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python run_testN10.py
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or enable gradient checkpointing in `opens2s_io.py`.

### Import Errors

Ensure OpenS2S is in Python path:

```bash
export PYTHONPATH=/data1/lixiang/Opens2s/OpenS2S:$PYTHONPATH
```

### Audio Loading Errors

Install required libraries:

```bash
pip install torchaudio soundfile
```

## Citation

If you use this code, please cite the original OpenS2S paper and acknowledge the white-box attack methodology.

## License

This code is for research purposes only.

---

**Last Updated**: 2026-01-09
