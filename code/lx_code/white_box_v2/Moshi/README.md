# Moshi White-Box Attack

White-box adversarial attack on Moshi speech model, adapted from OpenS2S attack framework.

## Overview

This project implements emotion manipulation attacks on the Moshi speech-to-speech model, targeting sad-to-happy emotion transformation on RAVDESS dataset.

## Directory Structure

```
Moshi/
├── config.py           # Attack configuration
├── moshi_io.py         # Moshi model I/O wrapper
├── attack_core.py      # Core attack algorithm (PGD + EoT)
├── eval_metrics.py     # Evaluation metrics (WER, SNR)
├── run_attack.py       # Main attack script
└── README.md           # This file
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Moshi model installed (`pip install moshi`)
- torchaudio
- RAVDESS dataset with pairs.json

## Quick Start

### 1. Configure Paths

Edit `config.py` to set correct paths:

```python
moshi_model_path = "/data1/lixiang/Moshi/moshiko-pytorch-bf16/kyutai/moshiko-pytorch-bf16"
ravdess_base = "/data1/lixiang/OpenS2S_dataset/RAVDESS"
pairs_json = "/data1/lixiang/OpenS2S_dataset/RAVDESS/pairs.json"
```

### 2. Run Attack

```bash
cd /data1/lixiang/lx_code/white_box_v2/Moshi
python run_attack.py
```

This will:
- Randomly select 10 sad audio samples from RAVDESS
- Attack each sample to flip emotion to happy
- Save adversarial audio files to `./results/attack_TIMESTAMP/`

## Attack Configuration

Key parameters in `config.py`:

```python
epsilon = 0.008          # L_inf perturbation constraint
total_steps = 60         # Attack iterations
stage_a_steps = 20       # Emotion-focused stage
lr = 0.003              # Learning rate
num_samples = 10        # Number of samples to attack
```

## Attack Methodology

**Two-Stage Optimization**:
- Stage A (20 steps): Focus on emotion manipulation
- Stage B (40 steps): Balance emotion + semantic preservation

**Multi-Objective Loss**:
- Emotion loss: Target emotion classification
- ASR loss: Semantic preservation
- Perceptual loss: Audio quality (STFT)

**EoT (Expectation over Transformation)**:
- Time shift: ±160 samples
- Gain: 0.8-1.2x
- Improves robustness

## Output

Results are saved to `./results/attack_TIMESTAMP/`:
- `sample_XXX_adv.wav`: Adversarial audio files
- `results_summary.json`: Attack results and metrics

## Important Notes

⚠️ **Current Limitations**:

1. **Emotion Loss**: The current implementation uses placeholder emotion loss functions. Moshi's emotion recognition capability needs to be explored and integrated.

2. **ASR Loss**: Semantic preservation loss is not yet fully implemented for Moshi's architecture.

3. **Model API**: Moshi's API differs from OpenS2S. The attack framework may need adjustments based on actual model behavior.

## Next Steps

To complete the implementation:

1. **Explore Moshi's Emotion Recognition**: Test how Moshi responds to emotion queries
2. **Implement Proper Loss Functions**: Adapt loss_emo and loss_asr for Moshi's output format
3. **Test and Validate**: Run attacks and verify effectiveness
4. **Add Evaluation**: Integrate WER, SNR, and emotion classification metrics

## References

- Moshi: https://github.com/kyutai-labs/moshi
- OpenS2S Attack Framework: `/data1/lixiang/lx_code/white_box_v2/codex/`
- RAVDESS Dataset: https://zenodo.org/record/1188976

## License

Research use only.
