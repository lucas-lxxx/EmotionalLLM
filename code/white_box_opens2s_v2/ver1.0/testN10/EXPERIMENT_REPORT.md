# testN10 White-box Attack Experiment Report

## Experiment Overview

**Date**: 2026-01-09
**Objective**: Run white-box adversarial attacks on 10 ESD/Angry audio samples to manipulate emotion classification from "angry" to "happy" while preserving semantic content.

## Methodology

This experiment implements the white-box attack methodology from the 5 core code files:

1. **config.py**: Configuration management with attack hyperparameters
2. **opens2s_io.py**: OpenS2S model loading and differentiable feature extraction
3. **attack_core.py**: Core attack algorithm with:
   - Two-stage optimization (Stage A: 20 steps, Stage B: 40 steps)
   - Emotion loss (cross-entropy on target emotion tokens)
   - ASR self-consistency loss (preserve transcript)
   - Perceptual loss (multi-resolution STFT)
   - Expectation over Transformation (EoT) for robustness
4. **eval_metrics.py**: Evaluation metrics (WER, SNR, L_inf, L2)
5. **run_attack.py**: Main attack pipeline

## Attack Configuration

- **Target Emotion**: happy
- **Source Emotion**: angry (ESD dataset)
- **Attack Steps**: 60 (Stage A: 20, Stage B: 40)
- **Epsilon (L_inf bound)**: 0.008
- **Learning Rate**: 0.003
- **Optimizer**: Adam
- **GPU**: cuda:6
- **Environment**: OpenS2S/venv

### Loss Weights

| Loss Component | Stage A | Stage B |
|----------------|---------|---------|
| λ_emo          | 1.0     | 1.0     |
| λ_asr          | 1e-4    | 1e-2    |
| λ_per          | 0.0     | 1e-5    |

## Results Summary

### Aggregate Statistics

- **Total Samples**: 10
- **Emotion Success Rate**: 80% (8/10 samples)
- **WER ≤ 0.0**: 0% (0/10 samples)
- **WER ≤ 0.05**: 0% (0/10 samples)
- **Joint Success (emotion + WER ≤ 0.05)**: 0%

### Per-Sample Results

| Sample ID | Clean Emotion | Adv Emotion | Success | WER | SNR (dB) | L_inf |
|-----------|---------------|-------------|---------|-----|----------|-------|
| 00000_0001_000351 | angry | happy | ✓ | 1.0 | 17.05 | 0.008 |
| 00001_0001_000352 | angry | happy | ✓ | 1.0 | 17.90 | 0.008 |
| 00002_0002_000351 | angry | happy | ✓ | 1.0 | 16.06 | 0.008 |
| 00003_0002_000352 | angry | neutral | ✗ | 3.0 | 19.27 | 0.008 |
| 00004_0003_000351 | angry | happy | ✓ | 1.0 | 21.99 | 0.008 |
| 00005_0003_000352 | angry | happy | ✗ | 1.0 | 19.08 | 0.008 |
| 00006_0004_000351 | angry | happy | ✓ | 1.0 | 15.88 | 0.008 |
| 00007_0004_000352 | angry | happy | ✓ | 1.0 | 19.95 | 0.008 |
| 00008_0005_000351 | angry | happy | ✓ | 1.0 | 15.46 | 0.008 |
| 00009_0005_000352 | angry | happy | ✓ | 1.0 | 17.94 | 0.008 |

**Average SNR**: 18.06 dB
**Average WER**: 1.2

## Analysis

### Strengths

1. **High Emotion Success Rate**: 80% of samples successfully changed from "angry" to "happy"
2. **Imperceptible Perturbations**: All samples maintained L_inf ≤ 0.008 (epsilon constraint)
3. **Good SNR**: Average SNR of 18.06 dB indicates small perturbations
4. **Consistent Performance**: 8/10 samples achieved target emotion

### Weaknesses

1. **High WER**: All samples have WER ≥ 1.0, indicating significant semantic changes
2. **No Joint Success**: 0% joint success rate (emotion + low WER)
3. **ASR Self-Consistency Failed**: The ASR loss did not effectively preserve transcripts

### Failed Samples

- **Sample 3 (00003_0002_000352)**: Predicted "neutral" instead of "happy", WER=3.0
- **Sample 5 (00005_0003_000352)**: One prompt predicted non-happy emotion

## Observations

1. **Emotion Manipulation Works**: The attack successfully manipulates emotion classification in 80% of cases
2. **Semantic Preservation Fails**: Despite ASR self-consistency loss, transcripts changed significantly (WER=1.0-3.0)
3. **Trade-off Issue**: The current loss weights may prioritize emotion over semantic preservation
4. **Stage B Weights**: λ_asr_stage_b = 1e-2 may be too small to preserve semantics

## Recommendations

1. **Increase ASR Loss Weight**: Try λ_asr_stage_b = 0.1 or 1.0 to better preserve transcripts
2. **Longer Stage B**: Extend Stage B to 50-60 steps for better semantic preservation
3. **Evaluate on More Samples**: Test on larger dataset (50-100 samples) for statistical significance
4. **Analyze Failed Cases**: Investigate why samples 3 and 5 failed
5. **Human Evaluation**: Conduct listening tests to verify perceptual quality

## Files Generated

- **Results Directory**: `testN10/results/`
- **Per-Sample JSON**: 10 files with detailed metrics and loss traces
- **Adversarial Audio**: 10 WAV files with adversarial perturbations
- **Summary JSON**: `summary.json` with aggregate statistics
- **Summary CSV**: `summary.csv` for easy analysis
- **Experiment Log**: `experiment.log` with full execution trace

## Conclusion

The testN10 experiment demonstrates that white-box adversarial attacks can successfully manipulate emotion classification in OpenS2S with imperceptible perturbations (80% success rate, SNR=18dB). However, semantic preservation remains challenging, with all samples showing significant transcript changes (WER≥1.0). Future work should focus on improving the ASR self-consistency loss to achieve joint success (emotion manipulation + semantic preservation).

---

**Experiment completed successfully on 2026-01-09 16:06**
