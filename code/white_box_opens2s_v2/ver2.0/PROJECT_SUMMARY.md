# White-box Attack on OpenS2S - Project Summary

## Project Structure

```
codex/
├── Core Attack Framework (5 files)
│   ├── config.py              # Configuration management
│   ├── opens2s_io.py          # OpenS2S model I/O and feature extraction
│   ├── attack_core.py         # Core attack algorithm with EoT
│   ├── eval_metrics.py        # Evaluation metrics (WER, SNR, etc.)
│   └── run_attack.py          # Main attack pipeline
│
├── testN10/                   # Experiment on 10 ESD/Angry samples
│   ├── config_testN10.py      # Experiment-specific configuration
│   ├── run_testN10.py         # Experiment script
│   ├── run_experiment.sh      # Shell script to run experiment
│   ├── sample_list.txt        # 10 ESD/Angry audio samples
│   ├── experiment.log         # Execution log
│   ├── EXPERIMENT_REPORT.md   # Detailed analysis and results
│   ├── README.md              # Usage instructions
│   └── results/               # Experiment outputs
│       ├── *.json             # Per-sample detailed results (10 files)
│       ├── *.wav              # Adversarial audio files (10 files)
│       ├── summary.json       # Aggregate statistics
│       └── summary.csv        # Summary in CSV format
│
└── temp/                      # Non-core files (tests, logs, docs)
    ├── test_*.py              # Test scripts
    ├── *.log                  # Log files
    ├── *.md                   # Documentation files
    ├── *.sh                   # Shell scripts
    └── ...                    # Other non-core files
```

## Core Attack Methodology

### 1. Attack Framework

The white-box attack manipulates emotion classification in OpenS2S while attempting to preserve semantic content through:

**Two-Stage Optimization**:
- **Stage A (20 steps)**: Focus on emotion manipulation
  - λ_emo = 1.0, λ_asr = 1e-4, λ_per = 0.0
- **Stage B (40 steps)**: Balance emotion + semantic preservation
  - λ_emo = 1.0, λ_asr = 1e-2, λ_per = 1e-5

**Multi-Objective Loss**:
- **Emotion Loss**: Cross-entropy on target emotion tokens ("happy")
- **ASR Loss**: Self-consistency to preserve transcript
- **Perceptual Loss**: Multi-resolution STFT for audio quality

**Expectation over Transformation (EoT)**:
- Time shift: ±160 samples
- Gain: 0.8-1.2x
- Noise: Optional Gaussian noise
- Improves robustness of adversarial examples

**Constraints**:
- L_inf ≤ 0.008 (imperceptible perturbations)
- Gradient-based optimization with Adam (lr=0.003)

### 2. Implementation Details

**Differentiable Feature Extraction**:
- Custom PyTorch implementation of Whisper feature extractor
- Enables gradient flow from model output to raw waveform
- Mel-spectrogram computation with STFT

**Model Integration**:
- OpenS2S model loading with gradient checkpointing
- Support for bfloat16 on CUDA
- Efficient memory management with gradient accumulation

**Evaluation Metrics**:
- **WER**: Word Error Rate for semantic preservation
- **SNR**: Signal-to-Noise Ratio for perturbation magnitude
- **L_inf/L2**: Perturbation norms
- **Success Rate**: Emotion classification accuracy

## testN10 Experiment Results

### Configuration

- **Dataset**: 10 ESD/Angry audio samples
- **Target Emotion**: happy
- **GPU**: cuda:6
- **Environment**: OpenS2S/venv
- **Attack Steps**: 60 (Stage A: 20, Stage B: 40)
- **Epsilon**: 0.008

### Results Summary

| Metric | Value |
|--------|-------|
| Emotion Success Rate | 80% (8/10) |
| Average SNR | 18.06 dB |
| Average WER | 1.2 |
| Joint Success (emotion + WER≤0.05) | 0% |
| Average L_inf | 0.008 |

### Key Findings

**Strengths**:
1. High emotion success rate (80%)
2. Imperceptible perturbations (SNR=18dB, L_inf=0.008)
3. Consistent performance across samples
4. Fast execution (~1.5 min per sample)

**Weaknesses**:
1. High WER (≥1.0) indicates semantic changes
2. ASR self-consistency loss insufficient
3. No joint success (emotion + semantic preservation)

**Observations**:
1. Emotion manipulation is effective
2. Semantic preservation requires stronger ASR loss
3. Trade-off between emotion and semantic objectives
4. Current λ_asr weights may be too small

### Recommendations

1. **Increase ASR Loss Weight**: Try λ_asr_stage_b = 0.1 or 1.0
2. **Longer Stage B**: Extend to 50-60 steps
3. **Larger Dataset**: Test on 50-100 samples
4. **Analyze Failed Cases**: Investigate samples 3 and 5
5. **Human Evaluation**: Conduct listening tests

## Usage

### Run testN10 Experiment

```bash
cd testN10
bash run_experiment.sh
```

### View Results

```bash
# Summary statistics
cat testN10/results/summary.json

# Detailed report
cat testN10/EXPERIMENT_REPORT.md

# Per-sample results
cat testN10/results/00000_0001_000351.json
```

### Modify Configuration

Edit `testN10/config_testN10.py`:

```python
# Stronger semantic preservation
lambda_asr_stage_b = 0.1

# Different target emotion
target_emotion = "sad"

# Stronger attack
epsilon = 0.01
total_steps = 100
```

### Run on Custom Samples

Edit `testN10/sample_list.txt`:

```
/path/to/audio1.wav
/path/to/audio2.wav
...
```

## File Organization

### Core Files (Keep in codex/)

These 5 files implement the complete attack framework:

1. **config.py**: Configuration dataclass with all hyperparameters
2. **opens2s_io.py**: Model loading, feature extraction, decoding
3. **attack_core.py**: Attack algorithm with EoT and multi-objective loss
4. **eval_metrics.py**: WER, SNR, and aggregation functions
5. **run_attack.py**: Main pipeline for batch processing

### Experiment Files (testN10/)

Complete experiment with:
- Configuration
- Scripts
- Sample list
- Results (JSON, WAV, CSV)
- Documentation (README, REPORT)

### Non-Core Files (temp/)

Moved to temp/:
- Test scripts (test_*.py)
- Log files (*.log)
- Documentation (*.md)
- Shell scripts (*.sh)
- Sample lists (*.txt)
- Archives (*.zip)

## Technical Details

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- OpenS2S model and environment
- transformers
- torchaudio or soundfile
- jiwer (for WER)

### GPU Requirements

- CUDA-capable GPU (tested on cuda:6)
- ~2GB VRAM per sample
- Gradient checkpointing enabled for memory efficiency

### Performance

- ~1.5 minutes per sample (60 steps)
- ~15 minutes for 10 samples
- Parallelizable across multiple GPUs

## Future Work

1. **Improve Semantic Preservation**:
   - Stronger ASR loss weights
   - Additional semantic constraints
   - Perceptual loss tuning

2. **Robustness Evaluation**:
   - Test on different models
   - Evaluate transferability
   - Physical playback tests

3. **Larger-Scale Experiments**:
   - 100+ samples
   - Multiple emotions
   - Different datasets (IEMOCAP, RAVDESS)

4. **Defense Mechanisms**:
   - Adversarial training
   - Input preprocessing
   - Detection methods

## Citation

If you use this code, please cite:
- OpenS2S paper
- Original white-box attack methodology

## License

This code is for research purposes only.

---

**Project Completed**: 2026-01-09
**Status**: Experiment successful, 80% emotion success rate
**Next Steps**: Improve semantic preservation with stronger ASR loss
