#!/usr/bin/env python3
"""Test decode with 24000Hz and proper mel parameters."""

import sys
import torch
import torchaudio

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

# Temporarily modify config
from config import cfg
cfg.sample_rate = 24000  # Use original sample rate
cfg.hop_length = 240  # 240/24000 = 10ms (same ratio as 160/16000)
cfg.n_fft = 600  # Proportional to sample rate
cfg.win_length = 600

from opens2s_io import load_opens2s, decode_text

print(f"Config: sr={cfg.sample_rate}, hop={cfg.hop_length}, n_fft={cfg.n_fft}")

print("\nLoading model...")
model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
model.eval()

test_audio = '/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav'
waveform, sr = torchaudio.load(test_audio)
print(f"Audio: sr={sr}, shape={waveform.shape}")

# No resampling!
waveform = waveform.to(cfg.device)

# Test ASR
print("\n=== ASR Test (24000Hz, no resample) ===")
result = decode_text(
    model=model,
    tokenizer=tokenizer,
    waveform=waveform,
    sr=sr,
    prompt=cfg.asr_prompts[0],
    max_new_tokens=cfg.asr_max_new_tokens,
    temperature=cfg.temperature,
)
print(f"ASR Result: '{result}'")

# Test emotion
print("\n=== Emotion Test (24000Hz, no resample) ===")
result = decode_text(
    model=model,
    tokenizer=tokenizer,
    waveform=waveform,
    sr=sr,
    prompt=cfg.emo_prompts[0],
    max_new_tokens=cfg.emo_max_new_tokens,
    temperature=cfg.temperature,
)
print(f"Emotion Result: '{result}'")
