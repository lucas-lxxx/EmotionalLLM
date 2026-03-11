#!/usr/bin/env python3
"""Test high-quality resampling methods."""

import sys
import torch
import torchaudio

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, decode_text

test_audio = '/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav'
waveform_orig, sr_orig = torchaudio.load(test_audio)

print(f"Original: sr={sr_orig}")
model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
model.eval()

# Test 1: torchaudio resample (default, fast but lower quality)
print("\n=== Method 1: torchaudio.functional.resample (fast, default quality) ===")
waveform_1 = torchaudio.functional.resample(waveform_orig, sr_orig, 16000)
waveform_1 = waveform_1.to(cfg.device)
result1 = decode_text(model, tokenizer, waveform_1, 16000, cfg.asr_prompts[0], cfg.asr_max_new_tokens, cfg.temperature)
print(f"ASR: {result1}")

# Test 2: torchaudio resample with sinc interpolation (higher quality)
print("\n=== Method 2: torchaudio.functional.resample (sinc_interp_kaiser, high quality) ===")
resampler = torchaudio.transforms.Resample(
    orig_freq=sr_orig,
    new_freq=16000,
    resampling_method="sinc_interp_kaiser",
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    beta=14.769656459379492,
)
waveform_2 = resampler(waveform_orig)
waveform_2 = waveform_2.to(cfg.device)
result2 = decode_text(model, tokenizer, waveform_2, 16000, cfg.asr_prompts[0], cfg.asr_max_new_tokens, cfg.temperature)
print(f"ASR: {result2}")

# Test 3: No resample (24kHz)
print("\n=== Method 3: No resample (24kHz original) ===")
waveform_3 = waveform_orig.to(cfg.device)
result3 = decode_text(model, tokenizer, waveform_3, sr_orig, cfg.asr_prompts[0], cfg.asr_max_new_tokens, cfg.temperature)
print(f"ASR: {result3}")
