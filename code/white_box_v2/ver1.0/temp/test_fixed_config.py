#!/usr/bin/env python3
"""Test with fixed config (sample_rate=24000)."""

import sys
import torch
import torchaudio

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

# Import fresh config
import importlib
import config as config_module
importlib.reload(config_module)
from config import cfg

print(f"Config sample_rate: {cfg.sample_rate}")

from opens2s_io import load_opens2s, decode_text

print("\nLoading model...")
model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
model.eval()

# Load audio (should NOT resample now)
test_audio = '/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav'
waveform, sr = torchaudio.load(test_audio)
print(f"Loaded: sr={sr}, shape={waveform.shape}")

# This mimics run_attack.py's load_audio function
if sr != cfg.sample_rate:
    print(f"WARNING: Would resample from {sr} to {cfg.sample_rate}")
    waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)
    sr = cfg.sample_rate
else:
    print(f"No resampling needed (sr={sr} == cfg.sample_rate={cfg.sample_rate})")

waveform = waveform.to(cfg.device)

# Test ASR
print("\n=== ASR Test ===")
result = decode_text(
    model=model,
    tokenizer=tokenizer,
    waveform=waveform,
    sr=sr,
    prompt=cfg.asr_prompts[0],
    max_new_tokens=cfg.asr_max_new_tokens,
    temperature=cfg.temperature,
)
print(f"Result: '{result}'")

# Test emotion
print("\n=== Emotion Test ===")
result = decode_text(
    model=model,
    tokenizer=tokenizer,
    waveform=waveform,
    sr=sr,
    prompt=cfg.emo_prompts[0],
    max_new_tokens=cfg.emo_max_new_tokens,
    temperature=cfg.temperature,
)
print(f"Result: '{result}'")
