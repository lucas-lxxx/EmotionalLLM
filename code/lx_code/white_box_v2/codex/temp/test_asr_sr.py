#!/usr/bin/env python3
"""Test ASR with different sample rates."""

import sys
import torch
import torchaudio

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, decode_text

print("Loading model...")
model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
model.eval()

test_audio = '/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav'
print(f"\nTesting: {test_audio}")

# Test 1: Original sample rate
print("\n=== Test 1: Original sample rate (24000 Hz) ===")
waveform, sr = torchaudio.load(test_audio)
print(f"sr={sr}, shape={waveform.shape}")
waveform = waveform.to(cfg.device)
result1 = decode_text(
    model=model,
    tokenizer=tokenizer,
    waveform=waveform,
    sr=sr,
    prompt=cfg.asr_prompts[0],
    max_new_tokens=cfg.asr_max_new_tokens,
    temperature=cfg.temperature,
)
print(f"ASR Result: '{result1}'")

# Test 2: Resampled to 16000 Hz
print("\n=== Test 2: Resampled to 16000 Hz ===")
waveform, sr = torchaudio.load(test_audio)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
    sr = 16000
print(f"sr={sr}, shape={waveform.shape}")
waveform = waveform.to(cfg.device)
result2 = decode_text(
    model=model,
    tokenizer=tokenizer,
    waveform=waveform,
    sr=sr,
    prompt=cfg.asr_prompts[0],
    max_new_tokens=cfg.asr_max_new_tokens,
    temperature=cfg.temperature,
)
print(f"ASR Result: '{result2}'")

print(f"\n=== Comparison ===")
print(f"Same result: {result1 == result2}")
