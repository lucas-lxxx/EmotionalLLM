#!/usr/bin/env python3
"""Test whether OpenS2S expects 16000Hz or can handle 24000Hz natively."""

import sys
import torch
import torchaudio

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, build_inputs, forward_logits

print("Loading model...")
model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
model.eval()

test_audio = '/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav'
waveform_orig, sr_orig = torchaudio.load(test_audio)
print(f"\nOriginal audio: sr={sr_orig}, shape={waveform_orig.shape}")

# Test 1: Check what OpenS2S's audio encoder expects
print("\n=== Checking audio encoder config ===")
audio_encoder = model.audio_encoder_model
print(f"Audio encoder type: {type(audio_encoder)}")
if hasattr(audio_encoder, 'config'):
    if hasattr(audio_encoder.config, 'sampling_rate'):
        print(f"Expected sampling_rate: {audio_encoder.config.sampling_rate}")
if hasattr(audio_encoder, 'feature_extractor'):
    print(f"Feature extractor: {audio_encoder.feature_extractor}")
    if hasattr(audio_encoder.feature_extractor, 'sampling_rate'):
        print(f"Feature extractor sampling_rate: {audio_encoder.feature_extractor.sampling_rate}")

# Test 2: Try passing 24000Hz directly
print("\n=== Test: Passing 24000Hz directly (no resample) ===")
try:
    waveform = waveform_orig.to(cfg.device)
    inputs = build_inputs(waveform, sr_orig, "Test prompt", tokenizer, cfg.device)
    print(f"build_inputs succeeded")
    print(f"  speech_values shape: {inputs['speech_values'].shape}")

    # Try forward pass
    outputs = forward_logits(model, inputs)
    print(f"forward_logits succeeded")
    print(f"  output shape: {outputs.logits.shape}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Try passing 16000Hz (resampled)
print("\n=== Test: Passing 16000Hz (resampled) ===")
try:
    waveform_16k = torchaudio.functional.resample(waveform_orig, sr_orig, 16000)
    waveform_16k = waveform_16k.to(cfg.device)
    inputs = build_inputs(waveform_16k, 16000, "Test prompt", tokenizer, cfg.device)
    print(f"build_inputs succeeded")
    print(f"  speech_values shape: {inputs['speech_values'].shape}")

    outputs = forward_logits(model, inputs)
    print(f"forward_logits succeeded")
    print(f"  output shape: {outputs.logits.shape}")
except Exception as e:
    print(f"ERROR: {e}")
