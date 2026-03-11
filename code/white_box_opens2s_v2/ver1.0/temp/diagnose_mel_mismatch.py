#!/usr/bin/env python3
"""Test: Compare our mel features vs OpenS2S's internal processing."""

import sys
import torch
import torchaudio

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, _torch_log_mel, build_inputs

# Load model
model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)

# Load test audio
test_audio = '/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav'
waveform, sr = torchaudio.load(test_audio)
print(f"Audio: sr={sr}, shape={waveform.shape}")

# Our mel features (24kHz with 16kHz parameters)
our_mel = _torch_log_mel(waveform, sr)
print(f"\nOur mel features: shape={our_mel.shape}")
print(f"  n_mels={cfg.n_mels}, n_fft={cfg.n_fft}, hop={cfg.hop_length}")
print(f"  Time frames: {our_mel.shape[-1]}")
print(f"  Frame rate: {our_mel.shape[-1] / (waveform.shape[1]/sr):.1f} frames/sec")

# Check if model has feature extractor
print(f"\n=== Model Audio Encoder ===")
audio_encoder = model.audio_encoder_model
print(f"Type: {type(audio_encoder)}")
print(f"Config: {audio_encoder.config}")

# Check expected parameters
if hasattr(audio_encoder.config, 'num_mel_bins'):
    print(f"Expected n_mels: {audio_encoder.config.num_mel_bins}")

# Try to find feature extractor
print(f"\n=== Looking for feature extractor ===")
if hasattr(model, 'feature_extractor'):
    print(f"model.feature_extractor: {model.feature_extractor}")
if hasattr(model, 'audio_feature_extractor'):
    print(f"model.audio_feature_extractor: {model.audio_feature_extractor}")

# Check model's processor
from src.modeling_omnispeech import OmniSpeechModel
print(f"\n=== Check if model uses WhisperFeatureExtractor internally ===")
print("Model expects Whisper-style features:")
print("  - sampling_rate: 16000 Hz")
print("  - n_fft: 400")
print("  - hop_length: 160")
print("  - n_mels: 80 or 128")

print(f"\n=== Our configuration mismatch ===")
print(f"We provide: sr={sr}, n_fft={cfg.n_fft}, hop={cfg.hop_length}, n_mels={cfg.n_mels}")
print(f"Model expects: sr=16000, n_fft=400, hop=160, n_mels=80/128")
print(f"\nMismatch: sr={sr} != 16000")
print(f"  -> Temporal resolution wrong: {cfg.hop_length}/{sr}*1000 = {cfg.hop_length/sr*1000:.2f}ms")
print(f"  -> Should be: {160/16000*1000:.2f}ms")
