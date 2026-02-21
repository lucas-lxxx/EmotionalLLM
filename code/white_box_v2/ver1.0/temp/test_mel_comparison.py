#!/usr/bin/env python3
"""Compare mel features at different sample rates."""

import sys
import torch
import torchaudio

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

test_audio = '/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav'
waveform_orig, sr_orig = torchaudio.load(test_audio)

# Resample to 16000
waveform_16k = torchaudio.functional.resample(waveform_orig, sr_orig, 16000)

print(f"Original: sr={sr_orig}, duration={waveform_orig.shape[1]/sr_orig:.3f}s, shape={waveform_orig.shape}")
print(f"Resampled: sr=16000, duration={waveform_16k.shape[1]/16000:.3f}s, shape={waveform_16k.shape}")

# Create mel spectrograms
mel_24k = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000,
    n_fft=400,
    hop_length=160,
    win_length=400,
    n_mels=128,
)(waveform_orig)

mel_16k = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    win_length=400,
    n_mels=128,
)(waveform_16k)

print(f"\nMel spectrogram (24kHz audio, sr=24000): shape={mel_24k.shape}")
print(f"  Time frames: {mel_24k.shape[2]}")
print(f"  Frame rate: {mel_24k.shape[2] / (waveform_orig.shape[1]/sr_orig):.1f} frames/sec")

print(f"\nMel spectrogram (16kHz audio, sr=16000): shape={mel_16k.shape}")
print(f"  Time frames: {mel_16k.shape[2]}")
print(f"  Frame rate: {mel_16k.shape[2] / (waveform_16k.shape[1]/16000):.1f} frames/sec")

print(f"\n=== Key Issue ===")
print(f"With fixed hop_length=160:")
print(f"  24kHz: hop = 160/24000 = {160/24000*1000:.2f}ms per frame")
print(f"  16kHz: hop = 160/16000 = {160/16000*1000:.2f}ms per frame")
print(f"\nDifferent hop times -> different temporal resolution!")
print(f"This breaks the model's learned representations.")

print(f"\n=== Correct Parameters ===")
print(f"For 16kHz (Whisper standard): n_fft=400, hop=160, win=400")
print(f"For 24kHz (proportional): n_fft={int(400*24/16)}, hop={int(160*24/16)}, win={int(400*24/16)}")
