#!/usr/bin/env python3
"""Test script to debug decode_text() function."""

import sys
import torch
import torchaudio
from pathlib import Path

# Add OpenS2S to path
sys.path.insert(0, "/data1/lixiang/Opens2s/OpenS2S")

from config import cfg
from opens2s_io import load_opens2s, decode_text

def main():
    # Load model
    print("Loading model...")
    model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
    model.eval()

    # Load a test audio file
    test_audio = "/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/male/25851.wav"
    print(f"Loading audio: {test_audio}")
    waveform, sr = torchaudio.load(test_audio)
    waveform = waveform.to(cfg.device)

    # Test emotion prompt
    print("\n=== Testing Emotion Prompt ===")
    prompt = cfg.emo_prompts[0]
    print(f"Prompt: {prompt}")
    result = decode_text(
        model=model,
        tokenizer=tokenizer,
        waveform=waveform,
        sr=sr,
        prompt=prompt,
        max_new_tokens=cfg.emo_max_new_tokens,
        temperature=cfg.temperature,
    )
    print(f"Result: '{result}'")

    # Test ASR prompt
    print("\n=== Testing ASR Prompt ===")
    prompt = cfg.asr_prompts[0]
    print(f"Prompt: {prompt}")
    result = decode_text(
        model=model,
        tokenizer=tokenizer,
        waveform=waveform,
        sr=sr,
        prompt=prompt,
        max_new_tokens=cfg.asr_max_new_tokens,
        temperature=cfg.temperature,
    )
    print(f"Result: '{result}'")

if __name__ == "__main__":
    main()
