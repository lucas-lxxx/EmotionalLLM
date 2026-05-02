#!/usr/bin/env python3
"""Test different prompt formats with Voxtral to find one that works."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/data1/lixiang/EmotionalLLM/code/white_box_voxtral")

import torch
import torchaudio
from pathlib import Path
from voxtral_io import load_voxtral, decode_text

model, processor, _ = load_voxtral(Path("/data1/lixiang/Voxtral"), "cuda:0")

# Load a clean angry sample
wav_path = "/data1/lixiang/OpenS2S_dataset/data/IEMOCAP_esd/Session2/angry/Ses02F_script03_2_F038.wav"
waveform, sr = torchaudio.load(wav_path)
waveform = waveform.to("cuda:0")

# Test various prompt styles
prompts = [
    # Original emotion prompt (known to work)
    "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",
    # Binary QA style 1 (current)
    "Is the speaker expressing angry in this audio? Answer only Yes or No.",
    # Binary QA style 2
    "Question: Is the emotion in this audio angry? Answer: ",
    # Binary QA style 3
    "Is the emotion of this audio angry? Reply with Yes or No only.",
    # Binary QA style 4 (more conversational)
    "Listen to the audio. Is the speaker angry? Say Yes or No.",
    # Binary QA style 5 (minimal)
    "Is this audio angry? Yes or No:",
    # Binary QA style 6 (longer instruction)
    "Based on the audio you hear, determine if the speaker sounds angry. Output only one word: Yes or No.",
    # Multiple choice
    "Is the speaker in this audio: (A) angry (B) not angry? Answer A or B.",
    # Flipped: asking about the correct emotion
    "The speaker in this audio sounds angry. Is this correct? Answer Yes or No.",
]

for i, prompt in enumerate(prompts):
    resp = decode_text(model, processor, waveform, sr, prompt, 32, 0.0)
    print(f"[{i}] Prompt: {prompt[:70]}...")
    print(f"    Response: {resp}")
    print()
