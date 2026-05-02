#!/usr/bin/env python3
"""Debug: test sample 00001 with open-ended questions vs binary QA."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/data1/lixiang/EmotionalLLM/code/white_box_voxtral")

import torch
import torchaudio
from pathlib import Path
from voxtral_io import load_voxtral, decode_text

model, processor, _ = load_voxtral(Path("/data1/lixiang/Voxtral"), "cuda:0")

# Test on a sample that returned all "No"
wav_path = "/data1/lixiang/OpenS2S_dataset/data/IEMOCAP_esd/Session2/angry/Ses02M_script01_2_F002.wav"
wav, sr = torchaudio.load(wav_path)
wav = wav.to("cuda:0")
print(f"Sample 00001: duration={wav.shape[1]/sr:.2f}s")

tests = [
    ("Open emotion", "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise."),
    ("Open language", "What language is the speaker using in this audio? Answer with one word."),
    ("Open describe", "Describe what you hear in this audio in one sentence."),
    ("Binary angry", "Is the speaker expressing angry in this audio? Answer only Yes or No."),
    ("Binary English", "Is the speaker speaking in English in this audio? Answer only Yes or No."),
    ("Binary speech", "Does this audio contain human speech? Answer only Yes or No."),
    ("Binary 2 angry", "Does the speaker in this audio sound angry? Answer Yes or No."),
    ("Binary 2 Eng", "Is this audio in the English language? Yes or No."),
]

for label, prompt in tests:
    resp = decode_text(model, processor, wav, sr, prompt, 64, 0.0)
    print(f"  [{label:16s}] {resp[:100]}")

# Also test on sample 00000 (which worked) for comparison
print("\nSample 00000 (reference):")
wav2, sr2 = torchaudio.load("/data1/lixiang/OpenS2S_dataset/data/IEMOCAP_esd/Session2/angry/Ses02F_script03_2_F038.wav")
wav2 = wav2.to("cuda:0")
for label, prompt in tests:
    resp = decode_text(model, processor, wav2, sr2, prompt, 64, 0.0)
    print(f"  [{label:16s}] {resp[:100]}")
