#!/usr/bin/env python3
"""Test Voxtral on multiple samples to debug the all-No issue."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/data1/lixiang/EmotionalLLM/code/white_box_voxtral")

import torch
import torchaudio
import json, glob
from pathlib import Path
from voxtral_io import load_voxtral, decode_text

model, processor, _ = load_voxtral(Path("/data1/lixiang/Voxtral"), "cuda:0")

result_dir = Path("/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_IEMOCAP")
jsons = sorted(glob.glob(str(result_dir / "**/*.json"), recursive=True))
jsons = [j for j in jsons if "summary" not in os.path.basename(j)]

prompt = "Is the speaker expressing angry in this audio? Answer only Yes or No."

for jf in jsons[:5]:
    with open(jf) as f:
        meta = json.load(f)
    sid = meta["sample_id"]
    gt_emo = meta["ground_truth_emotion"]
    clean_path = meta["path"]

    wav, sr = torchaudio.load(clean_path)
    duration = wav.shape[1] / sr
    print(f"\n{sid}: gt={gt_emo}, duration={duration:.2f}s, sr={sr}, shape={wav.shape}")

    wav_gpu = wav.to("cuda:0")

    # Test emotion prompt (known to work)
    resp1 = decode_text(model, processor, wav_gpu, sr,
        "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",
        32, 0.0)
    print(f"  Emotion prompt: {resp1}")

    # Test binary QA
    q = f"Is the speaker expressing {gt_emo} in this audio? Answer only Yes or No."
    resp2 = decode_text(model, processor, wav_gpu, sr, q, 32, 0.0)
    print(f"  Binary QA ({gt_emo}): {resp2}")

    # Test language
    resp3 = decode_text(model, processor, wav_gpu, sr,
        "Is the speaker speaking in English in this audio? Answer only Yes or No.",
        32, 0.0)
    print(f"  Language QA: {resp3}")
