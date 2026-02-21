#!/usr/bin/env python3
import torchaudio
from pathlib import Path

# Check multiple samples
samples = list(Path("/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Happy/adult/male/").glob("*.wav"))[:5]

sample_rates = set()
for s in samples:
    _, sr = torchaudio.load(str(s))
    sample_rates.add(sr)
    print(f"{s.name}: {sr}Hz")

print(f"\nAll sample rates in dataset: {sample_rates}")
