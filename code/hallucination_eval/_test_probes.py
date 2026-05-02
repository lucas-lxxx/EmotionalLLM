#!/usr/bin/env python3
"""Quick test: verify probe generation and sample discovery."""
import sys
sys.path.insert(0, "/data1/lixiang/EmotionalLLM/code/hallucination_eval")
from run_eval import discover_samples, generate_qa_probes, DATASET_LANGUAGE, MODEL_CONFIGS
from pathlib import Path

samples = discover_samples(MODEL_CONFIGS["voxtral"]["result_dirs"]["iemocap"], max_samples=2)
print(f"Found {len(samples)} samples")
for s in samples:
    sid = s["meta"]["sample_id"]
    print(f"\n  {sid}")
    print(f"    clean_wav: {s['clean_wav_path']}")
    print(f"    adv_wav: {s['adv_wav_path']}")
    probes = generate_qa_probes(s["meta"], DATASET_LANGUAGE["iemocap"])
    for p in probes:
        print(f"    [{p['id']:8s}] Q: {p['question'][:60]}... GT: {p['gt']}")
