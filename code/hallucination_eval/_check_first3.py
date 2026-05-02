#!/usr/bin/env python3
"""Check first 3 samples from the full results."""
import json
path = "/data1/lixiang/EmotionalLLM/code/hallucination_eval/results/voxtral_iemocap/hallucination_eval.json"
with open(path) as f:
    data = json.load(f)

for s in data["samples"][:5]:
    print(f"\n{'='*60}")
    print(f"Sample: {s['sample_id']}")
    print(f"GT: {s['ground_truth_emotion']} -> Target: {s['target_emotion']}")
    print(f"Clean responses:")
    for r in s["clean_responses"]:
        correct = "OK" if r["parsed"] == r["gt"] else "WRONG"
        print(f"  [{r['id']:8s}] GT={r['gt']:3s} Resp={r['response'][:80]:80s} Parsed={r['parsed']:7s} {correct}")
    print(f"Adv responses:")
    for r in s["adv_responses"]:
        correct = "OK" if r["parsed"] == r["gt"] else "WRONG"
        print(f"  [{r['id']:8s}] GT={r['gt']:3s} Resp={r['response'][:80]:80s} Parsed={r['parsed']:7s} {correct}")
