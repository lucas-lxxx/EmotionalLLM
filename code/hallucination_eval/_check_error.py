#!/usr/bin/env python3
"""Check the first sample's error message from the results."""
import json
with open("/data1/lixiang/EmotionalLLM/code/hallucination_eval/results/voxtral_iemocap/hallucination_eval.json") as f:
    data = json.load(f)

s = data["samples"][0]
print(f"Sample: {s['sample_id']}")
print(f"Num clean responses: {len(s['clean_responses'])}")
r = s["clean_responses"][0]
print(f"Response: {r['response'][:300]}")
print(f"Parsed: {r['parsed']}")
