#!/usr/bin/env python3
"""Quick test: run 2-sample evaluation on Voxtral/IEMOCAP."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/data1/lixiang/EmotionalLLM/code/hallucination_eval")

# Monkey-patch MAX_SAMPLES for quick test
import run_eval
run_eval.MAX_SAMPLES = 2

# Remove previous test results if any
import shutil
test_dir = "/data1/lixiang/EmotionalLLM/code/hallucination_eval/results/voxtral_iemocap"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

run_eval.evaluate("voxtral", "iemocap")

# Show results
import json
result_file = os.path.join(test_dir, "hallucination_eval.json")
if os.path.exists(result_file):
    with open(result_file) as f:
        data = json.load(f)
    for s in data["samples"]:
        print(f"\n--- {s['sample_id']} ---")
        print("Clean responses:")
        for r in s["clean_responses"]:
            mark = "✓" if r["parsed"] == r["gt"] else "✗"
            print(f"  {mark} [{r['id']}] Response: {r['response'][:60]} → {r['parsed']}")
        print("Adv responses:")
        for r in s["adv_responses"]:
            mark = "✓" if r["parsed"] == r["gt"] else "✗"
            print(f"  {mark} [{r['id']}] Response: {r['response'][:60]} → {r['parsed']}")
