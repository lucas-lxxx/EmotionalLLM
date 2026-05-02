"""Compute per-source-emotion ASR and detailed metrics for MERaLiON."""
import json
import glob
import os
from collections import defaultdict

import numpy as np

os.chdir("/data1/lixiang/EmotionalLLM/code/white_box_meralion")

datasets = {
    "ESD-EN": "result/MERaLiON_EN",
    "ESD-CN": "result/MERaLiON_CN",
    "IEMOCAP": "result/MERaLiON_IEMOCAP",
    "RAVDESS": "result/MERaLiON_RAVDESS",
}

all_by_emo = defaultdict(lambda: {"ok": 0, "n": 0})
all_snrs = []
all_sems = []

for name, path in datasets.items():
    files = sorted(glob.glob(f"{path}/**/*.json", recursive=True))
    files = [f for f in files if "summary" not in os.path.basename(f)]
    by_emo = defaultdict(lambda: {"ok": 0, "n": 0})
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if "sample_id" not in d:
            continue
        src = d.get("source_emotion") or d.get("ground_truth_emotion") or d.get("emotion", "unknown")
        ok = d.get("success_emo", False)
        by_emo[src]["n"] += 1
        by_emo[src]["ok"] += int(ok)
        all_by_emo[src]["n"] += 1
        all_by_emo[src]["ok"] += int(ok)
        if "snr_db" in d:
            all_snrs.append(d["snr_db"])
        if "semantic_sim" in d:
            all_sems.append(d["semantic_sim"])
    print(f"{name}:")
    for e, v in sorted(by_emo.items()):
        print(f"  {e:10s} ASR={v['ok']/v['n']*100:6.2f}%  ({v['ok']}/{v['n']})")

print("\n=== Overall per-emotion (all 4 datasets) ===")
for e, v in sorted(all_by_emo.items()):
    print(f"  {e:10s} ASR={v['ok']/v['n']*100:6.2f}%  ({v['ok']}/{v['n']})")

if all_snrs:
    print(f"\nSNR: mean={np.mean(all_snrs):.2f} std={np.std(all_snrs):.2f} "
          f"min={min(all_snrs):.2f} max={max(all_snrs):.2f}")
if all_sems:
    print(f"SemSim: mean={np.mean(all_sems):.4f} std={np.std(all_sems):.4f}")
