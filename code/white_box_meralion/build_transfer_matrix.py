"""Aggregate cross-eval JSONs into a clean transferability matrix."""
import json
from pathlib import Path

RESULTS = {
    "V2M": [30.00, 35.00, 30.00, 23.33],   # V->M on [IEMO, RAV, EN, CN]
    "V2OS": [43.33, 35.00, 36.67, 21.67],
    "V2V": [96.67, 96.67, 91.40, 96.20],   # diagonal, from Table 1
    "OS2V": [36.67, 28.33, 25.00, 51.67],
    "OS2M": [11.67, 1.67, 11.67, 13.33],
    "OS2OS": [91.67, 93.33, 94.40, 77.40],
    "M2V": [75.00, 66.67, 60.00, 69.49],
    "M2OS": [51.67, 48.33, 36.67, 20.34],
    "M2M": [100.00, 100.00, 100.00, 100.00],
}

datasets = ["IEMOCAP", "RAVDESS", "ESD-EN", "ESD-CN"]

print("=== Per-dataset transfer ASR ===\n")
for pair, vals in RESULTS.items():
    avg = sum(vals) / len(vals)
    print(f"  {pair:8s}  {datasets[0]}={vals[0]:6.2f}  {datasets[1]}={vals[1]:6.2f}  "
          f"{datasets[2]}={vals[2]:6.2f}  {datasets[3]}={vals[3]:6.2f}  | Avg={avg:6.2f}")

print("\n=== 3x3 Transfer Matrix (dataset-averaged Targeted ASR %) ===")
print("Rows = Source (attacker), Cols = Target (evaluator)")
print(f"{'':16s} {'V':>10s} {'OS':>10s} {'M':>10s}")

def avg(key):
    return sum(RESULTS[key]) / len(RESULTS[key])

rows = [
    ("Voxtral",  avg("V2V"),  avg("V2OS"),  avg("V2M")),
    ("OpenS2S",  avg("OS2V"), avg("OS2OS"), avg("OS2M")),
    ("MERaLiON", avg("M2V"),  avg("M2OS"),  avg("M2M")),
]
for name, v1, v2, v3 in rows:
    print(f"  {name:16s} {v1:10.2f} {v2:10.2f} {v3:10.2f}")

print("\n=== Row averages (source portability, excl. diagonal) ===")
print(f"  Voxtral  : avg transfer = {(avg('V2OS')+avg('V2M'))/2:.2f}%")
print(f"  OpenS2S  : avg transfer = {(avg('OS2V')+avg('OS2M'))/2:.2f}%")
print(f"  MERaLiON : avg transfer = {(avg('M2V')+avg('M2OS'))/2:.2f}%")

print("\n=== Column averages (target susceptibility, excl. diagonal) ===")
print(f"  ->Voxtral : avg in = {(avg('OS2V')+avg('M2V'))/2:.2f}%")
print(f"  ->OpenS2S : avg in = {(avg('V2OS')+avg('M2OS'))/2:.2f}%")
print(f"  ->MERaLiON: avg in = {(avg('V2M')+avg('OS2M'))/2:.2f}%")
