#!/usr/bin/env python3
"""Comprehensive server check for hallucination evaluation experiment."""
import json, os, glob

def show_json_sample(path, label):
    with open(path) as f:
        d = json.load(f)
    d.pop("loss_trace", None)
    d.pop("grad_norm_trace", None)
    print(f"\n=== {label} ===")
    print(f"File: {path}")
    print(f"Keys: {list(d.keys())}")
    print(json.dumps(d, indent=2, ensure_ascii=False)[:2000])
    return d

# ============================================================
# 1. Check all result directories: WAV + JSON counts
# ============================================================
print("=" * 60)
print("1. RESULT INVENTORY")
print("=" * 60)

result_dirs = {
    "Voxtral/IEMOCAP": "/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_IEMOCAP",
    "Voxtral/RAVDESS": "/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_RAVDESS",
    "Voxtral/ESD-EN": "/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_EN",
    "Voxtral/ESD-CN": "/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_CN",
    "MERaLiON/IEMOCAP": "/data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_IEMOCAP",
    "MERaLiON/RAVDESS": "/data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_RAVDESS",
    "MERaLiON/ESD-EN": "/data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_EN",
    "MERaLiON/ESD-CN": "/data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_CN",
    "OpenS2S/IEMOCAP": "/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/IEMOCAP",
    "OpenS2S/RAVDESS": "/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/RAVDESS",
    "OpenS2S/ESD": "/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/ESDfinal",
}

for name, path in result_dirs.items():
    if not os.path.exists(path):
        print(f"  {name:25s}: NOT FOUND")
        continue
    wavs = glob.glob(os.path.join(path, "**/*.wav"), recursive=True)
    jsons = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
    # Exclude summary JSONs
    per_sample_jsons = [j for j in jsons if "summary" not in os.path.basename(j)]
    print(f"  {name:25s}: {len(wavs):5d} WAVs, {len(per_sample_jsons):5d} JSONs")

# ============================================================
# 2. Per-sample JSON structure for each model
# ============================================================
print("\n" + "=" * 60)
print("2. PER-SAMPLE JSON STRUCTURES")
print("=" * 60)

# Voxtral
vox_jsons = glob.glob("/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_IEMOCAP/**/*.json", recursive=True)
vox_jsons = [j for j in vox_jsons if "summary" not in os.path.basename(j)]
if vox_jsons:
    show_json_sample(vox_jsons[0], "Voxtral per-sample")

# MERaLiON
mer_jsons = glob.glob("/data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_IEMOCAP/**/*.json", recursive=True)
mer_jsons = [j for j in mer_jsons if "summary" not in os.path.basename(j)]
if mer_jsons:
    show_json_sample(mer_jsons[0], "MERaLiON per-sample")

# OpenS2S IEMOCAP
o2s_jsons = glob.glob("/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/IEMOCAP/**/*.json", recursive=True)
o2s_jsons = [j for j in o2s_jsons if "summary" not in os.path.basename(j)]
if o2s_jsons:
    show_json_sample(o2s_jsons[0], "OpenS2S IEMOCAP per-sample")

# OpenS2S ESD
o2s_esd_jsons = glob.glob("/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/ESDfinal/**/*.json", recursive=True)
o2s_esd_jsons = [j for j in o2s_esd_jsons if "summary" not in os.path.basename(j) and "cleaned" not in os.path.basename(j) and "judge" not in os.path.basename(j)]
if o2s_esd_jsons:
    show_json_sample(o2s_esd_jsons[0], "OpenS2S ESD per-sample")

# ============================================================
# 3. Check OpenS2S model structure
# ============================================================
print("\n" + "=" * 60)
print("3. MODEL STRUCTURES")
print("=" * 60)
for model_path in ["/data1/lixiang/Voxtral", "/data1/lixiang/MERaLiON-2-3B", "/data1/lixiang/Opens2s/OpenS2S"]:
    if os.path.exists(model_path):
        items = os.listdir(model_path)
        has_config = "config.json" in items
        print(f"  {model_path}: {len(items)} items, config.json={has_config}")
    else:
        print(f"  {model_path}: NOT FOUND")

# ============================================================
# 4. Check OpenS2S code structure (ver2.0)
# ============================================================
print("\n" + "=" * 60)
print("4. OPENS2S CODE STRUCTURE")
print("=" * 60)
v2_code = "/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/ver2.0"
if os.path.exists(v2_code):
    for item in sorted(os.listdir(v2_code)):
        full = os.path.join(v2_code, item)
        sz = os.path.getsize(full) if os.path.isfile(full) else "DIR"
        print(f"  {item:40s} {sz}")

# ============================================================
# 5. Python environment
# ============================================================
print("\n" + "=" * 60)
print("5. PYTHON ENVIRONMENT")
print("=" * 60)
import sys
print(f"  Python: {sys.version}")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
except:
    print("  PyTorch: NOT FOUND")
try:
    import transformers
    print(f"  Transformers: {transformers.__version__}")
except:
    print("  Transformers: NOT FOUND")
try:
    from sentence_transformers import SentenceTransformer
    print("  SentenceTransformers: available")
except:
    print("  SentenceTransformers: NOT FOUND")
