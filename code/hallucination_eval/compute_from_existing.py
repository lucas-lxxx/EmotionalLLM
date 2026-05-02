#!/usr/bin/env python3
"""
Compute AHa-Bench-style hallucination metrics from EXISTING per-sample JSONs.

No new inference needed. Uses:
  - emo_pred_clean/adv (3 prompts each) for acoustic hallucination
  - semantic_sim for semantic hallucination
  - asr_text_clean/adv for content distortion

Outputs: ACC, ΔACC, Diff, Bias per model/dataset, plus aggregated.

Usage:
  python compute_from_existing.py
"""
import json
import os
import glob
from pathlib import Path
from collections import Counter

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path("/data1/lixiang/EmotionalLLM/code")

RESULT_MAP = {
    ("voxtral", "iemocap"):  BASE_DIR / "white_box_voxtral/result/Voxtral_IEMOCAP",
    ("voxtral", "ravdess"):  BASE_DIR / "white_box_voxtral/result/Voxtral_RAVDESS",
    ("voxtral", "esd_en"):   BASE_DIR / "white_box_voxtral/result/Voxtral_EN",
    ("voxtral", "esd_cn"):   BASE_DIR / "white_box_voxtral/result/Voxtral_CN",
    ("meralion", "iemocap"): BASE_DIR / "white_box_meralion/result/MERaLiON_IEMOCAP",
    ("meralion", "ravdess"): BASE_DIR / "white_box_meralion/result/MERaLiON_RAVDESS",
    ("meralion", "esd_en"):  BASE_DIR / "white_box_meralion/result/MERaLiON_EN",
    ("meralion", "esd_cn"):  BASE_DIR / "white_box_meralion/result/MERaLiON_CN",
    ("opens2s", "iemocap"):  BASE_DIR / "white_box_opens2s_v2/result/IEMOCAP",
    ("opens2s", "ravdess"):  BASE_DIR / "white_box_opens2s_v2/result/RAVDESS",
    ("opens2s", "esd_en"):   BASE_DIR / "white_box_opens2s_v2/result/ESDfinal",
    ("opens2s", "esd_cn"):   BASE_DIR / "white_box_opens2s_v2/result/ESDfinal",
}

SEMANTIC_THRESHOLD = 0.8
MAX_SAMPLES = 60


def load_samples(result_dir, max_samples=MAX_SAMPLES):
    """Load per-sample JSONs from a result directory."""
    json_files = sorted(glob.glob(str(result_dir / "**/*.json"), recursive=True))
    json_files = [f for f in json_files
                  if "summary" not in os.path.basename(f)
                  and "cleaned" not in os.path.basename(f)
                  and "judge" not in os.path.basename(f)
                  and "report" not in os.path.basename(f)]

    samples = []
    for jf in json_files[:max_samples]:
        with open(jf) as f:
            d = json.load(f)
        d.pop("loss_trace", None)
        d.pop("grad_norm_trace", None)
        samples.append(d)

    return samples


def compute_per_sample(sample):
    """Compute hallucination metrics for one sample from existing data.

    Uses emo_pred_clean/adv (list of 3 parsed emotion labels),
    semantic_sim, ground_truth_emotion, target_emotion.
    """
    gt = sample["ground_truth_emotion"]
    tgt = sample["target_emotion"]

    # Emotion predictions (3 prompts)
    emo_clean = sample.get("emo_pred_clean", [])
    emo_adv = sample.get("emo_pred_adv", [])

    # Filter empty predictions
    emo_clean = [e for e in emo_clean if e]
    emo_adv = [e for e in emo_adv if e]

    n_emo = max(len(emo_clean), len(emo_adv), 1)

    # --- Acoustic Hallucination: Emotion ACC ---
    acc_emo_clean = sum(1 for e in emo_clean if e == gt) / max(len(emo_clean), 1)
    acc_emo_adv = sum(1 for e in emo_adv if e == gt) / max(len(emo_adv), 1)

    # --- Semantic Hallucination: Content preservation ---
    sem_sim = sample.get("semantic_sim", 0.0)
    sem_preserved = 1.0 if sem_sim >= SEMANTIC_THRESHOLD else 0.0

    # --- Overall ACC (emotion probes + semantic probe) ---
    # 3 emotion probes + 1 semantic probe = 4 probes per audio
    n_probes = len(emo_clean) + 1  # +1 for semantic
    correct_clean = sum(1 for e in emo_clean if e == gt) + 1  # semantic always correct on clean
    correct_adv = sum(1 for e in emo_adv if e == gt) + (1 if sem_preserved else 0)
    acc_clean = correct_clean / n_probes if n_probes > 0 else 0.0
    acc_adv = correct_adv / n_probes if n_probes > 0 else 0.0

    # --- Diff (Emotion Response Inconsistency) ---
    # For 3 emotion prompts: fraction of pairs that disagree
    if len(emo_adv) >= 2:
        pairs = 0
        disagree = 0
        for i in range(len(emo_adv)):
            for j in range(i + 1, len(emo_adv)):
                pairs += 1
                if emo_adv[i] != emo_adv[j]:
                    disagree += 1
        diff_adv = disagree / pairs if pairs > 0 else 0.0
    else:
        diff_adv = 0.0

    # Same for clean
    if len(emo_clean) >= 2:
        pairs = 0
        disagree = 0
        for i in range(len(emo_clean)):
            for j in range(i + 1, len(emo_clean)):
                pairs += 1
                if emo_clean[i] != emo_clean[j]:
                    disagree += 1
        diff_clean = disagree / pairs if pairs > 0 else 0.0
    else:
        diff_clean = 0.0

    # --- Bias: P(predict target) on adversarial audio ---
    bias_adv = sum(1 for e in emo_adv if e == tgt) / max(len(emo_adv), 1)
    bias_clean = sum(1 for e in emo_clean if e == tgt) / max(len(emo_clean), 1)

    return {
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "acc_emo_clean": acc_emo_clean,
        "acc_emo_adv": acc_emo_adv,
        "sem_preserved": sem_preserved,
        "sem_sim": sem_sim,
        "diff_clean": diff_clean,
        "diff_adv": diff_adv,
        "bias_clean": bias_clean,
        "bias_adv": bias_adv,
    }


def compute_aggregate(samples):
    """Aggregate metrics across samples."""
    per = [compute_per_sample(s) for s in samples]
    if not per:
        return {}

    n = len(per)

    def avg(key):
        return sum(p[key] for p in per) / n

    result = {
        "n": n,
        # Overall ACC (emotion + semantic)
        "ACC_clean": avg("acc_clean"),
        "ACC_adv": avg("acc_adv"),
        "dACC": avg("acc_clean") - avg("acc_adv"),
        # Emotion-only ACC
        "ACC_emo_clean": avg("acc_emo_clean"),
        "ACC_emo_adv": avg("acc_emo_adv"),
        "dACC_emo": avg("acc_emo_clean") - avg("acc_emo_adv"),
        # Semantic
        "Sem_rate": avg("sem_preserved"),
        "Sem_sim_avg": avg("sem_sim"),
        # Diff (consistency)
        "Diff_clean": avg("diff_clean"),
        "Diff_adv": avg("diff_adv"),
        # Bias toward target
        "Bias_clean": avg("bias_clean"),
        "Bias_adv": avg("bias_adv"),
    }
    return result


def main():
    all_results = {}

    for (model, dataset), result_dir in sorted(RESULT_MAP.items()):
        if not result_dir.exists():
            continue

        # For OpenS2S ESD, separate CN/EN by checking audio paths
        samples = load_samples(result_dir)
        if not samples:
            continue

        # Filter for specific language if needed
        if model == "opens2s" and dataset in ("esd_en", "esd_cn"):
            if dataset == "esd_en":
                samples = [s for s in samples if "/EN/" in s.get("path", "")][:MAX_SAMPLES]
            else:
                samples = [s for s in samples if "/CN/" in s.get("path", "")][:MAX_SAMPLES]

        if not samples:
            continue

        key = f"{model}/{dataset}"
        metrics = compute_aggregate(samples)
        all_results[key] = metrics

        print(f"\n{'='*60}")
        print(f"{key} (n={metrics['n']})")
        print(f"  ACC:       clean={metrics['ACC_clean']:.3f}  adv={metrics['ACC_adv']:.3f}  dACC={metrics['dACC']:.3f}")
        print(f"  ACC_emo:   clean={metrics['ACC_emo_clean']:.3f}  adv={metrics['ACC_emo_adv']:.3f}  dACC_emo={metrics['dACC_emo']:.3f}")
        print(f"  Semantic:  rate={metrics['Sem_rate']:.3f}  sim_avg={metrics['Sem_sim_avg']:.3f}")
        print(f"  Diff:      clean={metrics['Diff_clean']:.3f}  adv={metrics['Diff_adv']:.3f}")
        print(f"  Bias(tgt): clean={metrics['Bias_clean']:.3f}  adv={metrics['Bias_adv']:.3f}")

    # ============================================================
    # Generate per-model aggregated metrics (for the paper table)
    # ============================================================
    print("\n" + "=" * 80)
    print("PER-MODEL AGGREGATED METRICS (for paper table)")
    print("=" * 80)

    models = ["voxtral", "meralion", "opens2s"]
    for m in models:
        model_results = {k: v for k, v in all_results.items() if k.startswith(m + "/")}
        if not model_results:
            continue

        # Average across datasets
        keys_to_avg = ["ACC_clean", "ACC_adv", "dACC", "ACC_emo_clean", "ACC_emo_adv",
                        "dACC_emo", "Sem_rate", "Sem_sim_avg", "Diff_clean", "Diff_adv",
                        "Bias_clean", "Bias_adv"]
        n_datasets = len(model_results)
        avg_metrics = {}
        for k in keys_to_avg:
            avg_metrics[k] = sum(v.get(k, 0) for v in model_results.values()) / n_datasets

        print(f"\n{m.upper()} (avg over {n_datasets} datasets):")
        print(f"  ACC_adv={avg_metrics['ACC_adv']*100:.1f}%  dACC={avg_metrics['dACC']*100:.1f}%  "
              f"Diff_adv={avg_metrics['Diff_adv']:.3f}  Bias_adv={avg_metrics['Bias_adv']*100:.1f}%")
        print(f"  Sem_rate={avg_metrics['Sem_rate']*100:.1f}%  Sem_sim={avg_metrics['Sem_sim_avg']:.3f}")

    # ============================================================
    # LaTeX table format
    # ============================================================
    print("\n" + "=" * 80)
    print("TABLE DATA: ACC_adv(%) | dACC(%) | Diff_adv | Bias_adv(%)")
    print("=" * 80)
    print(f"{'Key':30s} {'ACC_adv':>8s} {'dACC':>8s} {'Diff':>8s} {'Bias':>8s}")
    print("-" * 70)
    for key, m in sorted(all_results.items()):
        print(f"{key:30s} {m['ACC_adv']*100:7.1f}% {m['dACC']*100:7.1f}% "
              f"{m['Diff_adv']:7.3f} {m['Bias_adv']*100:7.1f}%")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "existing_data_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {out_dir / 'existing_data_metrics.json'}")


if __name__ == "__main__":
    main()
