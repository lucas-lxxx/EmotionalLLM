#!/usr/bin/env python3
"""
Compute FINAL table metrics for the paper.
4 columns replacing Sem/Joint/SNR/Conv:
  ACC↓   = Overall perception accuracy on adversarial audio
  ΔACC↑  = Accuracy drop (clean - adv)
  H_sem↑ = Semantic hallucination rate = 1 - P(sem_sim >= threshold)
  H_sac↑ = SA-Confusion = P(emo_flipped AND sem_preserved)

Data source: existing per-sample JSONs (no new inference needed).
"""
import json
import os
import glob
from pathlib import Path

BASE_DIR = Path("/data1/lixiang/EmotionalLLM/code")
SEMANTIC_THRESHOLD = 0.8
MAX_SAMPLES = 60

RESULT_MAP = {
    ("Voxtral", "IEMOCAP"):  BASE_DIR / "white_box_voxtral/result/Voxtral_IEMOCAP",
    ("Voxtral", "RAVDESS"):  BASE_DIR / "white_box_voxtral/result/Voxtral_RAVDESS",
    ("Voxtral", "ESD-EN"):   BASE_DIR / "white_box_voxtral/result/Voxtral_EN",
    ("Voxtral", "ESD-CN"):   BASE_DIR / "white_box_voxtral/result/Voxtral_CN",
    ("MERaLiON", "IEMOCAP"): BASE_DIR / "white_box_meralion/result/MERaLiON_IEMOCAP",
    ("MERaLiON", "RAVDESS"): BASE_DIR / "white_box_meralion/result/MERaLiON_RAVDESS",
    ("MERaLiON", "ESD-EN"):  BASE_DIR / "white_box_meralion/result/MERaLiON_EN",
    ("MERaLiON", "ESD-CN"):  BASE_DIR / "white_box_meralion/result/MERaLiON_CN",
    ("OpenS2S", "IEMOCAP"):  BASE_DIR / "white_box_opens2s_v2/result/IEMOCAP",
    ("OpenS2S", "RAVDESS"):  BASE_DIR / "white_box_opens2s_v2/result/RAVDESS",
    ("OpenS2S", "ESD-EN"):   BASE_DIR / "white_box_opens2s_v2/result/ESDfinal",
    ("OpenS2S", "ESD-CN"):   BASE_DIR / "white_box_opens2s_v2/result/ESDfinal",
}


def load_samples(result_dir, max_samples=MAX_SAMPLES, lang_filter=None):
    json_files = sorted(glob.glob(str(result_dir / "**/*.json"), recursive=True))
    json_files = [f for f in json_files
                  if not any(x in os.path.basename(f) for x in
                             ["summary", "cleaned", "judge", "report", "analyze"])]
    samples = []
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        if lang_filter and lang_filter not in d.get("path", ""):
            continue
        d.pop("loss_trace", None)
        d.pop("grad_norm_trace", None)
        samples.append(d)
        if len(samples) >= max_samples:
            break
    return samples


def compute_metrics(samples):
    """Compute the 4 AHa-Bench metrics for a set of samples."""
    n = len(samples)
    if n == 0:
        return None

    acc_clean_list = []
    acc_adv_list = []
    h_sem_list = []
    h_sac_list = []

    for s in samples:
        gt = s["ground_truth_emotion"]
        tgt = s["target_emotion"]
        emo_clean = [e for e in s.get("emo_pred_clean", []) if e]
        emo_adv = [e for e in s.get("emo_pred_adv", []) if e]
        sem_sim = s.get("semantic_sim", 0.0)
        sem_ok = 1.0 if sem_sim >= SEMANTIC_THRESHOLD else 0.0
        emo_flipped = s.get("success_emo", False)

        # ACC: (emo_correct + sem_preserved) / (n_emo_probes + 1)
        n_probes = max(len(emo_clean), len(emo_adv))
        if n_probes == 0:
            n_probes = 3  # fallback

        emo_correct_clean = sum(1 for e in emo_clean if e == gt)
        emo_correct_adv = sum(1 for e in emo_adv if e == gt)

        acc_c = (emo_correct_clean + 1.0) / (n_probes + 1)  # +1 for sem (always correct on clean)
        acc_a = (emo_correct_adv + sem_ok) / (n_probes + 1)

        acc_clean_list.append(acc_c)
        acc_adv_list.append(acc_a)

        # H_sem: semantic hallucination = 1 - sem_preserved
        h_sem_list.append(1.0 - sem_ok)

        # H_sac: SA-Confusion = emo flipped AND semantic preserved
        h_sac = 1.0 if (emo_flipped and sem_ok) else 0.0
        h_sac_list.append(h_sac)

    ACC_clean = sum(acc_clean_list) / n
    ACC_adv = sum(acc_adv_list) / n
    dACC = ACC_clean - ACC_adv
    H_sem = sum(h_sem_list) / n
    H_sac = sum(h_sac_list) / n

    return {
        "n": n,
        "ACC_adv": ACC_adv,
        "ACC_clean": ACC_clean,
        "dACC": dACC,
        "H_sem": H_sem,
        "H_sac": H_sac,
    }


def main():
    all_results = {}

    for (model, dataset), result_dir in sorted(RESULT_MAP.items()):
        if not result_dir.exists():
            continue

        lang_filter = None
        if model == "OpenS2S" and dataset == "ESD-EN":
            lang_filter = "/EN/"
        elif model == "OpenS2S" and dataset == "ESD-CN":
            lang_filter = "/CN/"

        samples = load_samples(result_dir, lang_filter=lang_filter)
        if not samples:
            print(f"[SKIP] {model}/{dataset}: no samples found")
            continue

        key = f"{model}/{dataset}"
        m = compute_metrics(samples)
        if m is None:
            continue
        all_results[key] = m

        print(f"{key:25s} (n={m['n']:3d}): ACC={m['ACC_adv']*100:5.1f}%  "
              f"dACC={m['dACC']*100:5.1f}%  H_sem={m['H_sem']*100:5.1f}%  "
              f"H_sac={m['H_sac']*100:5.1f}%")

    # ============================================================
    # Per-model aggregated (for the paper table's Attack Quality columns)
    # ============================================================
    print("\n" + "=" * 80)
    print("PAPER TABLE: Attack Quality (aggregated per model)")
    print("Columns: ACC(%)↓ | ΔACC(%)↑ | H_sem(%)↑ | H_sac(%)↑")
    print("=" * 80)

    models = ["Voxtral", "MERaLiON", "OpenS2S"]
    model_agg = {}
    for model_name in models:
        entries = {k: v for k, v in all_results.items() if k.startswith(model_name + "/")}
        if not entries:
            continue
        nd = len(entries)
        agg = {}
        for metric in ["ACC_adv", "dACC", "H_sem", "H_sac"]:
            agg[metric] = sum(v[metric] for v in entries.values()) / nd
        model_agg[model_name] = agg
        datasets_str = ", ".join(k.split("/")[1] for k in sorted(entries.keys()))
        print(f"  {model_name:12s} ({nd} datasets: {datasets_str})")
        print(f"    ACC={agg['ACC_adv']*100:5.1f}%  dACC={agg['dACC']*100:5.1f}%  "
              f"H_sem={agg['H_sem']*100:5.1f}%  H_sac={agg['H_sac']*100:5.1f}%")

    # ============================================================
    # LaTeX snippet
    # ============================================================
    print("\n" + "=" * 80)
    print("LATEX TABLE SNIPPET")
    print("=" * 80)
    print(r"% Columns: ACC(\%)$\downarrow$ & $\Delta$ACC(\%)$\uparrow$ & H$_\text{sem}$(\%)$\uparrow$ & H$_\text{sac}$(\%)$\uparrow$")
    for model_name in models:
        if model_name not in model_agg:
            continue
        a = model_agg[model_name]
        print(f"% {model_name}")
        print(f"  & {a['ACC_adv']*100:.1f} & {a['dACC']*100:.1f} & {a['H_sem']*100:.1f} & {a['H_sac']*100:.1f} \\\\")

    # ============================================================
    # Per-dataset detail (for potential per-row display)
    # ============================================================
    print("\n" + "=" * 80)
    print("PER-DATASET DETAIL")
    print("=" * 80)
    datasets = ["IEMOCAP", "RAVDESS", "ESD-EN", "ESD-CN"]
    header = f"{'Model':12s} {'Dataset':10s} {'n':>4s} {'ACC%':>6s} {'dACC%':>6s} {'H_sem%':>7s} {'H_sac%':>7s}"
    print(header)
    print("-" * len(header))
    for model_name in models:
        for ds in datasets:
            key = f"{model_name}/{ds}"
            if key not in all_results:
                print(f"{model_name:12s} {ds:10s}  {'---':>4s}  {'---':>5s}  {'---':>5s}  {'---':>6s}  {'---':>6s}")
                continue
            m = all_results[key]
            print(f"{model_name:12s} {ds:10s} {m['n']:4d} {m['ACC_adv']*100:5.1f}% {m['dACC']*100:5.1f}% "
                  f"{m['H_sem']*100:6.1f}% {m['H_sac']*100:6.1f}%")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = {"per_dataset": all_results, "per_model_agg": model_agg}
    with open(out_dir / "final_table_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {out_dir / 'final_table_metrics.json'}")


if __name__ == "__main__":
    main()
