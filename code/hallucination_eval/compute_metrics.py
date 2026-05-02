#!/usr/bin/env python3
"""
Compute AHa-Bench metrics from hallucination evaluation results.

Metrics:
  ACC   = fraction of probes answered correctly (parsed == gt)
  Bias  = |P(Yes) - P(No)| across all probes (0=balanced, 1=fully biased)
  Diff  = fraction of probe-pairs where pos and neg answers are inconsistent
  ΔACC  = ACC_clean - ACC_adv (higher = more attack-induced hallucination)

Usage:
  python compute_metrics.py                    # Process all result files
  python compute_metrics.py --results_dir ./results  # Specify dir
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


def compute_sample_metrics(sample: dict, audio_type: str) -> dict:
    """Compute per-sample metrics for clean or adversarial responses."""
    key = f"{audio_type}_responses"
    responses = sample.get(key, [])
    if not responses:
        return {}

    total = len(responses)
    correct = sum(1 for r in responses if r["parsed"] == r["gt"])
    yes_count = sum(1 for r in responses if r["parsed"] == "Yes")
    no_count = sum(1 for r in responses if r["parsed"] == "No")
    unknown_count = sum(1 for r in responses if r["parsed"] == "Unknown")

    acc = correct / total if total > 0 else 0.0

    # Bias: |P(Yes) - P(No)| among parsed responses (excluding Unknown)
    parsed_total = yes_count + no_count
    if parsed_total > 0:
        bias = abs(yes_count - no_count) / parsed_total
    else:
        bias = 0.0

    # Per-type accuracy
    type_acc = {}
    for htype in ["acoustic", "semantic", "sa_confusion"]:
        typed = [r for r in responses if r["type"] == htype]
        if typed:
            type_acc[htype] = sum(1 for r in typed if r["parsed"] == r["gt"]) / len(typed)

    # Diff: consistency between positive and negative probe pairs
    # For each type, check if the pos/neg pair gives consistent answers
    pair_inconsistent = 0
    pair_total = 0
    for htype in ["acoustic", "semantic", "sa_confusion"]:
        pos = [r for r in responses if r["type"] == htype and r["id"].endswith("_pos")]
        neg = [r for r in responses if r["type"] == htype and r["id"].endswith("_neg")]
        if pos and neg:
            pair_total += 1
            # Consistent = pos says Yes AND neg says No (both correct)
            # Inconsistent = any deviation
            pos_correct = pos[0]["parsed"] == pos[0]["gt"]
            neg_correct = neg[0]["parsed"] == neg[0]["gt"]
            if not (pos_correct and neg_correct):
                pair_inconsistent += 1

    diff = pair_inconsistent / pair_total if pair_total > 0 else 0.0

    return {
        "acc": acc,
        "bias": bias,
        "diff": diff,
        "correct": correct,
        "total": total,
        "yes_count": yes_count,
        "no_count": no_count,
        "unknown_count": unknown_count,
        "type_acc": type_acc,
    }


def compute_aggregate_metrics(eval_data: dict) -> dict:
    """Compute aggregate metrics for a model-dataset evaluation."""
    samples = eval_data.get("samples", [])
    if not samples:
        return {}

    clean_metrics = [compute_sample_metrics(s, "clean") for s in samples]
    adv_metrics = [compute_sample_metrics(s, "adv") for s in samples]

    # Filter out empty
    clean_metrics = [m for m in clean_metrics if m]
    adv_metrics = [m for m in adv_metrics if m]

    def avg(metrics_list, key):
        vals = [m[key] for m in metrics_list if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_type_acc(metrics_list, htype):
        vals = [m["type_acc"].get(htype, 0.0) for m in metrics_list if "type_acc" in m]
        return sum(vals) / len(vals) if vals else 0.0

    result = {
        "num_samples": len(samples),
        "clean": {
            "ACC": avg(clean_metrics, "acc"),
            "Bias": avg(clean_metrics, "bias"),
            "Diff": avg(clean_metrics, "diff"),
            "ACC_acoustic": avg_type_acc(clean_metrics, "acoustic"),
            "ACC_semantic": avg_type_acc(clean_metrics, "semantic"),
            "ACC_sa_confusion": avg_type_acc(clean_metrics, "sa_confusion"),
        },
        "adv": {
            "ACC": avg(adv_metrics, "acc"),
            "Bias": avg(adv_metrics, "bias"),
            "Diff": avg(adv_metrics, "diff"),
            "ACC_acoustic": avg_type_acc(adv_metrics, "acoustic"),
            "ACC_semantic": avg_type_acc(adv_metrics, "semantic"),
            "ACC_sa_confusion": avg_type_acc(adv_metrics, "sa_confusion"),
        },
    }

    # ΔACC = ACC_clean - ACC_adv
    result["delta_ACC"] = result["clean"]["ACC"] - result["adv"]["ACC"]
    result["delta_ACC_acoustic"] = result["clean"]["ACC_acoustic"] - result["adv"]["ACC_acoustic"]
    result["delta_ACC_semantic"] = result["clean"]["ACC_semantic"] - result["adv"]["ACC_semantic"]
    result["delta_ACC_sa_confusion"] = result["clean"]["ACC_sa_confusion"] - result["adv"]["ACC_sa_confusion"]

    return result


def process_all(results_dir: Path):
    """Process all hallucination_eval.json files and produce summary."""
    all_results = {}

    for subdir in sorted(results_dir.iterdir()):
        eval_file = subdir / "hallucination_eval.json"
        if not eval_file.exists():
            continue

        with open(eval_file) as f:
            data = json.load(f)

        model = data["model"]
        dataset = data["dataset"]
        key = f"{model}/{dataset}"

        metrics = compute_aggregate_metrics(data)
        all_results[key] = metrics
        print(f"\n{'='*60}")
        print(f"{key} ({metrics['num_samples']} samples)")
        print(f"{'='*60}")
        print(f"  Clean  ACC={metrics['clean']['ACC']:.3f}  Bias={metrics['clean']['Bias']:.3f}  Diff={metrics['clean']['Diff']:.3f}")
        print(f"  Adv    ACC={metrics['adv']['ACC']:.3f}  Bias={metrics['adv']['Bias']:.3f}  Diff={metrics['adv']['Diff']:.3f}")
        print(f"  ΔACC={metrics['delta_ACC']:.3f}")
        print(f"  Per-type ΔACC: acoustic={metrics['delta_ACC_acoustic']:.3f}, semantic={metrics['delta_ACC_semantic']:.3f}, sa_conf={metrics['delta_ACC_sa_confusion']:.3f}")

    # Save summary
    summary_file = results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {summary_file}")

    # Generate LaTeX-friendly table data
    print("\n" + "=" * 80)
    print("TABLE DATA (for LaTeX)")
    print("=" * 80)
    print(f"{'Model/Dataset':25s} {'ACC_adv':>8s} {'ΔACC':>8s} {'Bias':>8s} {'Diff':>8s}")
    print("-" * 60)
    for key, m in sorted(all_results.items()):
        print(f"{key:25s} {m['adv']['ACC']*100:7.1f}% {m['delta_ACC']*100:7.1f}% {m['adv']['Bias']:7.3f} {m['adv']['Diff']:7.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    process_all(Path(args.results_dir))
