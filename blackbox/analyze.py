"""Comprehensive analysis of black-box transfer attack results.

Generates all tables, figures, and statistics for §6 of the paper.

Usage:
    python analyze.py                    # Full analysis
    python analyze.py --collect          # Collect all summaries
    python analyze.py --figures          # Generate figures only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import cfg

# ── Result Collection ──

def collect_all_summaries() -> dict:
    """Collect summary.json from all surrogate × target × audio_type combinations."""
    data = {}
    for audio_type in ["adv", "clean", "noise"]:
        for surrogate_key in cfg.surrogate_groups:
            for target_key in cfg.target_list:
                summary_path = cfg.results_dir / audio_type / surrogate_key / target_key / "summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        key = f"{audio_type}/{surrogate_key}/{target_key}"
                        data[key] = summary
                    except Exception:
                        continue
    return data


def collect_all_results(audio_type: str = "adv") -> dict[str, list[dict]]:
    """Collect individual result JSONs for detailed analysis."""
    all_data = {}
    for surrogate_key in cfg.surrogate_groups:
        for target_key in cfg.target_list:
            result_dir = cfg.results_dir / audio_type / surrogate_key / target_key
            if not result_dir.exists():
                continue
            results = []
            for json_path in sorted(result_dir.rglob("*.json")):
                if json_path.name == "summary.json":
                    continue
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    if "sample_id" in data and "transfer_success" in data:
                        results.append(data)
                except Exception:
                    continue
            if results:
                all_data[f"{surrogate_key}/{target_key}"] = results
    return all_data


# ── Analysis Functions ──

def build_asr_matrix(summaries: dict) -> dict:
    """Build transfer ASR matrix (surrogate × target)."""
    matrix = {}
    for surrogate_key in cfg.surrogate_groups:
        row = {}
        for target_key in cfg.target_list:
            key = f"adv/{surrogate_key}/{target_key}"
            if key in summaries:
                row[target_key] = summaries[key].get("transfer_asr", None)
            else:
                row[target_key] = None
        matrix[surrogate_key] = row
    return matrix


def build_per_emotion_matrix(summaries: dict) -> dict:
    """Per-emotion transfer ASR for each surrogate × target."""
    matrix = {}
    for surrogate_key in cfg.surrogate_groups:
        for target_key in cfg.target_list:
            key = f"adv/{surrogate_key}/{target_key}"
            if key in summaries:
                pe = summaries[key].get("per_emotion", {})
                matrix[f"{surrogate_key}/{target_key}"] = {
                    emo: stats.get("transfer_asr", 0) for emo, stats in pe.items()
                }
    return matrix


def analyze_prediction_shift(all_results: dict[str, list[dict]]) -> dict:
    """Analyze where failed samples land (prediction distribution)."""
    shifts = {}
    for combo_key, results in all_results.items():
        dist = Counter()
        for r in results:
            label = r.get("majority_label", "")
            if label:
                dist[label] += 1
        shifts[combo_key] = dict(dist.most_common())
    return shifts


def analyze_cross_target_agreement(all_results: dict[str, list[dict]]) -> dict:
    """Find samples that succeed on multiple targets simultaneously."""
    # Group by surrogate, then by sample_id
    by_surrogate = defaultdict(lambda: defaultdict(dict))
    for combo_key, results in all_results.items():
        parts = combo_key.split("/")
        surrogate = parts[0]
        target = parts[1]
        for r in results:
            sid = r["sample_id"]
            by_surrogate[surrogate][sid][target] = r.get("transfer_success", False)

    agreement = {}
    for surrogate, samples in by_surrogate.items():
        n_targets_dist = Counter()
        for sid, target_results in samples.items():
            n_success = sum(1 for v in target_results.values() if v)
            n_targets_dist[n_success] += 1
        agreement[surrogate] = dict(sorted(n_targets_dist.items()))

    return agreement


def analyze_language_comparison(summaries: dict) -> dict:
    """Compare EN vs CN transfer rates controlling for surrogate."""
    comparisons = {}
    for surrogate_base in ["voxtral", "opens2s"]:
        en_key_prefix = f"adv/{surrogate_base}_en"
        cn_key_prefix = f"adv/{surrogate_base}_cn"
        en_asrs = {}
        cn_asrs = {}
        for target_key in cfg.target_list:
            en_k = f"{en_key_prefix}/{target_key}"
            cn_k = f"{cn_key_prefix}/{target_key}"
            if en_k in summaries:
                en_asrs[target_key] = summaries[en_k].get("transfer_asr", None)
            if cn_k in summaries:
                cn_asrs[target_key] = summaries[cn_k].get("transfer_asr", None)
        comparisons[surrogate_base] = {"EN": en_asrs, "CN": cn_asrs}
    return comparisons


def analyze_clean_vs_adv_vs_noise(summaries: dict) -> dict:
    """Three-way comparison: clean accuracy vs adversarial vs random noise."""
    comparison = {}
    for surrogate_key in cfg.surrogate_groups:
        for target_key in cfg.target_list:
            clean_k = f"clean/{surrogate_key}/{target_key}"
            adv_k = f"adv/{surrogate_key}/{target_key}"
            noise_k = f"noise/{surrogate_key}/{target_key}"

            entry = {}
            if clean_k in summaries:
                entry["clean_accuracy"] = summaries[clean_k].get("accuracy", None)
                entry["clean_target_rate"] = summaries[clean_k].get("target_rate", None)
            if adv_k in summaries:
                entry["adv_asr"] = summaries[adv_k].get("transfer_asr", None)
            if noise_k in summaries:
                entry["noise_accuracy"] = summaries[noise_k].get("accuracy", None)
                entry["noise_target_rate"] = summaries[noise_k].get("target_rate", None)

            if entry:
                comparison[f"{surrogate_key}/{target_key}"] = entry

    return comparison


# ── Output Formatting ──

def print_asr_matrix(matrix: dict):
    """Print formatted ASR matrix."""
    print("\n" + "=" * 90)
    print("Table: Transfer ASR Matrix (Surrogate × Target)")
    print("=" * 90)

    # Header
    targets = list(cfg.target_list.keys())
    target_names = [cfg.target_list[t]["name"][:15] for t in targets]
    header = f"{'Surrogate':<16}" + "".join(f"{n:>16}" for n in target_names)
    print(header)
    print("-" * len(header))

    # Rows
    for surrogate_key, row in matrix.items():
        name = cfg.surrogate_groups[surrogate_key]["name"]
        values = []
        for t in targets:
            v = row.get(t)
            if v is not None:
                values.append(f"{v:.2%}")
            else:
                values.append("—")
        print(f"{name:<16}" + "".join(f"{v:>16}" for v in values))

    # Average row
    print("-" * len(header))
    avg_values = []
    for t in targets:
        vals = [matrix[s].get(t) for s in matrix if matrix[s].get(t) is not None]
        if vals:
            avg_values.append(f"{np.mean(vals):.2%}")
        else:
            avg_values.append("—")
    print(f"{'Average':<16}" + "".join(f"{v:>16}" for v in avg_values))


def print_per_emotion(per_emotion_matrix: dict):
    """Print per-emotion transfer ASR."""
    print("\n" + "=" * 70)
    print("Per-Emotion Transfer ASR")
    print("=" * 70)

    emotions = ["angry", "sad", "neutral", "surprise"]
    header = f"{'Surrogate/Target':<30}" + "".join(f"{e:>12}" for e in emotions)
    print(header)
    print("-" * len(header))

    for combo, emo_data in sorted(per_emotion_matrix.items()):
        values = [f"{emo_data.get(e, 0):.2%}" for e in emotions]
        print(f"{combo:<30}" + "".join(f"{v:>12}" for v in values))


def print_prediction_shift(shifts: dict):
    """Print prediction shift distribution."""
    print("\n" + "=" * 70)
    print("Prediction Distribution (what API returns)")
    print("=" * 70)

    for combo, dist in sorted(shifts.items()):
        total = sum(dist.values())
        print(f"\n  {combo} (n={total}):")
        for label, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count} ({100*count/total:.1f}%)")


def print_language_comparison(comparisons: dict):
    """Print cross-language comparison."""
    print("\n" + "=" * 70)
    print("Cross-Language Comparison (EN vs CN)")
    print("=" * 70)

    for surrogate_base, data in comparisons.items():
        print(f"\n  {surrogate_base.upper()}:")
        for target_key in cfg.target_list:
            en_v = data["EN"].get(target_key)
            cn_v = data["CN"].get(target_key)
            en_s = f"{en_v:.2%}" if en_v is not None else "—"
            cn_s = f"{cn_v:.2%}" if cn_v is not None else "—"
            diff = ""
            if en_v is not None and cn_v is not None:
                d = cn_v - en_v
                diff = f" (Δ={d:+.2%})"
            target_name = cfg.target_list[target_key]["name"]
            print(f"    {target_name:<20} EN={en_s:>8}  CN={cn_s:>8}{diff}")


# ── LaTeX Output ──

def generate_latex_table(matrix: dict) -> str:
    """Generate LaTeX table for the paper."""
    targets = list(cfg.target_list.keys())
    n_cols = len(targets) + 1

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Black-box Transfer ASR (\%) across surrogate-target combinations. Each cell shows the targeted attack success rate (3-prompt majority vote, target emotion: \textit{happy}).}")
    lines.append(r"\label{tab:blackbox-transfer}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(targets) + "}")
    lines.append(r"\toprule")

    # Header
    header_cells = [r"\textbf{Surrogate}"]
    for t in targets:
        name = cfg.target_list[t]["name"]
        header_cells.append(r"\textbf{" + name + "}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for surrogate_key, row in matrix.items():
        name = cfg.surrogate_groups[surrogate_key]["name"]
        cells = [name]
        for t in targets:
            v = row.get(t)
            if v is not None:
                cells.append(f"{v*100:.1f}")
            else:
                cells.append("—")
        lines.append(" & ".join(cells) + r" \\")

    # Average
    lines.append(r"\midrule")
    avg_cells = [r"\textbf{Average}"]
    for t in targets:
        vals = [matrix[s].get(t) for s in matrix if matrix[s].get(t) is not None]
        if vals:
            avg_cells.append(f"\\textbf{{{np.mean(vals)*100:.1f}}}")
        else:
            avg_cells.append("—")
    lines.append(" & ".join(avg_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Figure Generation ──

def generate_figures(summaries: dict, all_results: dict):
    """Generate all paper figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available. Skipping figure generation.")
        return

    fig_dir = cfg.blackbox_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Also copy to finalpaper/figure/ for LaTeX
    paper_fig_dir = cfg.blackbox_root.parent / "finalpaper" / "figure"
    paper_fig_dir.mkdir(exist_ok=True)

    # Color scheme
    colors = {
        "happy": "#FF6B6B",
        "sad": "#4ECDC4",
        "angry": "#FF8C42",
        "neutral": "#95E1D3",
        "surprise": "#A78BFA",
    }
    target_colors = ["#2196F3", "#1565C0", "#FF5722", "#4CAF50", "#66BB6A", "#FFC107"]

    # ── Figure 1: Transfer ASR Heatmap ──
    matrix = build_asr_matrix(summaries)
    surrogates = [k for k in cfg.surrogate_groups if any(matrix[k][t] is not None for t in cfg.target_list)]
    targets_with_data = [t for t in cfg.target_list
                         if any(matrix[s].get(t) is not None for s in surrogates)]

    if surrogates and targets_with_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        data_arr = np.zeros((len(surrogates), len(targets_with_data)))
        for i, s in enumerate(surrogates):
            for j, t in enumerate(targets_with_data):
                v = matrix[s].get(t)
                data_arr[i, j] = v * 100 if v is not None else 0

        im = ax.imshow(data_arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=50)
        ax.set_xticks(range(len(targets_with_data)))
        ax.set_xticklabels([cfg.target_list[t]["name"] for t in targets_with_data], rotation=30, ha="right")
        ax.set_yticks(range(len(surrogates)))
        ax.set_yticklabels([cfg.surrogate_groups[s]["name"] for s in surrogates])

        for i in range(len(surrogates)):
            for j in range(len(targets_with_data)):
                v = data_arr[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                            color="white" if v > 25 else "black", fontsize=10, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Transfer ASR (%)")
        ax.set_title("Black-box Transfer ASR (%) — Surrogate × Target")
        plt.tight_layout()
        for dest in [fig_dir, paper_fig_dir]:
            plt.savefig(dest / "blackbox_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Generated: blackbox_heatmap.pdf")

    # ── Figure 2: Per-Emotion Grouped Bar Chart ──
    per_emo = build_per_emotion_matrix(summaries)
    if per_emo:
        emotions = ["angry", "sad", "neutral", "surprise"]
        combos = sorted(per_emo.keys())

        if combos:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(combos))
            width = 0.18

            for i, emo in enumerate(emotions):
                vals = [per_emo[c].get(emo, 0) * 100 for c in combos]
                ax.bar(x + i * width, vals, width, label=emo.capitalize(), color=colors.get(emo, "#999"))

            ax.set_ylabel("Transfer ASR (%)")
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels([c.replace("/", "\n→ ") for c in combos], fontsize=8)
            ax.legend()
            ax.set_title("Per-Emotion Transfer ASR")
            plt.tight_layout()
            for dest in [fig_dir, paper_fig_dir]:
                plt.savefig(dest / "blackbox_per_emotion.pdf", dpi=300, bbox_inches="tight")
            plt.close()
            print("  Generated: blackbox_per_emotion.pdf")

    # ── Figure 3: Prediction Shift Distribution (Stacked Bar) ──
    shifts = analyze_prediction_shift(all_results)
    if shifts:
        combos = sorted(shifts.keys())
        all_labels = sorted(set(l for d in shifts.values() for l in d))

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(combos))
        bottoms = np.zeros(len(combos))

        for label in all_labels:
            vals = []
            for c in combos:
                total = sum(shifts[c].values())
                vals.append(shifts[c].get(label, 0) / total * 100 if total > 0 else 0)
            vals = np.array(vals)
            ax.bar(x, vals, bottom=bottoms, label=label.capitalize(),
                   color=colors.get(label, "#999"), width=0.6)
            bottoms += vals

        ax.set_ylabel("Proportion (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("/", "\n→ ") for c in combos], fontsize=7)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title("Prediction Distribution of Black-box API Responses")
        plt.tight_layout()
        for dest in [fig_dir, paper_fig_dir]:
            plt.savefig(dest / "blackbox_pred_shift.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Generated: blackbox_pred_shift.pdf")

    # ── Figure 4: Clean vs Adv vs Noise Comparison ──
    three_way = analyze_clean_vs_adv_vs_noise(summaries)
    clean_data = {k: v for k, v in three_way.items() if "adv_asr" in v}
    if clean_data:
        combos = sorted(clean_data.keys())
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(combos))
        width = 0.25

        clean_vals = [clean_data[c].get("clean_accuracy", 0) * 100 if clean_data[c].get("clean_accuracy") is not None else 0 for c in combos]
        noise_vals = [clean_data[c].get("noise_accuracy", 0) * 100 if clean_data[c].get("noise_accuracy") is not None else 0 for c in combos]
        adv_vals = [clean_data[c].get("adv_asr", 0) * 100 if clean_data[c].get("adv_asr") is not None else 0 for c in combos]

        ax.bar(x - width, clean_vals, width, label="Clean", color="#4CAF50")
        ax.bar(x, noise_vals, width, label="Random Noise", color="#FFC107")
        ax.bar(x + width, adv_vals, width, label="Adversarial", color="#F44336")

        ax.set_ylabel("Rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("/", "\n→ ") for c in combos], fontsize=7)
        ax.legend()
        ax.set_title("Clean Accuracy vs Random-Noise Accuracy vs Adversarial ASR")
        plt.tight_layout()
        for dest in [fig_dir, paper_fig_dir]:
            plt.savefig(dest / "blackbox_three_way.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Generated: blackbox_three_way.pdf")

    # ── Figure 5: Cross-Language Comparison ──
    lang_comp = analyze_language_comparison(summaries)
    has_lang_data = any(
        any(v is not None for v in data["EN"].values()) and
        any(v is not None for v in data["CN"].values())
        for data in lang_comp.values()
    )
    if has_lang_data:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for idx, (surrogate_base, data) in enumerate(lang_comp.items()):
            ax = axes[idx]
            targets_plot = [t for t in cfg.target_list
                           if data["EN"].get(t) is not None or data["CN"].get(t) is not None]
            if not targets_plot:
                continue
            x = np.arange(len(targets_plot))
            en_vals = [data["EN"].get(t, 0) * 100 if data["EN"].get(t) is not None else 0 for t in targets_plot]
            cn_vals = [data["CN"].get(t, 0) * 100 if data["CN"].get(t) is not None else 0 for t in targets_plot]

            ax.bar(x - 0.2, en_vals, 0.35, label="English", color="#2196F3")
            ax.bar(x + 0.2, cn_vals, 0.35, label="Chinese", color="#FF5722")
            ax.set_xticks(x)
            ax.set_xticklabels([cfg.target_list[t]["name"] for t in targets_plot], rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Transfer ASR (%)")
            ax.set_title(f"{surrogate_base.capitalize()} Surrogate")
            ax.legend()

        plt.suptitle("Cross-Language Transfer Comparison (EN vs CN)")
        plt.tight_layout()
        for dest in [fig_dir, paper_fig_dir]:
            plt.savefig(dest / "blackbox_language.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Generated: blackbox_language.pdf")


# ── Main ──

def run_full_analysis():
    """Run complete analysis pipeline."""
    print("Collecting results...")
    summaries = collect_all_summaries()
    print(f"Found {len(summaries)} summary files")

    if not summaries:
        print("No results found. Run experiments first.")
        return

    # ASR Matrix
    matrix = build_asr_matrix(summaries)
    print_asr_matrix(matrix)

    # Per-emotion
    per_emo = build_per_emotion_matrix(summaries)
    if per_emo:
        print_per_emotion(per_emo)

    # Prediction shift
    all_results = collect_all_results("adv")
    if all_results:
        shifts = analyze_prediction_shift(all_results)
        print_prediction_shift(shifts)

        # Cross-target agreement
        agreement = analyze_cross_target_agreement(all_results)
        if agreement:
            print("\n" + "=" * 70)
            print("Cross-Target Agreement")
            print("=" * 70)
            for surrogate, dist in agreement.items():
                print(f"\n  {surrogate}: # targets succeeded -> # samples")
                for n_success, count in sorted(dist.items()):
                    print(f"    {n_success} targets: {count} samples")

    # Language comparison
    lang_comp = analyze_language_comparison(summaries)
    print_language_comparison(lang_comp)

    # Clean vs Adv vs Noise
    three_way = analyze_clean_vs_adv_vs_noise(summaries)
    if three_way:
        print("\n" + "=" * 70)
        print("Clean Accuracy vs Adversarial ASR vs Random-Noise Accuracy")
        print("=" * 70)
        for combo, data in sorted(three_way.items()):
            vals = ", ".join(f"{k}={v:.2%}" if v is not None else f"{k}=—" for k, v in data.items())
            print(f"  {combo}: {vals}")

    # LaTeX table
    latex = generate_latex_table(matrix)
    latex_path = cfg.blackbox_root / "table_transfer_asr.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"\nLaTeX table saved: {latex_path}")

    # Figures
    print("\nGenerating figures...")
    generate_figures(summaries, all_results)

    # Save full analysis JSON
    analysis = {
        "asr_matrix": matrix,
        "per_emotion": per_emo,
        "prediction_shifts": shifts if all_results else {},
        "language_comparison": lang_comp,
        "three_way_comparison": three_way,
    }
    analysis_path = cfg.blackbox_root / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Full analysis saved: {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Just collect summaries")
    parser.add_argument("--figures", action="store_true", help="Generate figures only")
    args = parser.parse_args()

    if args.collect:
        summaries = collect_all_summaries()
        for k, v in sorted(summaries.items()):
            print(f"  {k}: ASR={v.get('transfer_asr', 'N/A')}")
    elif args.figures:
        summaries = collect_all_summaries()
        all_results = collect_all_results("adv")
        generate_figures(summaries, all_results)
    else:
        run_full_analysis()
