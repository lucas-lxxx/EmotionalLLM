#!/usr/bin/env python3
"""
论文级图表生成脚本

读取 OPUS/results/ 中的实验数据，生成 Section 2 Observation 的所有图表。
本地运行，无需模型。

用法:
    python generate_figures.py --results_dir ../results --figures_dir ../figures
    python generate_figures.py --config ../experiments/config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml

# === 全局风格设置 ===
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
})

# 配色方案
COLORS = {
    "semantic": "#2196F3",   # blue
    "prosody": "#FF5722",    # red-orange
    "other": "#9E9E9E",      # gray
    "text": "#4CAF50",       # green
    "audio": "#FF9800",      # orange
    "entropy": "#9C27B0",    # purple
    "ci": "#E3F2FD",         # light blue fill
}

# 三阶段分界线
PHASE_BOUNDARIES = [14, 23]
PHASE_LABELS = ["Phase I\nAcoustic\nEncoding", "Phase II\nIntegration", "Phase III\nDecision\nCrystallization"]
PHASE_COLORS = ["#E3F2FD", "#FFF3E0", "#FCE4EC"]


def add_phase_shading(ax, n_layers=36, alpha=0.08):
    """添加三阶段背景色"""
    boundaries = [0] + PHASE_BOUNDARIES + [n_layers]
    for i in range(len(boundaries) - 1):
        ax.axvspan(boundaries[i], boundaries[i+1], alpha=alpha, color=PHASE_COLORS[i])
    for b in PHASE_BOUNDARIES:
        ax.axvline(x=b, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)


def fig_2_1_probe(results_dir: Path, figures_dir: Path):
    """
    Fig. 2.1: Probe 准确率 + D(layer) 曲线
    CSV columns: semantic_acc, semantic_acc_std, prosody_acc, prosody_acc_std,
                 dominance, dominance_conflict, layer, ...
    """
    probe_path = results_dir / "probe_metrics_per_layer.csv"
    if not probe_path.exists():
        print("  [Fig 2.1] Probe — probe_metrics_per_layer.csv 不存在，跳过")
        return False

    df = pd.read_csv(probe_path)
    layers = df["layer"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Probe accuracy (semantic vs prosody) ---
    add_phase_shading(ax1, n_layers=len(layers))

    ax1.plot(layers, df["semantic_acc_conflict"], color=COLORS["semantic"],
             label="Semantic Probe", marker="o", markersize=3)
    ax1.fill_between(
        layers,
        df["semantic_acc_conflict"] - df["semantic_acc_std"],
        df["semantic_acc_conflict"] + df["semantic_acc_std"],
        alpha=0.15, color=COLORS["semantic"],
    )

    ax1.plot(layers, df["prosody_acc_conflict"], color=COLORS["prosody"],
             label="Prosody Probe", marker="s", markersize=3)
    ax1.fill_between(
        layers,
        df["prosody_acc_conflict"] - df["prosody_acc_std"],
        df["prosody_acc_conflict"] + df["prosody_acc_std"],
        alpha=0.15, color=COLORS["prosody"],
    )

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probe Accuracy (conflict samples)")
    ax1.set_title("(a) Probe Accuracy")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="best")

    # --- Right: Dominance D(layer) = prosody_acc - semantic_acc ---
    add_phase_shading(ax2, n_layers=len(layers))

    dom = df["dominance_conflict"].values
    ax2.plot(layers, dom, color="#333333", linewidth=2.5)
    ax2.fill_between(layers, 0, dom,
                     where=(dom >= 0), interpolate=True,
                     alpha=0.3, color=COLORS["prosody"], label="Prosody dominant")
    ax2.fill_between(layers, 0, dom,
                     where=(dom < 0), interpolate=True,
                     alpha=0.3, color=COLORS["semantic"], label="Semantic dominant")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("D(layer) = Acc_prosody − Acc_semantic")
    ax2.set_title("(b) Dominance Curve")
    ax2.legend(loc="best")

    plt.tight_layout()
    out_path = figures_dir / "fig_2_1_probe.pdf"
    plt.savefig(out_path)
    plt.savefig(figures_dir / "fig_2_1_probe.png")
    plt.close()
    print(f"  [Fig 2.1] 保存至: {out_path}")
    return True


def fig_2_2_logit_lens(results_dir: Path, figures_dir: Path):
    """
    Fig. 2.2: Logit Lens Margin + Win-rate (subplot)
    """
    ci_path = results_dir / "logit_lens_bootstrap_ci.csv"
    if not ci_path.exists():
        print("  [Fig 2.2] Logit Lens — CI 数据不存在，跳过")
        return False

    df = pd.read_csv(ci_path)
    layers = df["layer"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Margin curve with CI ---
    add_phase_shading(ax1, n_layers=len(layers))
    ax1.plot(layers, df["margin_mean"], color=COLORS["semantic"], label="Mean Margin")
    ax1.fill_between(
        layers, df["margin_ci_lower"], df["margin_ci_upper"],
        alpha=0.2, color=COLORS["semantic"], label="95% CI"
    )
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Margin (Prosody − Semantic)")
    ax1.set_title("(a) Logit Lens Margin Curve")
    ax1.legend(loc="best")

    # --- Right: Win-rate stacked ---
    add_phase_shading(ax2, n_layers=len(layers))
    ax2.fill_between(layers, 0, df["win_semantic_mean"],
                     alpha=0.6, color=COLORS["semantic"], label="Win Semantic")
    ax2.fill_between(layers, df["win_semantic_mean"],
                     df["win_semantic_mean"] + df["win_prosody_mean"],
                     alpha=0.6, color=COLORS["prosody"], label="Win Prosody")
    ax2.fill_between(layers,
                     df["win_semantic_mean"] + df["win_prosody_mean"],
                     df["win_semantic_mean"] + df["win_prosody_mean"] + df["win_other_mean"],
                     alpha=0.4, color=COLORS["other"], label="Win Other")

    # Add CI bands for semantic win-rate
    ax2.plot(layers, df["win_semantic_ci_lower"], "--", color=COLORS["semantic"],
             alpha=0.4, linewidth=0.8)
    ax2.plot(layers, df["win_semantic_ci_upper"], "--", color=COLORS["semantic"],
             alpha=0.4, linewidth=0.8)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Win Rate")
    ax2.set_title("(b) Logit Lens Win-Rate")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="best")

    plt.tight_layout()
    out_path = figures_dir / "fig_2_2_logit_lens.pdf"
    plt.savefig(out_path)
    plt.savefig(figures_dir / "fig_2_2_logit_lens.png")
    plt.close()
    print(f"  [Fig 2.2] 保存至: {out_path}")
    return True


def fig_2_2c_entropy(results_dir: Path, figures_dir: Path):
    """
    Fig. 2.2c: Entropy trajectory (可合并入 Fig 2.2 或独立)
    """
    agg_path = results_dir / "entropy_trajectory_aggregated.csv"
    if not agg_path.exists():
        print("  [Fig 2.2c] Entropy — 数据不存在，跳过")
        return False

    df = pd.read_csv(agg_path)
    layers = df["layer"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    add_phase_shading(ax, n_layers=len(layers))

    ax.plot(layers, df["entropy_mean"], color=COLORS["entropy"], linewidth=2.5,
            label="Mean Entropy")
    ax.fill_between(
        layers, df["entropy_ci_lower"], df["entropy_ci_upper"],
        alpha=0.2, color=COLORS["entropy"], label="95% CI"
    )

    # 标注最大熵线 (uniform distribution: log2(5) ≈ 2.32)
    max_entropy = np.log2(5)
    ax.axhline(y=max_entropy, color="gray", linestyle=":", linewidth=1,
               alpha=0.5, label=f"Uniform entropy ({max_entropy:.2f})")

    # 加载结晶分析
    crystal_path = results_dir / "entropy_crystallization.json"
    if crystal_path.exists():
        with open(crystal_path) as f:
            crystal = json.load(f)
        steep_layer = crystal["steepest_drop_layer"]
        ax.axvline(x=steep_layer, color=COLORS["entropy"], linestyle="--",
                   linewidth=1.5, alpha=0.7)
        ax.annotate(
            f"Steepest drop\n(Layer {steep_layer})",
            xy=(steep_layer, df[df["layer"] == steep_layer]["entropy_mean"].values[0]),
            xytext=(steep_layer + 3, df["entropy_mean"].max() * 0.8),
            arrowprops=dict(arrowstyle="->", color=COLORS["entropy"]),
            fontsize=9, color=COLORS["entropy"],
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("5-way Decision Entropy Trajectory")
    ax.legend(loc="best")

    plt.tight_layout()
    out_path = figures_dir / "fig_2_2c_entropy.pdf"
    plt.savefig(out_path)
    plt.savefig(figures_dir / "fig_2_2c_entropy.png")
    plt.close()
    print(f"  [Fig 2.2c] 保存至: {out_path}")
    return True


def fig_2_3_patching(results_dir: Path, figures_dir: Path):
    """
    Fig. 2.3: Activation Patching Flip-to-Target + Delta Logit (semantic vs prosody)
    """
    sem_ci = results_dir / "patching_semantic_bootstrap_ci.csv"
    pro_ci = results_dir / "patching_prosody_bootstrap_ci.csv"

    if not sem_ci.exists() or not pro_ci.exists():
        print("  [Fig 2.3] Patching — CI 数据不存在，跳过")
        return False

    df_sem = pd.read_csv(sem_ci)
    df_pro = pd.read_csv(pro_ci)
    layers = df_sem["layer"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Flip-to-Target ---
    add_phase_shading(ax1, n_layers=len(layers))
    ax1.plot(layers, df_sem["flip_to_target_mean"], color=COLORS["semantic"],
             label="Semantic Patch")
    ax1.fill_between(layers, df_sem["flip_to_target_ci_lower"],
                     df_sem["flip_to_target_ci_upper"],
                     alpha=0.15, color=COLORS["semantic"])
    ax1.plot(layers, df_pro["flip_to_target_mean"], color=COLORS["prosody"],
             label="Prosody Patch")
    ax1.fill_between(layers, df_pro["flip_to_target_ci_lower"],
                     df_pro["flip_to_target_ci_upper"],
                     alpha=0.15, color=COLORS["prosody"])

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Flip-to-Target Rate")
    ax1.set_title("(a) Flip-to-Target Rate")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="best")

    # --- Right: Delta Logit ---
    add_phase_shading(ax2, n_layers=len(layers))
    ax2.plot(layers, df_sem["delta_logit_mean"], color=COLORS["semantic"],
             label="Semantic Patch")
    ax2.fill_between(layers, df_sem["delta_logit_ci_lower"],
                     df_sem["delta_logit_ci_upper"],
                     alpha=0.15, color=COLORS["semantic"])
    ax2.plot(layers, df_pro["delta_logit_mean"], color=COLORS["prosody"],
             label="Prosody Patch")
    ax2.fill_between(layers, df_pro["delta_logit_ci_lower"],
                     df_pro["delta_logit_ci_upper"],
                     alpha=0.15, color=COLORS["prosody"])

    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Delta Logit (target)")
    ax2.set_title("(b) Delta Logit Shift")
    ax2.legend(loc="best")

    plt.tight_layout()
    out_path = figures_dir / "fig_2_3_patching.pdf"
    plt.savefig(out_path)
    plt.savefig(figures_dir / "fig_2_3_patching.png")
    plt.close()
    print(f"  [Fig 2.3] 保存至: {out_path}")
    return True


def fig_2_5_cross_modal_patching(results_dir: Path, figures_dir: Path):
    """
    Fig. 2.5: Cross-modal PatchText vs PatchAudio (Flip Rate + Logit Shift)
    """
    cm_dir = results_dir / "cross_modal_patching"
    text_path = cm_dir / "cross_modal_metrics_text.json"
    audio_path = cm_dir / "cross_modal_metrics_audio.json"

    if not text_path.exists() or not audio_path.exists():
        print("  [Fig 2.5] Cross-modal Patching — 数据不存在，跳过")
        return False

    with open(text_path) as f:
        m_text = json.load(f)
    with open(audio_path) as f:
        m_audio = json.load(f)

    layers = np.array(m_text["layer_indices"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Flip Rate ---
    add_phase_shading(ax1, n_layers=len(layers))
    ax1.plot(layers, m_text["flip_to_target_rate"], color=COLORS["text"],
             label="PatchText", marker="o", markersize=3)
    ax1.plot(layers, m_audio["flip_to_target_rate"], color=COLORS["audio"],
             label="PatchAudio", marker="s", markersize=3)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Flip-to-Target Rate")
    ax1.set_title("(a) Cross-Modal Flip Rate")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="best")

    # --- Right: Logit Shift ---
    add_phase_shading(ax2, n_layers=len(layers))
    ax2.plot(layers, m_text["delta_logit_target_mean"], color=COLORS["text"],
             label="PatchText", marker="o", markersize=3)
    ax2.plot(layers, m_audio["delta_logit_target_mean"], color=COLORS["audio"],
             label="PatchAudio", marker="s", markersize=3)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Delta Logit (target)")
    ax2.set_title("(b) Cross-Modal Logit Shift")
    ax2.legend(loc="best")

    plt.tight_layout()
    out_path = figures_dir / "fig_2_5_cross_modal_patching.pdf"
    plt.savefig(out_path)
    plt.savefig(figures_dir / "fig_2_5_cross_modal_patching.png")
    plt.close()
    print(f"  [Fig 2.5] 保存至: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        results_dir = Path(config["paths"]["opus_results"])
        figures_dir = Path(config["paths"]["opus_figures"])
    else:
        results_dir = Path(args.results_dir or "OPUS/results")
        figures_dir = Path(args.figures_dir or "OPUS/figures")

    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("论文图表生成")
    print(f"数据源: {results_dir}")
    print(f"输出目标: {figures_dir}")
    print("=" * 60)

    generated = 0
    skipped = 0

    for name, func in [
        ("Fig 2.1: Probe", fig_2_1_probe),
        ("Fig 2.2: Logit Lens", fig_2_2_logit_lens),
        ("Fig 2.2c: Entropy", fig_2_2c_entropy),
        ("Fig 2.3: Patching", fig_2_3_patching),
        ("Fig 2.5: Cross-modal Patching", fig_2_5_cross_modal_patching),
    ]:
        print(f"\n--- {name} ---")
        ok = func(results_dir, figures_dir)
        if ok:
            generated += 1
        else:
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"完成: 生成 {generated} 张, 跳过 {skipped} 张")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
