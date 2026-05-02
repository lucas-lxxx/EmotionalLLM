from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import Normalize


matplotlib.use("Agg")

OUT_DIR = Path(__file__).resolve().parent

DEEP_BLUE = "#263b5e"
MID_BLUE = "#0073bd"
SOFT_BLUE = "#86a9c1"
PALE_BLUE = "#a6d9f5"
INK = "#1f2933"
GRID = "#d8e0e8"
PENDING = "#e8edf2"
ACCENT = "#d84315"
WARM = "#ff8a65"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def polish_axes(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("#303842")


def draw_coverage_map() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    stages = [
        {
            "num": "1",
            "title": "White-box",
            "body": "Targeted emotion flipping\non open ALLMs",
            "metric": "5 models  |  4 datasets",
            "chip": "controlled access",
            "xy": (0.04, 0.47),
            "color": DEEP_BLUE,
            "light": "#dce7f3",
        },
        {
            "num": "2",
            "title": "Transfer",
            "body": "Replay adversarial audio\nacross different ALLMs",
            "metric": "5 x 5 matrix  |  240/source",
            "chip": "surrogate ranking",
            "xy": (0.285, 0.47),
            "color": MID_BLUE,
            "light": "#d7ebf8",
        },
        {
            "num": "3",
            "title": "Closed APIs",
            "body": "Evaluate transfer on\ncommercial audio APIs",
            "metric": "4 pools  |  4 APIs",
            "chip": "Grok reserved",
            "xy": (0.53, 0.47),
            "color": SOFT_BLUE,
            "light": "#e4edf4",
        },
        {
            "num": "4",
            "title": "Hosted agents",
            "body": "Test voice-facing agents\nafter physical playback",
            "metric": "air + mic + frontend",
            "chip": "deployment path",
            "xy": (0.775, 0.47),
            "color": PALE_BLUE,
            "light": "#e7f4fb",
        },
    ]

    ax.plot([0.06, 0.94], [0.45, 0.45], color=GRID, lw=5, zorder=0, solid_capstyle="round")

    box_w = 0.19
    box_h = 0.32
    for idx, stage in enumerate(stages):
        x, y = stage["xy"]
        shadow = patches.FancyBboxPatch(
            (x + 0.008, y - 0.012),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0,
            facecolor="#c8d4df",
            alpha=0.22,
            zorder=1,
        )
        ax.add_patch(shadow)
        box = patches.FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#ffffff",
            facecolor=stage["light"],
            zorder=2,
        )
        ax.add_patch(box)

        stripe = patches.FancyBboxPatch(
            (x, y + box_h - 0.105),
            box_w,
            0.105,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0,
            facecolor=stage["color"],
            zorder=3,
        )
        ax.add_patch(stripe)

        badge = patches.Circle((x + 0.03, y + box_h - 0.052), 0.026, facecolor="white", edgecolor="white", zorder=4)
        ax.add_patch(badge)
        ax.text(
            x + 0.03,
            y + box_h - 0.052,
            stage["num"],
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=stage["color"],
            zorder=5,
        )
        ax.text(
            x + 0.062,
            y + box_h - 0.052,
            stage["title"],
            ha="left",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="white" if idx < 3 else INK,
            zorder=5,
        )
        ax.text(
            x + box_w / 2,
            y + 0.165,
            stage["body"],
            ha="center",
            va="center",
            fontsize=13.2,
            fontweight="bold",
            linespacing=1.28,
            color=INK,
            zorder=4,
        )
        ax.text(
            x + box_w / 2,
            y + 0.073,
            stage["metric"],
            ha="center",
            va="center",
            fontsize=11.4,
            fontweight="bold",
            color=stage["color"] if idx < 3 else DEEP_BLUE,
            zorder=4,
        )
        ax.text(
            x + box_w / 2,
            y - 0.07,
            stage["chip"],
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=INK,
            bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.25", alpha=0.98),
            zorder=5,
        )

        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + 0.056, 0.625),
                xytext=(x + box_w + 0.018, 0.625),
                arrowprops=dict(arrowstyle="-|>", lw=2.3, color=INK, shrinkA=0, shrinkB=0),
                zorder=5,
            )

    ax.text(
        0.5,
        0.92,
        "Experiment coverage map",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.845,
        "From laboratory control to deployment-facing transfer",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#526170",
    )
    ax.text(
        0.5,
        0.145,
        "The section moves from controlled white-box feasibility to transfer, closed APIs, and over-the-air hosted-agent deployment.",
        ha="center",
        va="center",
        fontsize=13.6,
        fontweight="bold",
    )

    save_figure(fig, "fig_exp_coverage_map")


def draw_per_emotion_asr() -> None:
    emotions = ["Angry", "Neutral", "Sad", "Surprise"]
    values = {
        "Voxtral": [92.20, 92.00, 96.00, 95.00],
        "OpenS2S": [71.21, 81.10, 81.54, 80.58],
        "MERaLiON": [100.00, 100.00, 100.00, 100.00],
    }
    colors = [DEEP_BLUE, MID_BLUE, SOFT_BLUE]
    hatches = ["/", "+", "x"]

    x = np.arange(len(emotions))
    width = 0.22
    offsets = np.linspace(-1, 1, len(values)) * width

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    for idx, (model, data) in enumerate(values.items()):
        data_arr = np.array(data, dtype=float)
        bars = ax.bar(
            x + offsets[idx],
            data_arr,
            width,
            label=model,
            color=colors[idx],
            hatch=hatches[idx],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.0,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_ylabel("Targeted ASR (%)", fontsize=20, fontweight="bold")
    ax.set_xlabel("Source emotion", fontsize=20, fontweight="bold", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=16, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="y", labelsize=15, width=1.2)
    ax.tick_params(axis="x", width=1.2)
    polish_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        fontsize=13,
        framealpha=0.90,
        edgecolor=GRID,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.4,
    )

    fig.tight_layout(pad=0.8)
    save_figure(fig, "fig_exp_per_emotion_asr")


def draw_blackbox_heatmap() -> None:
    rows = ["Voxtral EN", "Voxtral CN", "OpenS2S EN", "OpenS2S CN", "SALMONN", "Kimi-Audio", "Average"]
    cols = ["Gemini\nFlash", "Gemini\nPro", "Qwen3\nFlash", "Qwen\nTurbo", "Grok\nVoice", "Avg."]
    data = np.array(
        [
            [7.88, 12.47, 19.69, 33.15, np.nan, 18.30],
            [9.36, 0.94, 30.98, 34.20, np.nan, 18.87],
            [5.93, 0.00, 14.09, 22.78, np.nan, 10.70],
            [10.72, 0.00, 32.30, 37.34, np.nan, 20.09],
            [13.56, np.nan, np.nan, np.nan, 23.54, np.nan],
            [11.77, np.nan, np.nan, np.nan, 16.88, np.nan],
            [8.47, 3.35, 24.27, 31.87, np.nan, 16.99],
        ]
    )
    masked = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(PENDING)
    im = ax.imshow(masked, cmap=cmap, norm=Normalize(vmin=0, vmax=40), aspect="auto", zorder=1)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, fontsize=13, fontweight="bold")
    ax.set_yticklabels(rows, fontsize=13, fontweight="bold")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                color = "white" if data[i, j] >= 24 else INK
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
            else:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=PENDING,
                    edgecolor="white",
                    hatch="//",
                    linewidth=0.8,
                    zorder=2,
                )
                ax.add_patch(rect)
                ax.text(j, i, "pending", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#596878")

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(5.5, color="white", linewidth=3)
    ax.axvline(3.5, color="white", linewidth=3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label("Transfer ASR (%)", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=12, width=1.2)

    ax.set_title("Black-box transfer ASR by surrogate and target", fontsize=21, fontweight="bold", pad=48)
    fig.tight_layout(pad=0.8)
    save_figure(fig, "fig_exp_blackbox_heatmap")


def draw_platform_agents() -> None:
    entries = [
        ("Tencent Yuanbao", 32.57, DEEP_BLUE, "/"),
        ("Doubao", 22.17, MID_BLUE, "+"),
        ("Grok", 13.45, SOFT_BLUE, "x"),
        ("ChatGPT", 10.82, PALE_BLUE, "\\"),
        ("Gemini", 8.67, "#d8e3ec", "."),
    ]
    avg = 15.31
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(entries))
    bars = ax.barh(
        y,
        vals,
        height=0.62,
        color=[e[2] for e in entries],
        hatch=[e[3] for e in entries],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    for bar, value in zip(bars, vals):
        ax.text(
            value + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            ha="left",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

    ax.axvline(avg, color=ACCENT, linewidth=2.2, linestyle="--", zorder=4)
    ax.text(
        avg + 0.8,
        len(entries) - 0.34,
        f"Average {avg:.2f}%",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=ACCENT,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=16, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Transfer ASR under physical playback (%)", fontsize=18, fontweight="bold", labelpad=8)
    ax.set_xlim(0, 38)
    ax.set_xticks(np.arange(0, 41, 10))
    ax.tick_params(axis="x", labelsize=14, width=1.2)
    ax.tick_params(axis="y", width=1.2)
    polish_axes(ax, axis="x")

    fig.tight_layout(pad=0.8)
    save_figure(fig, "fig_exp_platform_agents")


def draw_review_preview() -> None:
    stems = [
        ("Per-emotion ASR", "fig_exp_per_emotion_asr.png"),
        ("Black-box heatmap", "fig_exp_blackbox_heatmap.png"),
        ("Platform agents", "fig_exp_platform_agents.png"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, (title, filename) in zip(axes, stems):
        image = plt.imread(OUT_DIR / filename)
        ax.imshow(image)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
        ax.set_axis_off()
    fig.tight_layout(pad=1.0)
    fig.savefig(OUT_DIR / "fig_exp_review_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    draw_per_emotion_asr()
    draw_blackbox_heatmap()
    draw_platform_agents()
    draw_review_preview()
    print("Generated experiment figure drafts in", OUT_DIR)


if __name__ == "__main__":
    main()
