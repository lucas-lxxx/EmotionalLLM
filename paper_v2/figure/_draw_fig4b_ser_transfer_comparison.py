import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


methods = ["STAA-Net", "VTLN", "McAdams", "MSS", "Ren et al."]
original_ser = np.array([79.24, 88.86, 90.34, 86.42, 68.60])
transfer_allm = np.array([2.50, 5.71, 4.19, 9.33, 2.86])


x = np.arange(len(methods))
width = 0.34

fig, ax = plt.subplots(figsize=(8.4, 5.2))

bars_original = ax.bar(
    x - width / 2,
    original_ser,
    width,
    label="Original SER",
    color="#263b5e",
    hatch="/",
    edgecolor="white",
    linewidth=1.2,
    zorder=3,
)
bars_transfer = ax.bar(
    x + width / 2,
    transfer_allm,
    width,
    label="Transfer to ALLM",
    color="#86a9c1",
    hatch="x",
    edgecolor="white",
    linewidth=1.2,
    zorder=3,
)


def annotate_bars(bars, y_offset):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#1f2933",
        )


annotate_bars(bars_original, 1.2)
annotate_bars(bars_transfer, 1.0)

ax.set_ylabel("Success / Change Rate (%)", fontsize=22, fontweight="bold")
ax.set_xlabel("SER Attack Method", fontsize=22, fontweight="bold", labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=17, fontweight="bold")
ax.tick_params(axis="y", labelsize=17, width=1.2)
ax.tick_params(axis="x", width=1.2)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))

ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)
ax.grid(False, axis="x")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(1.2)
    ax.spines[spine].set_color("#303842")

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=2,
    fontsize=16,
    framealpha=0.88,
    edgecolor="#d6dde5",
    columnspacing=1.2,
    handletextpad=0.45,
    handlelength=1.6,
)

fig.tight_layout(pad=0.6)
fig.savefig("fig4b_ser_transfer_comparison.pdf", bbox_inches="tight")
fig.savefig("_preview_fig4b_ser_transfer_comparison.png", dpi=300, bbox_inches="tight")
