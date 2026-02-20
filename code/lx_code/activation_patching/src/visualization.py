"""Plotting utilities for activation patching."""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def plot_flip_rate_curve(metrics: Dict, output_path: str, title: str) -> None:
    layers = np.array(metrics["layer_indices"])
    flip_to_target = np.array(metrics["flip_to_target_rate"])
    flip_from_base = np.array(metrics["flip_from_base_rate"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, flip_to_target, "r-", linewidth=2, marker="o", markersize=3, label="Flip to Target")
    ax.plot(layers, flip_from_base, "b-", linewidth=2, marker="s", markersize=3, label="Flip from Base")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Flip Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_delta_logit_curve(metrics: Dict, output_path: str, title: str) -> None:
    layers = np.array(metrics["layer_indices"])
    delta = np.array(metrics["delta_logit_target_mean"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, delta, "g-", linewidth=2, marker="o", markersize=3, label="Delta Logit (Target)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Delta Logit", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
