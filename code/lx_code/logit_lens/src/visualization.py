"""
可视化模块

绘制 Logit Lens 实验结果图表。
"""

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_margin_curve(
    metrics: Dict,
    output_path: str,
    title: str = "Logit Lens Margin Curve (Conflict Samples)",
):
    """
    绘制 margin 曲线

    Args:
        metrics: compute_metrics 返回的指标字典
        output_path: 输出文件路径
        title: 图表标题
    """
    n_layers = metrics['n_layers']
    mean_margins = metrics['mean_margins']
    std_margins = metrics['std_margins']

    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制 margin 曲线
    ax.plot(layers, mean_margins, 'b-', linewidth=2, label='Mean Margin')
    ax.fill_between(
        layers,
        mean_margins - std_margins,
        mean_margins + std_margins,
        alpha=0.3,
        color='blue',
        label='±1 Std'
    )

    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Margin (Prosody - Semantic)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Margin curve saved to: {output_path}")


def plot_winrate_curve(
    metrics: Dict,
    output_path: str,
    title: str = "Logit Lens Win-Rate Curve (Conflict Samples)",
):
    """
    绘制 win-rate 曲线

    Args:
        metrics: compute_metrics 返回的指标字典
        output_path: 输出文件路径
        title: 图表标题
    """
    n_layers = metrics['n_layers']
    win_semantic = metrics['win_semantic']
    win_prosody = metrics['win_prosody']
    win_other = metrics['win_other']

    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制 win-rate 曲线
    ax.plot(layers, win_semantic, 'g-', linewidth=2, label='Win Semantic', marker='o', markersize=3)
    ax.plot(layers, win_prosody, 'r-', linewidth=2, label='Win Prosody', marker='s', markersize=3)
    ax.plot(layers, win_other, 'gray', linewidth=2, label='Win Other', marker='^', markersize=3)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Win-rate curve saved to: {output_path}")
