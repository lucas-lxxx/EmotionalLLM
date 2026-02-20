"""
Visualization Module

绘制主导性曲线、准确率曲线等图表。
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_dominance_curve(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Modality Dominance Curve",
    figsize: tuple = (12, 6)
):
    """
    绘制主导性曲线 D(ℓ) = Acc_prosody - Acc_semantic

    Args:
        results_df: 包含逐层指标的DataFrame
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = results_df["layer"].values
    dominance = results_df["dominance"].values
    dominance_conflict = results_df["dominance_conflict"].values

    # 全量数据曲线
    ax.plot(layers, dominance, 'b-', linewidth=2, marker='o',
            markersize=4, label='All samples')

    # 冲突子集曲线
    if not np.all(np.isnan(dominance_conflict)):
        ax.plot(layers, dominance_conflict, 'r--', linewidth=2, marker='s',
                markersize=4, label='Conflict samples')

    # 零线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.7)

    # 填充区域
    ax.fill_between(layers, 0, dominance, where=(dominance > 0),
                    alpha=0.3, color='green', label='Prosody dominant')
    ax.fill_between(layers, 0, dominance, where=(dominance < 0),
                    alpha=0.3, color='orange', label='Semantic dominant')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Dominance D(ℓ) = Acc_prosody - Acc_semantic', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(layers.min() - 0.5, layers.max() + 0.5)

    # 添加层级分区线
    n_layers = len(layers)
    if n_layers >= 24:
        ax.axvline(x=n_layers // 3, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=2 * n_layers // 3, color='gray', linestyle=':', alpha=0.5)
        ax.text(n_layers // 6, ax.get_ylim()[1] * 0.9, 'Early', ha='center', fontsize=10, alpha=0.7)
        ax.text(n_layers // 2, ax.get_ylim()[1] * 0.9, 'Middle', ha='center', fontsize=10, alpha=0.7)
        ax.text(5 * n_layers // 6, ax.get_ylim()[1] * 0.9, 'Late', ha='center', fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved dominance curve to {output_path}")


def plot_accuracy_curves(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Probe Accuracy by Layer",
    figsize: tuple = (12, 6)
):
    """
    绘制Semantic和Prosody Probe的准确率曲线

    Args:
        results_df: 包含逐层指标的DataFrame
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = results_df["layer"].values
    semantic_acc = results_df["semantic_acc"].values
    prosody_acc = results_df["prosody_acc"].values
    semantic_std = results_df["semantic_acc_std"].values
    prosody_std = results_df["prosody_acc_std"].values

    # Semantic准确率
    ax.plot(layers, semantic_acc, 'b-', linewidth=2, marker='o',
            markersize=4, label='Semantic Probe')
    ax.fill_between(layers,
                    semantic_acc - semantic_std,
                    semantic_acc + semantic_std,
                    alpha=0.2, color='blue')

    # Prosody准确率
    ax.plot(layers, prosody_acc, 'r-', linewidth=2, marker='s',
            markersize=4, label='Prosody Probe')
    ax.fill_between(layers,
                    prosody_acc - prosody_std,
                    prosody_acc + prosody_std,
                    alpha=0.2, color='red')

    # 随机基线
    ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=1,
               alpha=0.7, label='Random baseline (5 classes)')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(layers.min() - 0.5, layers.max() + 0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy curves to {output_path}")


def plot_conflict_curves(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Probe Accuracy on Conflict Samples",
    figsize: tuple = (12, 6)
):
    """
    绘制冲突子集的准确率曲线

    Args:
        results_df: 包含逐层指标的DataFrame
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = results_df["layer"].values
    semantic_acc = results_df["semantic_acc_conflict"].values
    prosody_acc = results_df["prosody_acc_conflict"].values

    # 检查是否有有效数据
    if np.all(np.isnan(semantic_acc)):
        print("No conflict samples data available for plotting")
        plt.close()
        return

    # Semantic准确率 (冲突子集)
    ax.plot(layers, semantic_acc, 'b-', linewidth=2, marker='o',
            markersize=4, label='Semantic Probe (conflict)')

    # Prosody准确率 (冲突子集)
    ax.plot(layers, prosody_acc, 'r-', linewidth=2, marker='s',
            markersize=4, label='Prosody Probe (conflict)')

    # 全量数据对比 (虚线)
    ax.plot(layers, results_df["semantic_acc"].values, 'b--', linewidth=1,
            alpha=0.5, label='Semantic Probe (all)')
    ax.plot(layers, results_df["prosody_acc"].values, 'r--', linewidth=1,
            alpha=0.5, label='Prosody Probe (all)')

    # 随机基线
    ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=1,
               alpha=0.7, label='Random baseline')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(layers.min() - 0.5, layers.max() + 0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved conflict curves to {output_path}")


def plot_f1_curves(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Probe F1 Score by Layer",
    figsize: tuple = (12, 6)
):
    """
    绘制F1分数曲线

    Args:
        results_df: 包含逐层指标的DataFrame
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = results_df["layer"].values
    semantic_f1 = results_df["semantic_f1"].values
    prosody_f1 = results_df["prosody_f1"].values

    # Semantic F1
    ax.plot(layers, semantic_f1, 'b-', linewidth=2, marker='o',
            markersize=4, label='Semantic Probe')

    # Prosody F1
    ax.plot(layers, prosody_f1, 'r-', linewidth=2, marker='s',
            markersize=4, label='Prosody Probe')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(layers.min() - 0.5, layers.max() + 0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved F1 curves to {output_path}")


def plot_layer_heatmap(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Layer-wise Metrics Heatmap",
    figsize: tuple = (14, 8)
):
    """
    绘制逐层指标热力图

    Args:
        results_df: 包含逐层指标的DataFrame
        output_path: 输出路径
        title: 图表标题
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 选择要显示的指标
    metrics = ["semantic_acc", "prosody_acc", "dominance",
               "semantic_acc_conflict", "prosody_acc_conflict", "dominance_conflict"]

    # 准备热力图数据
    heatmap_data = results_df[["layer"] + metrics].set_index("layer")[metrics].T

    # 绘制热力图
    sns.heatmap(heatmap_data, ax=ax, cmap="RdYlBu_r", center=0,
                annot=False, fmt=".2f", linewidths=0.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def save_metrics_csv(
    results_df: pd.DataFrame,
    output_path: str
):
    """
    保存逐层指标到CSV文件

    Args:
        results_df: 包含逐层指标的DataFrame
        output_path: 输出路径
    """
    results_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved metrics to {output_path}")


def generate_all_plots(
    results_df: pd.DataFrame,
    output_dir: str,
    prefix: str = ""
):
    """
    生成所有图表

    Args:
        results_df: 包含逐层指标的DataFrame
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 主导性曲线
    plot_dominance_curve(
        results_df,
        str(output_dir / f"{prefix}dominance_curve.png")
    )

    # 准确率曲线
    plot_accuracy_curves(
        results_df,
        str(output_dir / f"{prefix}accuracy_curves.png")
    )

    # 冲突子集曲线
    plot_conflict_curves(
        results_df,
        str(output_dir / f"{prefix}conflict_curves.png")
    )

    # F1曲线
    plot_f1_curves(
        results_df,
        str(output_dir / f"{prefix}f1_curves.png")
    )

    # 热力图
    plot_layer_heatmap(
        results_df,
        str(output_dir / f"{prefix}layer_heatmap.png")
    )

    # 保存CSV
    save_metrics_csv(
        results_df,
        str(output_dir / f"{prefix}metrics_per_layer.csv")
    )

    print(f"All plots saved to {output_dir}")
