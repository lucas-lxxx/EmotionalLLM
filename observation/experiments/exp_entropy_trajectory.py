#!/usr/bin/env python3
"""
EXP-2: 熵轨迹分析 (Entropy Trajectory)

对已有 Logit Lens 的 sample-level 5-way logit 数据计算逐层 Shannon 熵。
量化决策置信度的演化轨迹，识别"决策结晶"层。

无需模型，只需 logit_lens_metrics_sample.csv。

用法:
    python exp_entropy_trajectory.py --config config.yaml
    python exp_entropy_trajectory.py --logit_lens_csv <path> --output_dir <path>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def compute_entropy(logits: np.ndarray, temperature: float = 1.0) -> float:
    """
    计算 restricted 5-way 分布的 Shannon 熵。

    Args:
        logits: shape [n_emotions] 的 logit 向量
        temperature: softmax temperature
    Returns:
        Shannon entropy (bits)
    """
    logits = logits / temperature
    logits = logits - np.max(logits)  # 数值稳定
    probs = np.exp(logits) / np.sum(np.exp(logits))
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def compute_entropy_trajectory(df: pd.DataFrame, emotions: list,
                               temperature: float = 1.0) -> pd.DataFrame:
    """
    对所有样本逐层计算熵，返回 sample-level 和 aggregated 数据。

    Args:
        df: logit_lens_metrics_sample.csv 的 DataFrame
        emotions: 情绪标签列表
        temperature: softmax temperature
    Returns:
        DataFrame with columns: layer, sample_id, entropy
    """
    logit_cols = [f"logit_{emo}" for emo in emotions]

    # 检查列是否存在
    missing = [c for c in logit_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    records = []
    for _, row in df.iterrows():
        logits = np.array([row[c] for c in logit_cols])
        entropy = compute_entropy(logits, temperature)
        records.append({
            "sample_id": row["sample_id"],
            "layer": int(row["layer"]),
            "entropy": entropy,
        })

    return pd.DataFrame(records)


def aggregate_entropy(entropy_df: pd.DataFrame,
                      n_bootstrap: int = 1000,
                      ci_level: float = 0.95,
                      seed: int = 42) -> pd.DataFrame:
    """
    按层聚合熵值，计算 mean + bootstrap CI。
    """
    layers = sorted(entropy_df["layer"].unique())
    records = []

    for layer in layers:
        values = entropy_df[entropy_df["layer"] == layer]["entropy"].values
        n = len(values)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        # Bootstrap CI
        rng = np.random.RandomState(seed + layer)
        boot_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            indices = rng.randint(0, n, size=n)
            boot_means[i] = np.mean(values[indices])

        alpha = 1 - ci_level
        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        records.append({
            "layer": layer,
            "n_samples": n,
            "entropy_mean": mean_val,
            "entropy_std": std_val,
            "entropy_ci_lower": ci_lower,
            "entropy_ci_upper": ci_upper,
        })

    return pd.DataFrame(records)


def find_crystallization_layer(agg_df: pd.DataFrame,
                               max_entropy: float = None) -> dict:
    """
    识别决策结晶层：熵下降最陡峭的区间。

    Returns:
        dict with: steepest_drop_layer, steepest_drop_magnitude,
                   half_entropy_layer, min_entropy_layer
    """
    layers = agg_df["layer"].values
    means = agg_df["entropy_mean"].values

    if max_entropy is None:
        max_entropy = np.max(means)

    # 逐层差分
    diffs = np.diff(means)
    steepest_idx = int(np.argmin(diffs))

    # 半衰层：熵首次降至最大值的一半以下
    half_threshold = max_entropy / 2
    half_layer = None
    for i, (l, m) in enumerate(zip(layers, means)):
        if m <= half_threshold:
            half_layer = int(l)
            break

    return {
        "steepest_drop_layer": int(layers[steepest_idx + 1]),
        "steepest_drop_magnitude": float(diffs[steepest_idx]),
        "half_entropy_layer": half_layer,
        "min_entropy_layer": int(layers[np.argmin(means)]),
        "min_entropy_value": float(np.min(means)),
        "max_entropy_value": float(max_entropy),
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-2: Entropy Trajectory")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--logit_lens_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        logit_lens_csv = Path(config["paths"]["logit_lens_outputs"]) / "logit_lens_metrics_sample.csv"
        output_dir = Path(config["paths"]["opus_results"])
        temperature = config["entropy"]["temperature"]
        emotions = config["data"]["emotions"]
        n_bootstrap = config["bootstrap"]["n_bootstrap"]
        ci_level = config["bootstrap"]["ci_level"]
        seed = config["bootstrap"]["seed"]
    else:
        logit_lens_csv = Path(args.logit_lens_csv)
        output_dir = Path(args.output_dir or ".")
        temperature = args.temperature or 1.0
        emotions = ["neutral", "happy", "sad", "angry", "surprised"]
        n_bootstrap = 1000
        ci_level = 0.95
        seed = 42

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP-2: 熵轨迹分析 (Entropy Trajectory)")
    print("=" * 60)

    # 加载数据
    print(f"\n[1] 加载 Logit Lens 数据: {logit_lens_csv}")
    df = pd.read_csv(logit_lens_csv)
    n_samples = df["sample_id"].nunique()
    n_layers = df["layer"].nunique()
    print(f"    样本数: {n_samples}, 层数: {n_layers}")

    # 计算逐样本逐层熵
    print(f"\n[2] 计算 Shannon 熵 (temperature={temperature})...")
    entropy_df = compute_entropy_trajectory(df, emotions, temperature)

    # 保存 sample-level
    sample_path = output_dir / "entropy_trajectory_sample.csv"
    entropy_df.to_csv(sample_path, index=False)
    print(f"    Sample-level 保存至: {sample_path}")

    # 聚合 + Bootstrap CI
    print(f"\n[3] 聚合 + Bootstrap CI (n={n_bootstrap})...")
    agg_df = aggregate_entropy(entropy_df, n_bootstrap, ci_level, seed)

    agg_path = output_dir / "entropy_trajectory_aggregated.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"    Aggregated 保存至: {agg_path}")

    # 识别决策结晶层
    print("\n[4] 识别决策结晶层...")
    crystal = find_crystallization_layer(agg_df)
    print(f"    最大熵: {crystal['max_entropy_value']:.3f} bits")
    print(f"    最小熵: {crystal['min_entropy_value']:.3f} bits (Layer {crystal['min_entropy_layer']})")
    print(f"    最陡下降层: Layer {crystal['steepest_drop_layer']} "
          f"(Δ = {crystal['steepest_drop_magnitude']:.3f})")
    if crystal["half_entropy_layer"] is not None:
        print(f"    半衰层: Layer {crystal['half_entropy_layer']}")

    # 保存结晶点分析
    import json
    crystal_path = output_dir / "entropy_crystallization.json"
    with open(crystal_path, "w") as f:
        json.dump(crystal, f, indent=2)
    print(f"    结晶分析保存至: {crystal_path}")

    # 打印摘要表
    print("\n[5] 熵轨迹摘要:")
    print(f"    {'Layer':>5} | {'Entropy Mean':>12} | {'95% CI':>20}")
    print(f"    {'-'*5}-+-{'-'*12}-+-{'-'*20}")
    for _, row in agg_df.iterrows():
        print(f"    {int(row['layer']):5d} | {row['entropy_mean']:12.4f} | "
              f"[{row['entropy_ci_lower']:.4f}, {row['entropy_ci_upper']:.4f}]")

    print("\n" + "=" * 60)
    print("EXP-2 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
