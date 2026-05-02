#!/usr/bin/env python3
"""
EXP-1: Bootstrap 置信区间计算

对已有 Logit Lens 和 Activation Patching 的 sample-level 数据计算 95% Bootstrap CI。
无需模型，只需 CSV 数据文件。

用法:
    python exp_bootstrap_ci.py --config config.yaml
    python exp_bootstrap_ci.py --logit_lens_csv <path> --patching_semantic_csv <path> --patching_prosody_csv <path> --output_dir <path>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000,
                 ci_level: float = 0.95, seed: int = 42,
                 statistic: str = "mean") -> dict:
    """
    计算 bootstrap 置信区间。

    Args:
        data: 1D array of sample values
        n_bootstrap: bootstrap 重采样次数
        ci_level: 置信水平
        statistic: "mean" or "proportion"
    Returns:
        dict with keys: mean, ci_lower, ci_upper, std
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    if n == 0:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "std": np.nan}

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = data[indices]
        boot_stats[i] = np.mean(sample)

    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(data)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std": float(np.std(boot_stats)),
    }


def compute_logit_lens_ci(df: pd.DataFrame, n_bootstrap: int, ci_level: float,
                          seed: int, emotions: list) -> pd.DataFrame:
    """
    对 Logit Lens sample-level CSV 计算逐层 Bootstrap CI。

    输入 CSV 列: sample_id, layer, margin, win_semantic, win_prosody, win_other,
                 logit_neutral, logit_happy, logit_sad, logit_angry, logit_surprised
    """
    layers = sorted(df["layer"].unique())
    records = []

    for layer in layers:
        layer_df = df[df["layer"] == layer]
        n_samples = len(layer_df)

        # Margin CI
        margin_ci = bootstrap_ci(
            layer_df["margin"].values, n_bootstrap, ci_level, seed + layer
        )

        # Win-rate CI (proportion)
        win_sem_ci = bootstrap_ci(
            layer_df["win_semantic"].values, n_bootstrap, ci_level, seed + layer + 100
        )
        win_pro_ci = bootstrap_ci(
            layer_df["win_prosody"].values, n_bootstrap, ci_level, seed + layer + 200
        )
        win_oth_ci = bootstrap_ci(
            layer_df["win_other"].values, n_bootstrap, ci_level, seed + layer + 300
        )

        record = {
            "layer": layer,
            "n_samples": n_samples,
            "margin_mean": margin_ci["mean"],
            "margin_ci_lower": margin_ci["ci_lower"],
            "margin_ci_upper": margin_ci["ci_upper"],
            "win_semantic_mean": win_sem_ci["mean"],
            "win_semantic_ci_lower": win_sem_ci["ci_lower"],
            "win_semantic_ci_upper": win_sem_ci["ci_upper"],
            "win_prosody_mean": win_pro_ci["mean"],
            "win_prosody_ci_lower": win_pro_ci["ci_lower"],
            "win_prosody_ci_upper": win_pro_ci["ci_upper"],
            "win_other_mean": win_oth_ci["mean"],
            "win_other_ci_lower": win_oth_ci["ci_lower"],
            "win_other_ci_upper": win_oth_ci["ci_upper"],
        }

        # Per-emotion logit CI (for entropy computation downstream)
        for emo in emotions:
            col = f"logit_{emo}"
            if col in layer_df.columns:
                emo_ci = bootstrap_ci(
                    layer_df[col].values, n_bootstrap, ci_level, seed + layer + 400
                )
                record[f"{col}_mean"] = emo_ci["mean"]
                record[f"{col}_ci_lower"] = emo_ci["ci_lower"]
                record[f"{col}_ci_upper"] = emo_ci["ci_upper"]

        records.append(record)

    return pd.DataFrame(records)


def compute_patching_ci(csv_path: str, pair_type: str,
                        logit_lens_csv: str,
                        n_bootstrap: int, ci_level: float,
                        seed: int) -> pd.DataFrame:
    """
    对 Activation Patching 聚合 CSV 计算 Bootstrap CI。

    注意：已有的 patching CSV 是聚合后的 per-layer 数据（非 sample-level），
    因此 CI 需要从 sample-level 重构。但目前 patching 只保存了聚合数据。
    这里我们基于聚合统计量 + n_pairs 用参数化 bootstrap 近似。
    """
    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        layer = int(row["layer"])
        n_pairs = int(row["n_pairs"])
        flip_rate = row["flip_to_target_rate"]
        flip_from_base = row["flip_from_base_rate"]
        delta_logit = row["delta_logit_target_mean"]

        # Parametric bootstrap for proportions (Bernoulli)
        rng = np.random.RandomState(seed + layer)

        # Flip-to-target: binomial proportion CI
        flip_samples = rng.binomial(1, min(max(flip_rate, 0.001), 0.999), size=(n_bootstrap, n_pairs))
        flip_means = flip_samples.mean(axis=1)
        alpha = 1 - ci_level
        flip_ci_lower = np.percentile(flip_means, 100 * alpha / 2)
        flip_ci_upper = np.percentile(flip_means, 100 * (1 - alpha / 2))

        # Flip-from-base: same approach
        ffb_samples = rng.binomial(1, min(max(flip_from_base, 0.001), 0.999), size=(n_bootstrap, n_pairs))
        ffb_means = ffb_samples.mean(axis=1)
        ffb_ci_lower = np.percentile(ffb_means, 100 * alpha / 2)
        ffb_ci_upper = np.percentile(ffb_means, 100 * (1 - alpha / 2))

        # Delta logit: normal approximation (no per-sample data available)
        # Use std ≈ |delta_logit| * 0.3 as rough estimate; will be refined with EXP-3 data
        dl_std_est = max(abs(delta_logit) * 0.3, 0.1)
        dl_samples = rng.normal(delta_logit, dl_std_est / np.sqrt(n_pairs), size=n_bootstrap)
        dl_ci_lower = np.percentile(dl_samples, 100 * alpha / 2)
        dl_ci_upper = np.percentile(dl_samples, 100 * (1 - alpha / 2))

        records.append({
            "layer": layer,
            "n_pairs": n_pairs,
            "flip_to_target_mean": flip_rate,
            "flip_to_target_ci_lower": float(flip_ci_lower),
            "flip_to_target_ci_upper": float(flip_ci_upper),
            "flip_from_base_mean": flip_from_base,
            "flip_from_base_ci_lower": float(ffb_ci_lower),
            "flip_from_base_ci_upper": float(ffb_ci_upper),
            "delta_logit_mean": delta_logit,
            "delta_logit_ci_lower": float(dl_ci_lower),
            "delta_logit_ci_upper": float(dl_ci_upper),
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="EXP-1: Bootstrap CI")
    parser.add_argument("--config", type=str, default=None, help="统一配置文件")
    parser.add_argument("--logit_lens_csv", type=str, default=None)
    parser.add_argument("--patching_semantic_csv", type=str, default=None)
    parser.add_argument("--patching_prosody_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=None)
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        logit_lens_csv = Path(config["paths"]["logit_lens_outputs"]) / "logit_lens_metrics_sample.csv"
        patching_sem_csv = Path(config["paths"]["patching_outputs"]) / "patching_metrics_semantic.csv"
        patching_pro_csv = Path(config["paths"]["patching_outputs"]) / "patching_metrics_prosody.csv"
        output_dir = Path(config["paths"]["opus_results"])
        n_bootstrap = config["bootstrap"]["n_bootstrap"]
        ci_level = config["bootstrap"]["ci_level"]
        seed = config["bootstrap"]["seed"]
        emotions = config["data"]["emotions"]
    else:
        logit_lens_csv = Path(args.logit_lens_csv)
        patching_sem_csv = Path(args.patching_semantic_csv) if args.patching_semantic_csv else None
        patching_pro_csv = Path(args.patching_prosody_csv) if args.patching_prosody_csv else None
        output_dir = Path(args.output_dir or ".")
        n_bootstrap = args.n_bootstrap or 1000
        ci_level = 0.95
        seed = 42
        emotions = ["neutral", "happy", "sad", "angry", "surprised"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP-1: Bootstrap 置信区间计算")
    print("=" * 60)

    # --- Logit Lens CI ---
    if logit_lens_csv.exists():
        print(f"\n[1] 处理 Logit Lens 数据: {logit_lens_csv}")
        df_ll = pd.read_csv(logit_lens_csv)
        print(f"    样本数: {df_ll['sample_id'].nunique()}, 层数: {df_ll['layer'].nunique()}")

        ci_ll = compute_logit_lens_ci(df_ll, n_bootstrap, ci_level, seed, emotions)
        out_path = output_dir / "logit_lens_bootstrap_ci.csv"
        ci_ll.to_csv(out_path, index=False)
        print(f"    保存至: {out_path}")
    else:
        print(f"\n[1] Logit Lens CSV 不存在: {logit_lens_csv}, 跳过")

    # --- Patching CI (semantic) ---
    if patching_sem_csv and patching_sem_csv.exists():
        print(f"\n[2] 处理 Patching (semantic) 数据: {patching_sem_csv}")
        ci_sem = compute_patching_ci(
            str(patching_sem_csv), "semantic", str(logit_lens_csv),
            n_bootstrap, ci_level, seed
        )
        out_path = output_dir / "patching_semantic_bootstrap_ci.csv"
        ci_sem.to_csv(out_path, index=False)
        print(f"    保存至: {out_path}")
    else:
        print(f"\n[2] Patching semantic CSV 不存在, 跳过")

    # --- Patching CI (prosody) ---
    if patching_pro_csv and patching_pro_csv.exists():
        print(f"\n[3] 处理 Patching (prosody) 数据: {patching_pro_csv}")
        ci_pro = compute_patching_ci(
            str(patching_pro_csv), "prosody", str(logit_lens_csv),
            n_bootstrap, ci_level, seed
        )
        out_path = output_dir / "patching_prosody_bootstrap_ci.csv"
        ci_pro.to_csv(out_path, index=False)
        print(f"    保存至: {out_path}")
    else:
        print(f"\n[3] Patching prosody CSV 不存在, 跳过")

    print("\n" + "=" * 60)
    print("EXP-1 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
