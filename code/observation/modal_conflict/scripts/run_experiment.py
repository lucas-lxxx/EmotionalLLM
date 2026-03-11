#!/usr/bin/env python3
"""
OpenS2S Modal Conflict Experiment - Main Entry Point

研究OpenS2S模型中语义情绪与韵律情绪的模态冲突。
"""

import argparse
import json
import sys
import os
import hashlib
from pathlib import Path
from datetime import datetime

import yaml
import torch
import random
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ModalConflictDataset, load_dataset
from src.models.feature_extractor import HiddenStatesExtractor, load_extractor
from src.evaluation.cross_validation import run_evaluation
from src.visualization.plotting import generate_all_plots


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_output_dir(config: dict) -> Path:
    """设置输出目录"""
    output_dir = Path(config["output"]["results_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_summary(summary: dict, output_path: str):
    """保存实验摘要"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved summary to {output_path}")


def set_seed(seed: int):
    """设置随机种子以提升复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    print("=" * 60)
    print("OpenS2S Modal Conflict Experiment")
    print("=" * 60)

    # 加载配置
    print(f"\n[1/6] Loading config from {args.config}")
    config = load_config(args.config)

    # 覆盖配置 (命令行参数优先)
    if args.device:
        config["model"]["device"] = args.device
    if args.cache_dir:
        config["extraction"]["cache_dir"] = args.cache_dir
    if args.no_cache:
        config["extraction"]["use_cache"] = False

    # 设置输出目录
    run_dir = setup_output_dir(config)
    print(f"Output directory: {run_dir}")

    # 保存使用的配置
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # 设置随机种子
    set_seed(config["evaluation"]["random_seed"])

    # 加载数据集
    print(f"\n[2/6] Loading dataset")
    dataset = load_dataset(config)
    print(dataset)
    print(f"  - Total samples: {dataset.n_samples}")
    print(f"  - Conflict samples: {dataset.n_conflict}")
    print(f"  - Consistent samples: {dataset.n_consistent}")

    # 保存数据集信息
    with open(run_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset.summary(), f, indent=2, ensure_ascii=False)

    # 提取Hidden States
    print(f"\n[3/6] Extracting hidden states")
    extractor = load_extractor(config)

    # 检查是否有缓存的完整提取结果
    cache_key = hashlib.md5(
        f"{config['model']['model_path']}_{config.get('prompt')}_{config.get('system_prompt')}".encode()
    ).hexdigest()
    cache_file = Path(config["extraction"]["cache_dir"]) / f"all_hidden_states_{cache_key}.pt"
    if config["extraction"]["use_cache"] and cache_file.exists():
        print(f"Loading cached hidden states from {cache_file}")
        extracted_data = torch.load(cache_file, map_location="cpu")
    else:
        extracted_data = extractor.extract_batch(dataset.samples, show_progress=True)
        # 保存完整提取结果
        if config["extraction"]["use_cache"]:
            torch.save(extracted_data, cache_file)
            print(f"Saved hidden states to {cache_file}")

    hidden_states = extracted_data["hidden_states"]
    semantic_labels = extracted_data["semantic_labels"]
    prosody_labels = extracted_data["prosody_labels"]
    text_ids = extracted_data["text_ids"]
    is_conflict = extracted_data["is_conflict"]

    print(f"  - Hidden states shape: {hidden_states.shape}")
    print(f"  - Number of layers: {hidden_states.shape[1]}")
    print(f"  - Hidden dimension: {hidden_states.shape[2]}")

    # 运行评估
    print(f"\n[4/6] Running layer-wise evaluation")
    results_df, summary = run_evaluation(
        hidden_states=hidden_states,
        semantic_labels=semantic_labels,
        prosody_labels=prosody_labels,
        text_ids=text_ids,
        is_conflict=is_conflict,
        config=config
    )

    # 打印摘要
    print(f"\n[5/6] Results Summary")
    print("-" * 40)
    print(f"Overall dominant modality: {summary['overall_dominant_modality']}")
    print(f"Average dominance: {summary['average_dominance']:.4f}")
    print(f"Max semantic accuracy: Layer {summary['max_semantic_acc']['layer']} "
          f"({summary['max_semantic_acc']['accuracy']:.4f})")
    print(f"Max prosody accuracy: Layer {summary['max_prosody_acc']['layer']} "
          f"({summary['max_prosody_acc']['accuracy']:.4f})")
    print(f"Max prosody dominance: Layer {summary['max_prosody_dominance']['layer']} "
          f"({summary['max_prosody_dominance']['dominance']:.4f})")
    print(f"Max semantic dominance: Layer {summary['max_semantic_dominance']['layer']} "
          f"({summary['max_semantic_dominance']['dominance']:.4f})")

    print("\nLayer trends:")
    trends = summary['layer_trends']
    print(f"  Early layers (0-11): {trends['early_layers_dominance']:.4f}")
    print(f"  Middle layers (12-23): {trends['middle_layers_dominance']:.4f}")
    print(f"  Late layers (24-35): {trends['late_layers_dominance']:.4f}")

    print("\nConflict subset:")
    conflict = summary['conflict_subset']
    print(f"  Avg semantic acc: {conflict['avg_semantic_acc']:.4f}")
    print(f"  Avg prosody acc: {conflict['avg_prosody_acc']:.4f}")
    print(f"  Avg dominance: {conflict['avg_dominance']:.4f}")

    # 生成可视化
    print(f"\n[6/6] Generating visualizations")
    if config["output"]["save_plots"]:
        generate_all_plots(results_df, str(run_dir))

    # 保存摘要
    if config["output"]["save_summary"]:
        save_summary(summary, str(run_dir / "summary.json"))

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)

    return results_df, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenS2S Modal Conflict Experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache directory"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
