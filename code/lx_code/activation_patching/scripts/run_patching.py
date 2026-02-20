#!/usr/bin/env python3
"""Main entry for activation patching experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml
import csv

src_root = Path(__file__).resolve().parent.parent / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from label_tokenizer import LabelTokenizer
from pair_constructor import (
    construct_prosody_pairs,
    construct_semantic_pairs,
    pair_to_dict,
)
from patching import load_opens2s_io, run_activation_patching
from visualization import plot_flip_rate_curve, plot_delta_logit_curve


def setup_dataset_path(dataset_module: Path) -> None:
    dataset_src = dataset_module.parent.parent
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_metrics_csv(path: Path, metrics: dict) -> None:
    rows = []
    for idx, layer in enumerate(metrics["layer_indices"]):
        rows.append(
            {
                "layer": layer,
                "flip_to_target_rate": metrics["flip_to_target_rate"][idx],
                "flip_from_base_rate": metrics["flip_from_base_rate"][idx],
                "delta_logit_target_mean": metrics["delta_logit_target_mean"][idx],
                "eligible_count": metrics["eligible_count"][idx],
                "n_pairs": metrics["n_pairs"],
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_quick_test_overrides(config: dict) -> dict:
    qt = config.get("quick_test", {})
    if not qt.get("enabled", False):
        return config
    config["pairing"]["max_pairs_prosody"] = qt.get("max_pairs_prosody", 10)
    config["pairing"]["max_pairs_semantic"] = qt.get("max_pairs_semantic", 10)
    config["patching"]["layers_to_patch"] = qt.get("layers", [])
    config["output"]["save_records"] = qt.get("save_records", False)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation Patching Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="/data1/lixiang/lx_code/activation_patching/configs/patching_config.yaml",
    )
    parser.add_argument("--quick_test", action="store_true", help="Enable quick test overrides")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cpu or cuda)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.quick_test:
        config.setdefault("quick_test", {})
        config["quick_test"]["enabled"] = True
    if args.device:
        config["model"]["device"] = args.device

    config = apply_quick_test_overrides(config)

    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    opens2s_io = load_opens2s_io(Path(config["model"]["opens2s_io_path"]))
    model, tokenizer, audio_extractor, _torch_extractor = opens2s_io.load_opens2s(
        Path(config["model"]["model_path"]),
        config["model"]["device"],
        Path(config["model"]["opens2s_root"]),
    )

    # Load dataset
    dataset_module = Path(config["data"]["dataset_module"])
    setup_dataset_path(dataset_module)
    from data.dataset import ModalConflictDataset

    dataset = ModalConflictDataset(
        text_jsonl=config["data"]["text_jsonl"],
        audio_root=config["data"]["audio_root"],
        emotions=config["data"].get("emotions"),
    )

    if config["data"].get("conflict_only", True):
        samples = dataset.get_conflict_samples()
    else:
        samples = list(dataset)

    label_tokenizer = LabelTokenizer(tokenizer, emotions=config["data"].get("emotions"))
    label_tokenizer.print_summary()
    label_tokenizer.save_report(str(output_dir / "tokenization_report.json"))

    # Pair construction
    prosody_pairs, prosody_report = construct_prosody_pairs(
        samples,
        max_pairs=config["pairing"].get("max_pairs_prosody"),
        seed=config["pairing"].get("pair_sampling_seed", 42),
    )
    semantic_pairs, semantic_report = construct_semantic_pairs(
        samples,
        max_pairs=config["pairing"].get("max_pairs_semantic"),
        seed=config["pairing"].get("pair_sampling_seed", 42),
    )

    pair_report = {
        "prosody": prosody_report,
        "semantic": semantic_report,
    }
    save_json(output_dir / "pair_construction_report.json", pair_report)

    save_jsonl(output_dir / "pair_list_prosody.jsonl", [pair_to_dict(p) for p in prosody_pairs])
    save_jsonl(output_dir / "pair_list_semantic.jsonl", [pair_to_dict(p) for p in semantic_pairs])

    save_json(output_dir / "patching_config.json", config)

    layers_to_patch = config["patching"].get("layers_to_patch")
    if not layers_to_patch:
        n_layers = model.llm_model.config.num_hidden_layers
        layers_to_patch = list(range(n_layers))

    # Run patching
    for pair_type, pairs in [("prosody", prosody_pairs), ("semantic", semantic_pairs)]:
        metrics, records = run_activation_patching(
            pairs=pairs,
            pair_type=pair_type,
            model=model,
            tokenizer=tokenizer,
            audio_extractor=audio_extractor,
            opens2s_io=opens2s_io,
            label_tokenizer=label_tokenizer,
            prompt=config["prompt"],
            system_prompt=config.get("system_prompt"),
            layers_to_patch=layers_to_patch,
            align_strategy=config["patching"].get("align_strategy", "truncate_to_min"),
            patch_alpha=config["patching"].get("patch_alpha", 1.0),
            cache_device=config["patching"].get("cache_device", "cpu"),
            save_records=config["output"].get("save_records", False),
        )

        metrics_dict = metrics.to_dict()
        metrics_path = output_dir / f"patching_metrics_{pair_type}.csv"
        save_metrics_csv(metrics_path, metrics_dict)

        if records and config["output"].get("save_records", False):
            records_path = output_dir / f"patching_records_{pair_type}.jsonl"
            save_jsonl(records_path, records)

        # Plots
        plot_flip_rate_curve(
            metrics_dict,
            str(output_dir / f"flip_rate_curve_{pair_type}.png"),
            title=f"Flip Rate Curve ({pair_type})",
        )
        if config["output"].get("save_delta_logit", True):
            plot_delta_logit_curve(
                metrics_dict,
                str(output_dir / f"delta_logit_curve_{pair_type}.png"),
                title=f"Delta Logit Curve ({pair_type})",
            )

    print("Activation patching finished.")


if __name__ == "__main__":
    main()
