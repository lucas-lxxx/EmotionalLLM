#!/usr/bin/env python3
"""Sanity checks for activation patching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import yaml

import torch

src_root = Path(__file__).resolve().parent.parent / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from label_tokenizer import LabelTokenizer
from patching import (
    load_opens2s_io,
    load_audio,
    add_labels,
    find_audio_token_index,
    compute_audio_span,
    compute_readout_pos,
    restricted_pred,
    run_activation_patching,
)
from pair_constructor import PairSpec


def setup_dataset_path(dataset_module: Path) -> None:
    dataset_src = dataset_module.parent.parent
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation Patching Sanity Checks")
    parser.add_argument(
        "--config",
        type=str,
        default="/data1/lixiang/lx_code/activation_patching/configs/patching_config.yaml",
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cpu or cuda)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.quick_test:
        config.setdefault("quick_test", {})
        config["quick_test"]["enabled"] = True
    if args.device:
        config["model"]["device"] = args.device

    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    opens2s_io = load_opens2s_io(Path(config["model"]["opens2s_io_path"]))
    model, tokenizer, audio_extractor, _ = opens2s_io.load_opens2s(
        Path(config["model"]["model_path"]),
        config["model"]["device"],
        Path(config["model"]["opens2s_root"]),
    )

    dataset_module = Path(config["data"]["dataset_module"])
    setup_dataset_path(dataset_module)
    from data.dataset import ModalConflictDataset

    dataset = ModalConflictDataset(
        text_jsonl=config["data"]["text_jsonl"],
        audio_root=config["data"]["audio_root"],
        emotions=config["data"].get("emotions"),
    )
    samples = dataset.get_conflict_samples()

    label_tokenizer = LabelTokenizer(tokenizer, emotions=config["data"].get("emotions"))

    # 1) Audio span check
    span_records = []
    for sample in samples[: args.num_samples]:
        wave, sr = load_audio(sample.audio_path, device="cpu")
        inputs = opens2s_io.build_inputs(
            wave,
            sr,
            config["prompt"],
            tokenizer,
            device=str(next(model.parameters()).device),
            audio_extractor=audio_extractor,
            torch_extractor=None,
            differentiable=False,
            system_prompt=config.get("system_prompt"),
            dtype=next(model.parameters()).dtype,
        )
        add_labels(inputs)
        input_len = inputs["input_ids"].shape[1]
        audio_idx = find_audio_token_index(inputs["input_ids"])

        with torch.inference_mode():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                speech_values=inputs.get("speech_values"),
                speech_mask=inputs.get("speech_mask"),
                labels=inputs.get("labels"),
                token_types=None,
                speech_units=None,
                speech_units_mask=None,
                spk_embs=None,
                return_dict=True,
            )

        logits_len = outputs.logits.shape[1]
        start, end, speech_len = compute_audio_span(input_len, audio_idx, logits_len)
        readout_pos = compute_readout_pos(input_len, logits_len)

        span_records.append(
            {
                "sample_id": sample.sample_id,
                "input_len": input_len,
                "logits_len": logits_len,
                "audio_token_idx": audio_idx,
                "audio_span_start": start,
                "audio_span_end": end,
                "speech_len": speech_len,
                "readout_pos": readout_pos,
            }
        )

    save_json(output_dir / "sanity_check_audio_span.json", span_records)

    # 2) Self-patch check (A == B)
    rng = random.Random(42)
    chosen = rng.sample(samples, k=min(args.num_samples, len(samples)))
    pairs = [PairSpec(f"self_{s.sample_id}", "prosody", s, s) for s in chosen]

    layers = config["patching"].get("layers_to_patch")
    if args.quick_test:
        layers = config.get("quick_test", {}).get("layers", layers)
    if not layers:
        layers = list(range(model.llm_model.config.num_hidden_layers))

    metrics, records = run_activation_patching(
        pairs=pairs,
        pair_type="prosody",
        model=model,
        tokenizer=tokenizer,
        audio_extractor=audio_extractor,
        opens2s_io=opens2s_io,
        label_tokenizer=label_tokenizer,
        prompt=config["prompt"],
        system_prompt=config.get("system_prompt"),
        layers_to_patch=layers,
        align_strategy=config["patching"].get("align_strategy", "truncate_to_min"),
        patch_alpha=config["patching"].get("patch_alpha", 1.0),
        cache_device=config["patching"].get("cache_device", "cpu"),
        save_records=True,
    )

    mismatch = [r for r in records if r["pred_patch"] != r["pred_base"]]
    save_json(
        output_dir / "sanity_check_self_patch.json",
        {
            "total_checks": len(records),
            "mismatch_count": len(mismatch),
            "mismatch_rate": len(mismatch) / max(len(records), 1),
        },
    )

    # 3) Random patch check (small)
    rand_pairs = []
    for s in chosen:
        other = rng.choice([x for x in samples if x.sample_id != s.sample_id])
        rand_pairs.append(PairSpec(f"random_{s.sample_id}", "prosody", s, other))

    rand_metrics, _ = run_activation_patching(
        pairs=rand_pairs,
        pair_type="prosody",
        model=model,
        tokenizer=tokenizer,
        audio_extractor=audio_extractor,
        opens2s_io=opens2s_io,
        label_tokenizer=label_tokenizer,
        prompt=config["prompt"],
        system_prompt=config.get("system_prompt"),
        layers_to_patch=layers,
        align_strategy=config["patching"].get("align_strategy", "truncate_to_min"),
        patch_alpha=config["patching"].get("patch_alpha", 1.0),
        cache_device=config["patching"].get("cache_device", "cpu"),
        save_records=False,
    )

    save_json(output_dir / "sanity_check_random_patch.json", rand_metrics.to_dict())

    print("Sanity checks completed.")


if __name__ == "__main__":
    main()
