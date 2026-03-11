#!/usr/bin/env python3
"""
testN10 White-box Attack Experiment
Based on the methodology from the 5 core files:
- config.py: Configuration management
- opens2s_io.py: OpenS2S model loading and I/O
- attack_core.py: Core attack algorithm with EoT
- eval_metrics.py: Evaluation metrics (WER, SNR, etc.)
- run_attack.py: Main attack pipeline

This script runs white-box adversarial attacks on 10 ESD/Angry audio samples
to manipulate emotion classification while preserving semantic content.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import torch

# Add parent directory to path to import core modules
parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))

from attack_core import attack_one_sample, compute_target_token_ids
from eval_metrics import aggregate_results, compute_wer, normalize_text, signal_metrics
from opens2s_io import decode_text, load_opens2s

# Import testN10 config
from config_testN10 import cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.get_device_properties(0)
    except Exception:
        return False
    return True


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio

        waveform, sr = torchaudio.load(str(path))
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
        return waveform.float(), sr
    except Exception:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).unsqueeze(0)
            if sr != target_sr:
                raise RuntimeError("Resample requires torchaudio or ffmpeg; install or pre-resample inputs.")
            return waveform, sr
        except Exception:
            import array
            import subprocess

            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "-f",
                "f32le",
                "-",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            data = array.array("f")
            data.frombytes(proc.stdout)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            return waveform, target_sr


def save_audio(path: Path, waveform: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = waveform.detach().cpu()
    if data.dim() == 2 and data.size(0) == 1:
        data = data.squeeze(0)
    try:
        import soundfile as sf

        sf.write(str(path), data.numpy(), sr)
    except Exception:
        import torchaudio

        torchaudio.save(str(path), data.unsqueeze(0), sr)


def parse_sample_list(path: Path) -> list[Path]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        samples.append(Path(parts[0]))
    return samples


def normalize_emo(text: str, labels: list[str]) -> str:
    # Methodology §1.2: success if exact target word after normalization.
    text = normalize_text(text)
    text = re.sub(r"[^a-z]+", " ", text)
    if not text.strip():
        return ""
    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return text.strip().split(" ")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="testN10 White-box Attack Experiment")
    parser.add_argument("--sample_list", type=str, default=str(cfg.sample_list_path))
    parser.add_argument("--results_dir", type=str, default=str(cfg.results_dir))
    args = parser.parse_args()

    print("=" * 80)
    print("testN10 White-box Attack Experiment")
    print("=" * 80)
    print(f"Sample list: {args.sample_list}")
    print(f"Results dir: {args.results_dir}")
    print(f"Device: {cfg.device}")
    print(f"Target emotion: {cfg.target_emotion}")
    print(f"Attack steps: {cfg.total_steps}")
    print(f"Epsilon: {cfg.epsilon}")
    print("=" * 80)

    set_seed(cfg.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = cfg.device
    if runtime_device.startswith("cuda") and not cuda_available():
        print("WARNING: CUDA not available, falling back to CPU")
        runtime_device = "cpu"

    print(f"\nLoading OpenS2S model from {cfg.model_path}...")
    model, tokenizer, audio_extractor, torch_extractor = load_opens2s(
        cfg.model_path, runtime_device, cfg.opens2s_root
    )
    print("Model loaded successfully!")

    target_token_ids = compute_target_token_ids(tokenizer, cfg.target_emotion)
    print(f"Target token IDs for '{cfg.target_emotion}': {target_token_ids}")

    ignore_index = -100
    target_sr = audio_extractor.sampling_rate

    sample_paths = parse_sample_list(Path(args.sample_list))
    print(f"\nFound {len(sample_paths)} samples to process")

    per_sample_results = []

    for idx, sample_path in enumerate(sample_paths):
        sample_id = f"{idx:05d}_{sample_path.stem}"
        out_json = results_dir / f"{sample_id}.json"
        out_wav = results_dir / f"{sample_id}.wav"

        if cfg.skip_existing and out_json.exists():
            print(f"\n[{idx+1}/{len(sample_paths)}] Skipping {sample_id} (already exists)")
            continue

        print(f"\n[{idx+1}/{len(sample_paths)}] Processing {sample_id}")
        print(f"  Source: {sample_path}")

        waveform, sr = load_audio(sample_path, target_sr)
        waveform = waveform.to(runtime_device)

        # Methodology §8.1: decode clean emotion and transcript with fixed settings.
        print("  Decoding clean audio...")
        emo_text_clean = [
            decode_text(
                model,
                tokenizer,
                waveform,
                sr,
                prompt,
                cfg.emo_max_new_tokens,
                cfg.temperature,
                audio_extractor=audio_extractor,
                system_prompt=cfg.system_prompt,
            )
            for prompt in cfg.emo_prompts
        ]
        asr_text_clean = decode_text(
            model,
            tokenizer,
            waveform,
            sr,
            cfg.asr_prompts[0],
            cfg.asr_max_new_tokens,
            cfg.temperature,
            audio_extractor=audio_extractor,
            system_prompt=cfg.system_prompt,
        )

        emo_pred_clean = [normalize_emo(t, cfg.emo_labels) for t in emo_text_clean]
        print(f"  Clean emotion predictions: {emo_pred_clean}")
        print(f"  Clean transcript: {asr_text_clean}")

        # Methodology §5.2: self-consistency uses OpenS2S transcript as target.
        asr_target_token_ids = tokenizer.encode(asr_text_clean, add_special_tokens=False)
        if not asr_target_token_ids:
            asr_target_token_ids = tokenizer.encode(" " + asr_text_clean, add_special_tokens=False)

        print(f"  Running attack ({cfg.total_steps} steps)...")
        attack_out = attack_one_sample(
            model=model,
            tokenizer=tokenizer,
            waveform=waveform,
            sr=sr,
            target_token_ids=target_token_ids,
            asr_prompt=cfg.asr_prompts[0],
            asr_target_token_ids=asr_target_token_ids,
            device=runtime_device,
            ignore_index=ignore_index,
            torch_extractor=torch_extractor,
            system_prompt=cfg.system_prompt,
        )

        waveform_adv = attack_out["waveform_adv"]
        save_audio(out_wav, waveform_adv, sr)
        print(f"  Saved adversarial audio to {out_wav}")

        # Methodology §8.1: decode adversarial outputs.
        print("  Decoding adversarial audio...")
        emo_text_adv = [
            decode_text(
                model,
                tokenizer,
                waveform_adv,
                sr,
                prompt,
                cfg.emo_max_new_tokens,
                cfg.temperature,
                audio_extractor=audio_extractor,
                system_prompt=cfg.system_prompt,
            )
            for prompt in cfg.emo_prompts
        ]
        asr_text_adv = decode_text(
            model,
            tokenizer,
            waveform_adv,
            sr,
            cfg.asr_prompts[0],
            cfg.asr_max_new_tokens,
            cfg.temperature,
            audio_extractor=audio_extractor,
            system_prompt=cfg.system_prompt,
        )

        # Methodology §1.2: success criteria.
        emo_pred_adv = [normalize_emo(t, cfg.emo_labels) for t in emo_text_adv]
        success_emo = all(p == cfg.target_emotion for p in emo_pred_adv)
        wer = compute_wer(asr_text_clean, asr_text_adv)

        metrics = signal_metrics(waveform_adv, waveform)

        print(f"  Adversarial emotion predictions: {emo_pred_adv}")
        print(f"  Adversarial transcript: {asr_text_adv}")
        print(f"  Success: {success_emo}, WER: {wer:.4f}")
        print(f"  L_inf: {metrics['delta_linf']:.6f}, SNR: {metrics['snr_db']:.2f} dB")

        sample_result = {
            "sample_id": sample_id,
            "path": str(sample_path),
            "emo_text_clean": emo_text_clean,
            "emo_text_adv": emo_text_adv,
            "emo_pred_clean": emo_pred_clean,
            "emo_pred_adv": emo_pred_adv,
            "asr_text_clean": asr_text_clean,
            "asr_text_adv": asr_text_adv,
            "success_emo": success_emo,
            "wer": wer,
            "delta_linf": metrics["delta_linf"],
            "delta_l2": metrics["delta_l2"],
            "snr_db": metrics["snr_db"],
            "grad_norm_trace": attack_out["grad_trace"],
            "loss_trace": attack_out["loss_trace"],
        }

        out_json.write_text(json.dumps(sample_result, ensure_ascii=True, indent=2), encoding="utf-8")
        per_sample_results.append(sample_result)

    print("\n" + "=" * 80)
    print("Computing aggregate statistics...")
    summary = aggregate_results(per_sample_results, cfg.wer_thresholds)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")

    summary_csv = results_dir / "summary.csv"
    if summary:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print(f"Saved summary CSV to {summary_csv}")

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value}")
    print("=" * 80)
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
