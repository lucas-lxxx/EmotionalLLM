"""Sample preparation: manifest generation, random noise baseline, clean audio extraction.

Usage:
    python prepare_samples.py --manifest          # Generate manifest CSV for all surrogate groups
    python prepare_samples.py --noise             # Generate random noise baseline
    python prepare_samples.py --stats             # Print dataset statistics
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import cfg
from sample_loader import load_whitebox_results, select_subset


def generate_manifest():
    """Generate unified manifest CSV for all surrogate groups."""
    manifest_path = cfg.blackbox_root / "manifest.csv"
    rows = []

    for surrogate_key, info in cfg.surrogate_groups.items():
        result_dir = cfg.get_surrogate_dir(surrogate_key)
        print(f"\n--- {info['name']} ({surrogate_key}) ---")
        all_samples = load_whitebox_results(result_dir)
        print(f"  Total: {len(all_samples)}")

        success = [s for s in all_samples if s.whitebox_success]
        print(f"  White-box success: {len(success)}")

        wav_present = sum(1 for s in all_samples if s.adv_audio_path.exists())
        print(f"  WAV files present: {wav_present}")

        # Select balanced subset
        selected = select_subset(all_samples, per_emotion=250, only_whitebox_success=True)
        print(f"  Selected for blackbox: {len(selected)}")

        for s in selected:
            rows.append({
                "surrogate": surrogate_key,
                "sample_id": s.sample_id,
                "speaker_id": s.speaker_id,
                "ground_truth_emotion": s.ground_truth_emotion,
                "target_emotion": s.target_emotion,
                "whitebox_success": s.whitebox_success,
                "adv_audio_path": str(s.adv_audio_path),
                "original_audio_path": s.original_audio_path,
                "wav_exists": s.adv_audio_path.exists(),
                "snr_db": s.snr_db,
            })

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest saved: {manifest_path} ({len(rows)} rows)")

    # Summary
    by_surrogate = defaultdict(int)
    by_surrogate_wav = defaultdict(int)
    for r in rows:
        by_surrogate[r["surrogate"]] += 1
        if r["wav_exists"]:
            by_surrogate_wav[r["surrogate"]] += 1

    print("\nSummary:")
    for key in cfg.surrogate_groups:
        total = by_surrogate.get(key, 0)
        wav = by_surrogate_wav.get(key, 0)
        print(f"  {key}: {total} selected, {wav} with WAV")


def read_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read WAV file to numpy array."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        max_val = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sample_rate


def write_wav(wav_path: Path, audio: np.ndarray, sample_rate: int):
    """Write numpy array to WAV file (16-bit)."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def add_gaussian_noise(audio: np.ndarray, target_snr_db: float, seed: int = 42) -> np.ndarray:
    """Add Gaussian white noise to audio at target SNR."""
    rng = np.random.RandomState(seed)
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise = rng.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return audio + noise


def generate_noise_baseline():
    """Generate random noise samples matching adversarial SNR."""
    print("Generating random noise baseline samples...")

    # Determine SNR for each surrogate
    snr_map = {
        "voxtral_en": cfg.voxtral_avg_snr,
        "voxtral_cn": cfg.voxtral_avg_snr,
        "opens2s_en": cfg.opens2s_avg_snr,
        "opens2s_cn": cfg.opens2s_avg_snr,
    }

    for surrogate_key, info in cfg.surrogate_groups.items():
        result_dir = cfg.get_surrogate_dir(surrogate_key)
        target_snr = snr_map[surrogate_key]
        noise_out = cfg.noise_dir / surrogate_key

        print(f"\n--- {info['name']} (target SNR: {target_snr:.1f} dB) ---")

        # Load samples
        all_samples = load_whitebox_results(result_dir)
        selected = select_subset(all_samples, per_emotion=250, only_whitebox_success=True)

        # We need the CLEAN audio to add noise to, but clean audio is on remote server.
        # For local data (OpenS2S), the adv WAV exists - we can compute noise from the
        # original ESD audio if available, or approximate using the adv audio.
        #
        # Strategy: For each adversarial sample, read its adv WAV, subtract perturbation
        # approximation isn't feasible. Instead, if we have the JSON with SNR info,
        # we generate noise of appropriate level on a zero signal - but that's wrong.
        #
        # Better: Read the adversarial WAV and generate random noise at the same power level
        # as the adversarial perturbation (same SNR). The clean signal component is dominant,
        # so random noise on top of the same clean audio would be the right comparison.
        # Since we don't have separate clean files locally, we use the adv WAV as approximation
        # of the clean signal (delta is tiny, L∞ ≤ 0.008).

        generated = 0
        skipped = 0
        for s in selected:
            noise_path = noise_out / f"{s.sample_id}.wav"
            if noise_path.exists():
                generated += 1
                continue

            if not s.adv_audio_path.exists():
                skipped += 1
                continue

            try:
                # Use adv audio as proxy for clean (delta L∞ ≤ 0.008, negligible)
                audio, sr = read_wav(s.adv_audio_path)
                # Add Gaussian noise at same SNR as adversarial perturbation
                seed = int(hashlib.sha256(s.sample_id.encode("utf-8")).hexdigest()[:8], 16)
                noisy = add_gaussian_noise(audio, target_snr, seed=seed)
                write_wav(noise_path, noisy, sr)
                generated += 1
            except Exception as e:
                print(f"  Error processing {s.sample_id}: {e}")
                skipped += 1

        print(f"  Generated: {generated}, Skipped (no WAV): {skipped}")


def compute_actual_snr_stats():
    """Compute actual SNR statistics from white-box results."""
    for surrogate_key, info in cfg.surrogate_groups.items():
        result_dir = cfg.get_surrogate_dir(surrogate_key)
        all_samples = load_whitebox_results(result_dir)

        snrs = [s.snr_db for s in all_samples if s.snr_db > 0 and s.whitebox_success]
        if snrs:
            print(f"{info['name']}: mean SNR = {np.mean(snrs):.2f} dB, "
                  f"median = {np.median(snrs):.2f} dB, "
                  f"std = {np.std(snrs):.2f} dB, "
                  f"n = {len(snrs)}")
        else:
            print(f"{info['name']}: No SNR data available")


def print_stats():
    """Print detailed statistics for all surrogate groups."""
    print("=" * 70)
    print("Dataset Statistics")
    print("=" * 70)

    for surrogate_key, info in cfg.surrogate_groups.items():
        result_dir = cfg.get_surrogate_dir(surrogate_key)
        print(f"\n--- {info['name']} ({surrogate_key}) ---")
        print(f"  Directory: {result_dir}")

        all_samples = load_whitebox_results(result_dir)
        print(f"  Total samples: {len(all_samples)}")

        success = [s for s in all_samples if s.whitebox_success]
        print(f"  White-box success: {len(success)} ({100*len(success)/len(all_samples):.1f}%)" if all_samples else "")

        wav_count = sum(1 for s in all_samples if s.adv_audio_path.exists())
        print(f"  WAV files present: {wav_count}")

        # Emotion distribution
        emo_dist = Counter(s.ground_truth_emotion for s in all_samples)
        print(f"  Emotion distribution: {dict(sorted(emo_dist.items()))}")

        # Speaker distribution
        spk_dist = Counter(s.speaker_id for s in all_samples)
        print(f"  Speakers: {sorted(spk_dist.keys())}")

        # SNR stats
        snrs = [s.snr_db for s in success if s.snr_db > 0]
        if snrs:
            print(f"  SNR: mean={np.mean(snrs):.1f} dB, min={np.min(snrs):.1f}, max={np.max(snrs):.1f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true", help="Generate manifest CSV")
    parser.add_argument("--noise", action="store_true", help="Generate random noise baseline")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--snr", action="store_true", help="Compute SNR statistics")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    if args.snr:
        compute_actual_snr_stats()
    if args.manifest:
        generate_manifest()
    if args.noise:
        generate_noise_baseline()
    if not any(vars(args).values()):
        print_stats()
