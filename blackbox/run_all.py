"""Orchestrator: run the complete black-box experiment pipeline.

Usage:
    python run_all.py                          # Run everything possible
    python run_all.py --phase prep             # Only sample preparation
    python run_all.py --phase attack           # Only main attack (adv samples)
    python run_all.py --phase noise            # Only noise baseline
    python run_all.py --phase clean            # Only clean baseline
    python run_all.py --phase analyze          # Only analysis
    python run_all.py --surrogates opens2s_en opens2s_cn  # Specific surrogates
    python run_all.py --targets gemini_flash qwen3_omni   # Specific targets
    python run_all.py --dry_run                # Dry run all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import cfg


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are available."""
    status = {}
    if cfg.gemini_api_key:
        status["gemini_flash"] = True
        status["gemini_pro"] = True
    if cfg.dashscope_api_key:
        status["qwen3_omni"] = True
        status["qwen_turbo"] = True
    if cfg.openai_api_key:
        status["gpt4o_audio"] = True

    return status


def check_wav_availability() -> dict[str, int]:
    """Check how many WAV files are available per surrogate."""
    from sample_loader import load_whitebox_results
    counts = {}
    for key in cfg.surrogate_groups:
        result_dir = cfg.get_surrogate_dir(key)
        samples = load_whitebox_results(result_dir)
        wav_count = sum(1 for s in samples if s.whitebox_success and s.adv_audio_path.exists())
        counts[key] = wav_count
    return counts


def run_prep(surrogates: list[str]):
    """Phase: Sample preparation."""
    print("\n" + "=" * 70)
    print("PHASE: Sample Preparation")
    print("=" * 70)

    from prepare_samples import generate_manifest, generate_noise_baseline, print_stats

    print("\n--- Dataset Statistics ---")
    print_stats()

    print("\n--- Generating Manifest ---")
    generate_manifest()

    print("\n--- Generating Random Noise Baseline ---")
    generate_noise_baseline()


def _run_single_evaluation(surrogate_key: str, target_key: str, audio_type: str, dry_run: bool):
    from evaluate import run_evaluation
    return run_evaluation(
        surrogate_key=surrogate_key,
        target_key=target_key,
        per_emotion=cfg.per_emotion,
        dry_run=dry_run,
        audio_type=audio_type,
    )


def run_attack(
    surrogates: list[str],
    targets: list[str],
    audio_type: str = "adv",
    dry_run: bool = False,
    max_workers: int = 1,
):
    """Phase: Run transfer attack evaluation."""
    phase_name = {"adv": "Transfer Attack", "clean": "Clean Baseline", "noise": "Noise Baseline"}
    print(f"\n{'='*70}")
    print(f"PHASE: {phase_name.get(audio_type, audio_type)}")
    print(f"Surrogates: {surrogates}")
    print(f"Targets: {targets}")
    print(f"{'='*70}")

    api_keys = check_api_keys()
    wav_avail = check_wav_availability()

    tasks: list[tuple[str, str]] = []
    for surrogate_key in surrogates:
        if audio_type == "adv" and wav_avail.get(surrogate_key, 0) == 0:
            print(f"\n  SKIP {surrogate_key}: no WAV files available")
            continue

        for target_key in targets:
            if target_key not in api_keys:
                print(f"\n  SKIP {target_key}: API key not set")
                continue
            tasks.append((surrogate_key, target_key))

    results = {}
    if not tasks:
        return results

    worker_count = max(1, min(max_workers, len(tasks)))
    if worker_count == 1:
        for surrogate_key, target_key in tasks:
            try:
                summary = _run_single_evaluation(surrogate_key, target_key, audio_type, dry_run)
                if summary:
                    results[f"{surrogate_key}/{target_key}"] = summary
            except Exception as e:
                print(f"\n  ERROR {surrogate_key} → {target_key}: {e}")
        return results

    print(f"\nLaunching {len(tasks)} evaluation jobs with up to {worker_count} workers")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_run_single_evaluation, surrogate_key, target_key, audio_type, dry_run): (surrogate_key, target_key)
            for surrogate_key, target_key in tasks
        }
        for future in as_completed(future_map):
            surrogate_key, target_key = future_map[future]
            try:
                summary = future.result()
                if summary:
                    results[f"{surrogate_key}/{target_key}"] = summary
            except Exception as e:
                print(f"\n  ERROR {surrogate_key} → {target_key}: {e}")

    return results


def run_analysis():
    """Phase: Comprehensive analysis."""
    print(f"\n{'='*70}")
    print("PHASE: Analysis")
    print(f"{'='*70}")

    from analyze import run_full_analysis
    from generate_report import generate_report
    run_full_analysis()
    report_path = generate_report()
    print(f"Report written: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Black-box experiment orchestrator")
    parser.add_argument("--phase", type=str, nargs="+",
                        choices=["prep", "attack", "clean", "noise", "analyze", "all"],
                        default=["all"])
    parser.add_argument("--surrogates", type=str, nargs="+",
                        default=list(cfg.surrogate_groups.keys()))
    parser.add_argument("--targets", type=str, nargs="+",
                        default=list(cfg.target_list.keys()))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_workers", type=int, default=cfg.run_all_max_workers)
    args = parser.parse_args()

    phases = args.phase
    if "all" in phases:
        phases = ["prep", "attack", "clean", "noise", "analyze"]

    print("=" * 70)
    print("BLACK-BOX TRANSFER ATTACK — FULL EXPERIMENT PIPELINE")
    print("=" * 70)

    # Status check
    print("\n--- API Key Status ---")
    api_keys = check_api_keys()
    for target_key in cfg.target_list:
        status = "✓" if target_key in api_keys else "✗ NOT SET"
        print(f"  {cfg.target_list[target_key]['name']}: {status}")

    print("\n--- WAV Availability ---")
    wav_avail = check_wav_availability()
    for key, count in wav_avail.items():
        status = f"✓ {count} files" if count > 0 else "✗ no WAV (remote only)"
        print(f"  {cfg.surrogate_groups[key]['name']}: {status}")

    # Execute phases
    if "prep" in phases:
        run_prep(args.surrogates)

    if "attack" in phases:
        run_attack(args.surrogates, args.targets, "adv", args.dry_run, args.max_workers)

    if "clean" in phases:
        run_attack(args.surrogates, args.targets, "clean", args.dry_run, args.max_workers)

    if "noise" in phases:
        run_attack(args.surrogates, args.targets, "noise", args.dry_run, args.max_workers)

    if "analyze" in phases:
        run_analysis()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
