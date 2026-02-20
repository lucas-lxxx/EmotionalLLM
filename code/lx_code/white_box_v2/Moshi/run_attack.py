"""
Main script to run Moshi white-box attack
Randomly select 10 sad->happy pairs from RAVDESS and attack them
"""
import json
import random
import os
from pathlib import Path
from datetime import datetime

import torch

from config import cfg
from moshi_io import load_moshi_model, load_audio, save_audio
from attack_core import attack_one_sample


def load_pairs(pairs_json_path: str) -> list:
    """Load pairs from JSON file"""
    with open(pairs_json_path, 'r') as f:
        pairs = json.load(f)
    return pairs


def select_random_samples(pairs: list, num_samples: int = 10) -> list:
    """
    Randomly select samples from pairs

    Args:
        pairs: List of emotion pairs
        num_samples: Number of samples to select

    Returns:
        List of selected pairs
    """
    if len(pairs) < num_samples:
        print(f"Warning: Only {len(pairs)} pairs available, selecting all")
        return pairs

    selected = random.sample(pairs, num_samples)
    print(f"Selected {len(selected)} random samples")
    return selected


def resolve_audio_path(relative_path: str, base_dir: str) -> str:
    """
    Resolve relative audio path to absolute path

    Args:
        relative_path: Relative path from pairs.json (e.g., "datasets/RAVDESS/...")
        base_dir: Base directory for RAVDESS dataset

    Returns:
        Absolute path to audio file
    """
    # Remove "datasets/RAVDESS/" prefix if present
    if relative_path.startswith("datasets/RAVDESS/"):
        relative_path = relative_path.replace("datasets/RAVDESS/", "")

    # Construct absolute path
    abs_path = os.path.join(base_dir, relative_path)

    return abs_path


def main():
    """Main attack pipeline"""
    print("=" * 60)
    print("Moshi White-Box Attack - Sad to Happy")
    print("=" * 60)
    print(f"Config:")
    print(f"  Model: {cfg.moshi_model_path}")
    print(f"  Dataset: {cfg.ravdess_base}")
    print(f"  Num samples: {cfg.num_samples}")
    print(f"  Epsilon: {cfg.epsilon}")
    print(f"  Steps: {cfg.total_steps}")
    print(f"  Device: {cfg.device}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir) / f"attack_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load pairs
    print("Loading pairs from JSON...")
    pairs = load_pairs(cfg.pairs_json)
    print(f"Total pairs: {len(pairs)}")

    # Select random samples
    selected_pairs = select_random_samples(pairs, cfg.num_samples)

    # Load Moshi model
    print("\nLoading Moshi model...")
    model = load_moshi_model(cfg.moshi_model_path, device=cfg.device)
    print()

    # Attack each sample
    results = []
    for idx, pair in enumerate(selected_pairs):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(selected_pairs)}")
        print(f"{'='*60}")

        # Get sad audio path
        sad_path_rel = pair['sad']
        sad_path = resolve_audio_path(sad_path_rel, cfg.ravdess_base)

        print(f"Sad audio: {sad_path}")
        print(f"Actor: {pair['actor']}, Statement: {pair['statement']}")

        # Check if file exists
        if not os.path.exists(sad_path):
            print(f"ERROR: File not found: {sad_path}")
            continue

        # Load audio
        try:
            waveform, sr = load_audio(sad_path, device=cfg.device)
            print(f"Audio loaded: shape={waveform.shape}, sr={sr}")
        except Exception as e:
            print(f"ERROR loading audio: {e}")
            continue

        # Attack
        try:
            result = attack_one_sample(
                model=model,
                waveform=waveform,
                sr=sr,
                target_emotion=cfg.target_emotion,
                device=cfg.device
            )

            # Save adversarial audio
            if cfg.save_audio:
                output_filename = f"sample_{idx:03d}_adv.wav"
                output_path = output_dir / output_filename
                save_audio(result['waveform_adv'], str(output_path), sr=sr)
                print(f"Saved adversarial audio: {output_path}")

            # Store result
            results.append({
                'idx': idx,
                'pair': pair,
                'sad_path': sad_path,
                'output_path': str(output_dir / output_filename) if cfg.save_audio else None,
                'loss_trace': result['loss_trace'],
                'grad_trace': result['grad_trace'],
            })

        except Exception as e:
            print(f"ERROR during attack: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results summary
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    print("\n" + "="*60)
    print("Attack completed!")
    print(f"Successfully attacked: {len(results)}/{len(selected_pairs)} samples")
    print("="*60)


if __name__ == "__main__":
    main()
