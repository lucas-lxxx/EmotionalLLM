"""
Test Script: White-Box Attack on 20 Sad Audio Samples
Adjusted parameters based on N=10 experiment results
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add OpenS2S to path for src module imports
sys.path.insert(0, "/data1/lixiang/Opens2s/OpenS2S")

from scripts.sad2happy_batch_experiment import clean_inference, run_attack
from utils_audio import load_model, load_audio_extractor, load_waveform
from utils.emotion_classifier import FrozenEmotionClassifier
from constants import *

import torch
import numpy as np
import soundfile as sf
import json
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
OMNISPEECH_PATH = "/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S"
CHECKPOINT_PATH = "checkpoints/sad_happy_classifier.pt"
SAMPLE_LIST_PATH = "test2/sad_samples_20.txt"
OUTPUT_DIR = "test2/results_n20"
DEVICE = "cuda:0"

# Adjusted Attack Parameters (based on test_N10 analysis)
EPSILON = 0.005      # Increased from 0.002 (2.5x)
STEPS = 50           # Increased from 30
LAMBDA_EMO = 10.0    # Increased from 1.0 (10x) - prioritize emotion flip
LAMBDA_SEM = 0.005   # Decreased from 0.01 - reduce semantic constraint
LAMBDA_PER = 0.00005 # Decreased from 0.0001 - reduce perceptual constraint

# Prompt
PROMPT = "What is the emotion of this audio? Please answer with only one word: the emotion label (happy, sad, angry, or neutral)."

def main():
    print("=" * 80)
    print("OpenS2S White-Box Attack Test (N=20) - Adjusted Parameters")
    print("=" * 80)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio" / "clean").mkdir(parents=True, exist_ok=True)
    (output_dir / "audio" / "adv").mkdir(parents=True, exist_ok=True)
    (output_dir / "text").mkdir(parents=True, exist_ok=True)

    # Load sample list
    with open(SAMPLE_LIST_PATH, 'r') as f:
        audio_paths = [line.strip() for line in f if line.strip()]

    print(f"\nLoaded {len(audio_paths)} audio samples")
    print(f"Output directory: {output_dir}")
    print(f"\n🔧 Attack parameters (ADJUSTED):")
    print(f"  - epsilon: {EPSILON} (was 0.002)")
    print(f"  - steps: {STEPS} (was 30)")
    print(f"  - lambda_emo: {LAMBDA_EMO} (was 1.0)")
    print(f"  - lambda_sem: {LAMBDA_SEM} (was 0.01)")
    print(f"  - lambda_per: {LAMBDA_PER} (was 0.0001)")
    print(f"\nPrompt: {PROMPT}")
    print("=" * 80)

    # Load models
    print("\nLoading OpenS2S model...")
    model, tokenizer = load_model(OMNISPEECH_PATH, device=DEVICE)
    audio_extractor = load_audio_extractor(OMNISPEECH_PATH)

    # Load emotion classifier
    print(f"Loading emotion classifier from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    emotion_label_to_idx = checkpoint['label_to_idx']
    input_dim = checkpoint['svd_components'].shape[0] if 'svd_components' in checkpoint else checkpoint['input_dim']
    num_emotions = len(emotion_label_to_idx)

    emotion_classifier = FrozenEmotionClassifier(input_dim, num_emotions)
    emotion_classifier.load_state_dict(checkpoint['classifier'])
    emotion_classifier.to(DEVICE)
    emotion_classifier.freeze()

    print(f"  Emotion mapping: {emotion_label_to_idx}")
    print(f"  Input dim: {input_dim}, Num emotions: {num_emotions}")

    # Save configuration
    config = {
        "experiment": "test2_N20",
        "based_on": "test_N10 results (0% success rate)",
        "omnispeech_path": OMNISPEECH_PATH,
        "checkpoint_path": CHECKPOINT_PATH,
        "num_samples": len(audio_paths),
        "parameter_changes": {
            "epsilon": {"old": 0.002, "new": EPSILON, "change": "+150%"},
            "steps": {"old": 30, "new": STEPS, "change": "+67%"},
            "lambda_emo": {"old": 1.0, "new": LAMBDA_EMO, "change": "+900%"},
            "lambda_sem": {"old": 0.01, "new": LAMBDA_SEM, "change": "-50%"},
            "lambda_per": {"old": 0.0001, "new": LAMBDA_PER, "change": "-50%"}
        },
        "prompt": PROMPT,
        "device": DEVICE,
        "emotion_label_to_idx": emotion_label_to_idx
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Run attacks
    results = []
    success_count = 0
    print("\n" + "=" * 80)
    print("Running attacks...")
    print("=" * 80)

    for idx, audio_path in enumerate(audio_paths):
        sample_id = f"sample_{idx+1:03d}"
        print(f"\n[{idx+1}/{len(audio_paths)}] Processing: {Path(audio_path).name}")

        try:
            # Clean inference
            print("  [1/3] Clean inference...")
            clean_text = clean_inference(
                model, tokenizer, audio_extractor,
                audio_path, PROMPT, DEVICE
            )
            print(f"    Clean output: {clean_text[:100]}...")

            # Run attack
            print(f"  [2/3] Running attack (epsilon={EPSILON}, steps={STEPS})...")

            # Ensure SVD components are on correct device
            svd_components = checkpoint.get('svd_components', None)
            svd_mean = checkpoint.get('svd_mean', None)
            if svd_components is not None:
                if isinstance(svd_components, np.ndarray):
                    svd_components = torch.from_numpy(svd_components).float()
                svd_components = svd_components.to(DEVICE)
            if svd_mean is not None:
                if isinstance(svd_mean, np.ndarray):
                    svd_mean = torch.from_numpy(svd_mean).float()
                svd_mean = svd_mean.to(DEVICE)

            waveform_adv, metrics, attack_time, sample_rate = run_attack(
                model=model,
                tokenizer=tokenizer,
                audio_extractor=audio_extractor,
                audio_path=audio_path,
                prompt=PROMPT,
                target_emotion="Happy",
                source_emotion="Sad",
                emotion_classifier=emotion_classifier,
                emotion_label_to_idx=emotion_label_to_idx,
                svd_components=svd_components,
                svd_mean=svd_mean,
                epsilon=EPSILON,
                steps=STEPS,
                alpha=EPSILON / 10.0,
                lambda_emo=LAMBDA_EMO,
                lambda_sem=LAMBDA_SEM,
                lambda_per=LAMBDA_PER,
                device=DEVICE
            )

            # Save adversarial audio
            adv_audio_path = output_dir / "audio" / "adv" / f"{sample_id}.wav"
            sf.write(str(adv_audio_path), waveform_adv.cpu().numpy(), sample_rate)

            # Create clean audio symlink
            clean_audio_link = output_dir / "audio" / "clean" / f"{sample_id}.wav"
            if not clean_audio_link.exists():
                os.symlink(os.path.abspath(audio_path), clean_audio_link)

            # Attack inference
            print(f"  [3/3] Attack inference...")
            attack_text = clean_inference(
                model, tokenizer, audio_extractor,
                str(adv_audio_path), PROMPT, DEVICE
            )
            print(f"    Attack output: {attack_text[:100]}...")

            # Check success (case-insensitive contains "happy")
            is_success = "happy" in attack_text.lower()
            if is_success:
                success_count += 1
                print(f"    ✓ SUCCESS: Flipped to Happy!")
            else:
                print(f"    ✗ Failed: Still {attack_text.split()[0] if attack_text else 'unknown'}")

            # Record result
            result = {
                'sample_id': sample_id,
                'audio_path': audio_path,
                'clean_text': clean_text,
                'attack_text': attack_text,
                'success': is_success,
                'attack_time': attack_time,
                'linf': metrics.get('linf', None),
                'l2': metrics.get('l2', None),
                'snr_db': metrics.get('snr', None),
            }
            results.append(result)

            # Save texts
            with open(output_dir / "text" / f"{sample_id}_clean.txt", 'w') as f:
                f.write(clean_text)
            with open(output_dir / "text" / f"{sample_id}_attack.txt", 'w') as f:
                f.write(attack_text)

            print(f"  ⏱ Completed in {attack_time:.2f}s")
            print(f"    Metrics: linf={metrics.get('linf', 0):.6f}, "
                  f"l2={metrics.get('l2', 0):.2f}, "
                  f"snr={metrics.get('snr', 0):.2f}dB")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'sample_id': sample_id,
                'audio_path': audio_path,
                'error': str(e)
            })

    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)

    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {results_file}")

    # Statistics
    total_samples = len(results)
    processed_count = sum(1 for r in results if 'error' not in r)
    failed_count = total_samples - processed_count
    success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Total samples: {total_samples}")
    print(f"  Processed: {processed_count}")
    print(f"  Failed (errors): {failed_count}")
    print(f"  Emotion flip success: {success_count}/{processed_count}")
    print(f"  SUCCESS RATE: {success_rate:.1f}%")

    if processed_count > 0:
        avg_linf = np.mean([r['linf'] for r in results if 'linf' in r and r['linf'] is not None])
        avg_l2 = np.mean([r['l2'] for r in results if 'l2' in r and r['l2'] is not None])
        avg_snr = np.mean([r['snr_db'] for r in results if 'snr_db' in r and r['snr_db'] is not None])
        avg_time = np.mean([r['attack_time'] for r in results if 'attack_time' in r])

        print(f"\n  Average Metrics:")
        print(f"    L∞: {avg_linf:.6f}")
        print(f"    L2: {avg_l2:.3f}")
        print(f"    SNR: {avg_snr:.2f} dB")
        print(f"    Attack time: {avg_time:.2f}s")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
