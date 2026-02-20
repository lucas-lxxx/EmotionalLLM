"""
Test Script: White-Box Attack on 100 Sad Audio Samples
Further optimized parameters based on test2 N=20 results
Goal: Maximize success rate with aggressive attack parameters
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
SAMPLE_LIST_PATH = "test3/sad_samples_100.txt"
OUTPUT_DIR = "test3/results_n100"
DEVICE = "cuda:0"

# AGGRESSIVE Attack Parameters (further optimized from test2)
# Based on test2 analysis: successful sample had SNR=2.41dB (very strong perturbation)
EPSILON = 0.008      # Increased from 0.005 (+60%)
STEPS = 80           # Increased from 50 (+60%)
LAMBDA_EMO = 30.0    # Increased from 10.0 (3x) - strongly prioritize emotion flip
LAMBDA_SEM = 0.001   # Decreased from 0.005 (5x less) - minimal semantic constraint
LAMBDA_PER = 0.00001 # Decreased from 0.00005 (5x less) - minimal perceptual constraint

# Prompt
PROMPT = "What is the emotion of this audio? Please answer with only one word: the emotion label (happy, sad, angry, or neutral)."

def main():
    print("=" * 80)
    print("OpenS2S White-Box Attack Test (N=100) - AGGRESSIVE Parameters")
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
    print(f"\n🚀 Attack parameters (AGGRESSIVE - optimized from test2):")
    print(f"  - epsilon: {EPSILON} (was 0.005 in test2, 0.002 in test1)")
    print(f"  - steps: {STEPS} (was 50 in test2, 30 in test1)")
    print(f"  - lambda_emo: {LAMBDA_EMO} (was 10.0 in test2, 1.0 in test1)")
    print(f"  - lambda_sem: {LAMBDA_SEM} (was 0.005 in test2, 0.01 in test1)")
    print(f"  - lambda_per: {LAMBDA_PER} (was 0.00005 in test2, 0.0001 in test1)")
    print(f"\nStrategy: Based on test2 successful sample (SNR=2.41dB), use very strong perturbation")
    print(f"Prompt: {PROMPT}")
    print("=" * 80)

    # Load models
    print("\nLoading OpenS2S model...")
    model, tokenizer = load_model(OMNISPEECH_PATH, device=DEVICE)
    audio_extractor = load_audio_extractor(OMNISPEECH_PATH)

    # Load emotion classifier
    print(f"Loading emotion classifier from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

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
        "experiment": "test3_N100",
        "based_on": "test2_N20 results (5% success rate)",
        "strategy": "Aggressive parameters to maximize success rate",
        "omnispeech_path": OMNISPEECH_PATH,
        "checkpoint_path": CHECKPOINT_PATH,
        "num_samples": len(audio_paths),
        "parameter_evolution": {
            "test1_N10": {"epsilon": 0.002, "steps": 30, "lambda_emo": 1.0, "success_rate": "0%"},
            "test2_N20": {"epsilon": 0.005, "steps": 50, "lambda_emo": 10.0, "success_rate": "5%"},
            "test3_N100": {"epsilon": EPSILON, "steps": STEPS, "lambda_emo": LAMBDA_EMO, "success_rate": "TBD"}
        },
        "parameter_changes_from_test2": {
            "epsilon": {"old": 0.005, "new": EPSILON, "change": "+60%"},
            "steps": {"old": 50, "new": STEPS, "change": "+60%"},
            "lambda_emo": {"old": 10.0, "new": LAMBDA_EMO, "change": "+200%"},
            "lambda_sem": {"old": 0.005, "new": LAMBDA_SEM, "change": "-80%"},
            "lambda_per": {"old": 0.00005, "new": LAMBDA_PER, "change": "-80%"}
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
    print("Running attacks on 100 samples...")
    print("=" * 80)

    start_time = time.time()

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
            print(f"    Clean: {clean_text[:60]}...")

            # Run attack
            print(f"  [2/3] Attack (eps={EPSILON}, steps={STEPS})...")

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
            print(f"    Attack: {attack_text[:60]}...")

            # Check success (case-insensitive contains "happy")
            is_success = "happy" in attack_text.lower()
            if is_success:
                success_count += 1
                print(f"    ✅ SUCCESS #{success_count}: Flipped to Happy!")
            else:
                detected_emotion = attack_text.split()[0] if attack_text else 'unknown'
                print(f"    ❌ Failed: {detected_emotion}")

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

            print(f"  ⏱ {attack_time:.2f}s | Metrics: linf={metrics.get('linf', 0):.6f}, "
                  f"l2={metrics.get('l2', 0):.2f}, snr={metrics.get('snr', 0):.2f}dB")

            # Progress update every 10 samples
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                eta = avg_time * (len(audio_paths) - idx - 1)
                current_success_rate = success_count / (idx + 1) * 100
                print(f"\n  📊 Progress: {idx+1}/{len(audio_paths)} | Success: {success_count} ({current_success_rate:.1f}%) | ETA: {eta/60:.1f}min\n")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'sample_id': sample_id,
                'audio_path': audio_path,
                'error': str(e)
            })

    # Save results
    total_time = time.time() - start_time
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
    print(f"FINAL SUMMARY - Test3 N=100")
    print(f"{'='*80}")
    print(f"  Total samples: {total_samples}")
    print(f"  Processed: {processed_count}")
    print(f"  Failed (errors): {failed_count}")
    print(f"  Emotion flip success: {success_count}/{processed_count}")
    print(f"  🎯 SUCCESS RATE: {success_rate:.1f}%")
    print(f"  ⏱ Total time: {total_time/60:.1f} minutes ({total_time/processed_count:.2f}s per sample)")

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

    # Compare with previous experiments
    print(f"\n  📈 Success Rate Evolution:")
    print(f"    test1 N=10:  0.0% (epsilon=0.002, lambda_emo=1.0)")
    print(f"    test2 N=20:  5.0% (epsilon=0.005, lambda_emo=10.0)")
    print(f"    test3 N=100: {success_rate:.1f}% (epsilon={EPSILON}, lambda_emo={LAMBDA_EMO})")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
