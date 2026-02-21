#!/usr/bin/env python3
"""Test OpenS2S emotion recognition accuracy on Sad dataset."""

import sys
import torch
import torchaudio
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, decode_text

def normalize_emo(text: str) -> str:
    """Extract emotion label from model output."""
    text = text.lower().strip()
    for emo in ['happy', 'sad', 'angry', 'neutral']:
        if emo in text:
            return emo
    return text  # Return as-is if no valid emotion found

def test_emotion_recognition(audio_dir: Path, num_samples: int = 50):
    """Test emotion recognition on samples from a directory."""
    print(f"Loading model...")
    model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
    model.eval()

    # Get audio files
    audio_files = list(audio_dir.rglob("*.wav"))[:num_samples]
    print(f"Testing on {len(audio_files)} samples from {audio_dir.name}/")

    results = []
    emotion_counts = Counter()

    for idx, audio_path in enumerate(audio_files):
        # Load audio (no resampling, use original 24kHz)
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.to(cfg.device)

        # Test with all 3 emotion prompts
        predictions = []
        for prompt in cfg.emo_prompts:
            output = decode_text(
                model=model,
                tokenizer=tokenizer,
                waveform=waveform,
                sr=sr,
                prompt=prompt,
                max_new_tokens=cfg.emo_max_new_tokens,
                temperature=cfg.temperature,
            )
            pred = normalize_emo(output)
            predictions.append(pred)

        # Count predictions
        for pred in predictions:
            emotion_counts[pred] += 1

        results.append({
            'file': audio_path.name,
            'predictions': predictions,
        })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(audio_files)}...")

    return results, emotion_counts

def main():
    sad_dir = Path("/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad")

    print("=" * 60)
    print("Testing OpenS2S Emotion Recognition on Sad Dataset")
    print("=" * 60)
    print(f"Config: sample_rate={cfg.sample_rate}, n_fft={cfg.n_fft}, hop={cfg.hop_length}")
    print()

    results, emotion_counts = test_emotion_recognition(sad_dir, num_samples=50)

    # Calculate statistics
    total_predictions = sum(emotion_counts.values())

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Total predictions (3 prompts × {len(results)} samples): {total_predictions}")
    print()

    print("Emotion Distribution:")
    for emotion, count in emotion_counts.most_common():
        percentage = count / total_predictions * 100
        print(f"  {emotion:15s}: {count:3d} ({percentage:5.1f}%)")

    print()
    print("Expected: 'sad' should be dominant (ideally >80%)")
    sad_accuracy = emotion_counts.get('sad', 0) / total_predictions * 100
    print(f"Actual 'sad' accuracy: {sad_accuracy:.1f}%")

    if sad_accuracy < 50:
        print("\n⚠️  WARNING: Model performs poorly on Sad emotion recognition!")
        print("   This explains why attack baseline shows no 'sad' predictions.")

    # Show some examples
    print("\n" + "=" * 60)
    print("Sample Predictions (first 10):")
    print("=" * 60)
    for i, result in enumerate(results[:10]):
        print(f"{i+1:2d}. {result['file']:20s} -> {result['predictions']}")

    # Check for invalid predictions
    invalid_preds = [pred for pred in emotion_counts.keys()
                     if pred not in ['happy', 'sad', 'angry', 'neutral']]
    if invalid_preds:
        print("\n⚠️  Invalid predictions found:")
        for pred in invalid_preds:
            count = emotion_counts[pred]
            print(f"   '{pred}': {count} times")

if __name__ == "__main__":
    main()
