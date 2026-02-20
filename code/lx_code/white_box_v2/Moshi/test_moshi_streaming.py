#!/usr/bin/env python3
"""
Test Moshi inference using correct streaming API
"""
import sys
import torch
from pathlib import Path

from moshi_io import load_audio
from moshi.models import loaders, LMGen


def test_moshi_streaming(model_path: str, audio_path: str, device: str = "cuda"):
    """
    Test Moshi streaming inference

    Args:
        model_path: Path to Moshi model
        audio_path: Path to test audio file
        device: Device to use
    """
    print("=" * 70)
    print("Moshi Streaming Inference Test")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Audio: {audio_path}")
    print(f"Device: {device}")
    print()

    # Load Mimi
    print("Loading Mimi encoder...")
    mimi_path = f"{model_path}/tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi = loaders.get_mimi(mimi_path, device=device)
    mimi.set_num_codebooks(8)
    print(f"✓ Mimi loaded (8 codebooks)")
    print(f"  Sample rate: {mimi.sample_rate}")
    print(f"  Frame rate: {mimi.frame_rate}")
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    print(f"  Frame size: {frame_size} samples")
    print()

    # Load Moshi LM
    print("Loading Moshi LM...")
    moshi_lm_path = f"{model_path}/model.safetensors"
    lm = loaders.get_moshi_lm(moshi_lm_path, device=device)
    print(f"✓ Moshi LM loaded")
    print(f"  Model type: {type(lm)}")
    print(f"  Num codebooks: {lm.num_codebooks}")
    print()

    # Create LMGen
    print("Creating LMGen...")
    lm_gen = LMGen(lm, temp=0.8, temp_text=0.7)
    print("✓ LMGen created")
    print()

    # Load audio
    print("Loading audio...")
    waveform, sr = load_audio(audio_path, device=device)
    print(f"✓ Audio loaded: shape={waveform.shape}, sr={sr}")

    # Check if audio length is multiple of frame_size
    audio_length = waveform.shape[-1]
    if audio_length % frame_size != 0:
        # Pad to multiple of frame_size
        pad_length = frame_size - (audio_length % frame_size)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        print(f"  Padded audio to {waveform.shape[-1]} samples")
    print()

    # Set streaming mode
    print("Setting streaming mode...")
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    print("✓ Streaming mode enabled")
    print()

    # Process audio frame by frame
    print("-" * 70)
    print("Processing audio frames...")
    print("-" * 70)

    num_frames = waveform.shape[-1] // frame_size
    print(f"Total frames: {num_frames}")
    print()

    all_tokens = []
    all_text_tokens = []

    with torch.no_grad():
        for frame_idx in range(num_frames):
            # Extract frame
            start = frame_idx * frame_size
            end = start + frame_size
            frame = waveform[:, :, start:end]

            # Encode frame
            codes = mimi.encode(frame)

            # Process each code in the frame
            for c in range(codes.shape[-1]):
                code_slice = codes[:, :, c: c + 1]

                # Generate tokens
                tokens = lm_gen.step(code_slice)

                if tokens is None:
                    continue

                all_tokens.append(tokens)

                # Extract text token
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):  # Skip special tokens
                    all_text_tokens.append(text_token)

                # Print progress
                if len(all_tokens) % 10 == 0:
                    print(f"  Processed {len(all_tokens)} steps, "
                          f"text tokens: {len(all_text_tokens)}")

    print()
    print(f"✓ Processing complete")
    print(f"  Total steps: {len(all_tokens)}")
    print(f"  Text tokens collected: {len(all_text_tokens)}")
    print()

    # Analyze tokens
    if len(all_tokens) > 0:
        print("-" * 70)
        print("Token Analysis")
        print("-" * 70)

        sample_tokens = all_tokens[0]
        print(f"Token shape: {sample_tokens.shape}")
        print(f"Token dtype: {sample_tokens.dtype}")
        print(f"Token device: {sample_tokens.device}")
        print()

        # Show first few text tokens
        if len(all_text_tokens) > 0:
            print(f"First 10 text tokens: {all_text_tokens[:10]}")
            print()

    print("=" * 70)
    print("Test completed!")
    print("=" * 70)


def main():
    """Main function"""
    model_path = "/data1/lixiang/Moshi/moshiko-pytorch-bf16/kyutai/moshiko-pytorch-bf16"

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "/data1/lixiang/OpenS2S_dataset/RAVDESS/Actor_01/03-01-04-01-01-01-01.wav"
        print(f"No audio path provided, using default: {audio_path}")
        print()

    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        print("Usage: python test_moshi_streaming.py <audio_path>")
        sys.exit(1)

    test_moshi_streaming(model_path, audio_path)


if __name__ == "__main__":
    main()
