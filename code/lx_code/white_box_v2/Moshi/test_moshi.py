#!/usr/bin/env python3
"""
Test script to observe Moshi's response and internal process
"""
import sys
import torch
import torchaudio
from pathlib import Path

from moshi_io import load_moshi_model, load_audio


def test_moshi_inference(model_path: str, audio_path: str, device: str = "cuda"):
    """
    Test Moshi model inference on a single audio file

    Args:
        model_path: Path to Moshi model
        audio_path: Path to test audio file
        device: Device to use
    """
    print("=" * 70)
    print("Moshi Model Test - Observing Response and Internal Process")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Audio: {audio_path}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading Moshi model...")
    model = load_moshi_model(model_path, device=device)

    # Set number of codebooks to 8 (Moshi requirement)
    model.mimi.set_num_codebooks(8)
    print("✓ Mimi set to 8 codebooks")

    print("✓ Model loaded")
    print()

    # Load audio
    print("Loading audio...")
    waveform, sr = load_audio(audio_path, device=device)
    print(f"✓ Audio loaded: shape={waveform.shape}, sr={sr}")
    print()

    # Test 1: Encode audio to codes
    print("-" * 70)
    print("Test 1: Audio Encoding (Mimi)")
    print("-" * 70)
    with torch.no_grad():
        codes = model.mimi.encode(waveform)
    print(f"Encoded codes shape: {codes.shape}")
    print(f"Codes dtype: {codes.dtype}")
    print(f"Codes device: {codes.device}")
    print(f"Codes range: [{codes.min().item():.2f}, {codes.max().item():.2f}]")
    print()

    # Test 2: Forward through Moshi LM
    print("-" * 70)
    print("Test 2: Moshi LM Forward Pass")
    print("-" * 70)
    try:
        with torch.no_grad():
            outputs = model.moshi_lm(codes)

        print(f"Output type: {type(outputs)}")

        # Try to inspect outputs
        if hasattr(outputs, 'logits'):
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Logits dtype: {outputs.logits.dtype}")
        elif isinstance(outputs, torch.Tensor):
            print(f"Output tensor shape: {outputs.shape}")
            print(f"Output tensor dtype: {outputs.dtype}")
        else:
            print(f"Output attributes: {dir(outputs)}")

        print()

    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 3: Check model architecture
    print("-" * 70)
    print("Test 3: Model Architecture Info")
    print("-" * 70)
    print(f"Mimi type: {type(model.mimi)}")
    print(f"Moshi LM type: {type(model.moshi_lm)}")
    print()

    # Try to get model info
    try:
        print("Mimi attributes:")
        mimi_attrs = [attr for attr in dir(model.mimi) if not attr.startswith('_')]
        for attr in mimi_attrs[:20]:  # Show first 20
            print(f"  - {attr}")
        print()

        print("Moshi LM attributes:")
        moshi_attrs = [attr for attr in dir(model.moshi_lm) if not attr.startswith('_')]
        for attr in moshi_attrs[:20]:  # Show first 20
            print(f"  - {attr}")
        print()

    except Exception as e:
        print(f"Error inspecting model: {e}")
        print()

    # Test 4: Check if model has generation capability
    print("-" * 70)
    print("Test 4: Generation Capability")
    print("-" * 70)
    if hasattr(model.moshi_lm, 'generate'):
        print("✓ Model has 'generate' method")
        try:
            # Try generation
            print("Attempting generation...")
            with torch.no_grad():
                generated = model.moshi_lm.generate(codes, max_length=10)
            print(f"Generated output shape: {generated.shape}")
        except Exception as e:
            print(f"Generation failed: {e}")
    else:
        print("✗ Model does not have 'generate' method")
    print()

    print("=" * 70)
    print("Test completed!")
    print("=" * 70)


def main():
    """Main function"""
    # Default paths
    model_path = "/data1/lixiang/Moshi/moshiko-pytorch-bf16/kyutai/moshiko-pytorch-bf16"

    # Check if audio path provided
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Use a sample from RAVDESS if available
        audio_path = "/data1/lixiang/OpenS2S_dataset/RAVDESS/Actor_01/03-01-04-01-01-01-01.wav"
        print(f"No audio path provided, using default: {audio_path}")
        print()

    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        print("Usage: python test_moshi.py <audio_path>")
        sys.exit(1)

    # Run test
    test_moshi_inference(model_path, audio_path)


if __name__ == "__main__":
    main()
