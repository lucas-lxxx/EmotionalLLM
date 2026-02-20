#!/usr/bin/env python3
"""
Decode Moshi text tokens to see what it said
"""
import sys
import torch
from pathlib import Path
import sentencepiece

from moshi_io import load_audio
from moshi.models import loaders, LMGen


def decode_moshi_response(model_path: str, audio_path: str, device: str = "cuda"):
    """
    Run Moshi inference and decode text response
    """
    print("=" * 70)
    print("Moshi Response Decoder")
    print("=" * 70)
    print(f"Audio: {audio_path}")
    print()

    # Load text tokenizer
    print("Loading text tokenizer...")
    tokenizer_path = f"{model_path}/tokenizer_spm_32k_3.model"
    text_tokenizer = sentencepiece.SentencePieceProcessor()
    text_tokenizer.load(tokenizer_path)
    print(f"✓ Tokenizer loaded (vocab size: {text_tokenizer.vocab_size()})")
    print()

    # Load models
    print("Loading models...")
    mimi_path = f"{model_path}/tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi = loaders.get_mimi(mimi_path, device=device)
    mimi.set_num_codebooks(8)

    moshi_lm_path = f"{model_path}/model.safetensors"
    lm = loaders.get_moshi_lm(moshi_lm_path, device=device)

    lm_gen = LMGen(lm, temp=0.8, temp_text=0.7)
    print("✓ Models loaded")
    print()

    # Load audio
    print("Loading audio...")
    waveform, sr = load_audio(audio_path, device=device)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    # Pad if needed
    audio_length = waveform.shape[-1]
    if audio_length % frame_size != 0:
        pad_length = frame_size - (audio_length % frame_size)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    print(f"✓ Audio loaded: {waveform.shape[-1]} samples")
    print()

    # Set streaming mode
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # Process audio
    print("Processing audio and collecting text...")
    print("-" * 70)

    num_frames = waveform.shape[-1] // frame_size
    text_tokens = []
    text_pieces = []

    with torch.no_grad():
        for frame_idx in range(num_frames):
            start = frame_idx * frame_size
            end = start + frame_size
            frame = waveform[:, :, start:end]

            codes = mimi.encode(frame)

            for c in range(codes.shape[-1]):
                code_slice = codes[:, :, c: c + 1]
                tokens = lm_gen.step(code_slice)

                if tokens is None:
                    continue

                # Extract text token
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):  # Skip special tokens
                    text_tokens.append(text_token)
                    # Decode immediately
                    text_piece = text_tokenizer.id_to_piece(text_token)
                    text_piece = text_piece.replace("▁", " ")
                    text_pieces.append(text_piece)
                    print(f"  Token {text_token:5d} -> '{text_piece}'")

    print("-" * 70)
    print()

    # Combine text
    full_text = "".join(text_pieces)
    print("=" * 70)
    print("Moshi's Response:")
    print("=" * 70)
    print(f"'{full_text}'")
    print()
    print(f"Total tokens: {len(text_tokens)}")
    print("=" * 70)


def main():
    model_path = "/data1/lixiang/Moshi/moshiko-pytorch-bf16/kyutai/moshiko-pytorch-bf16"

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "/data1/lixiang/OpenS2S_dataset/RAVDESS/Actor_01/03-01-04-01-01-01-01.wav"

    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    decode_moshi_response(model_path, audio_path)


if __name__ == "__main__":
    main()
