#!/usr/bin/env python3
"""Test script to observe model's thinking process during emotion recognition."""

import sys
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, build_inputs

def test_with_thinking(audio_path: Path, model, tokenizer):
    """Test emotion recognition and capture generation process."""
    print(f"\n{'='*80}")
    print(f"Testing: {audio_path.name}")
    print(f"{'='*80}")

    # Load audio
    waveform, sr = torchaudio.load(str(audio_path))
    waveform = waveform.to(cfg.device)
    print(f"Audio: sr={sr}, duration={waveform.shape[1]/sr:.2f}s")

    # Test with first emotion prompt
    prompt = cfg.emo_prompts[0]
    print(f"\nPrompt: \"{prompt}\"")

    # Build inputs
    inputs = build_inputs(waveform, sr, prompt, tokenizer, cfg.device)
    print(f"\nInput shapes:")
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  speech_values: {inputs['speech_values'].shape}")
    print(f"  speech_mask: {inputs['speech_mask'].shape}")

    # Show input_ids (skip decoding due to special audio token)
    print(f"\nInput token IDs (first 20):")
    print(f"  {inputs['input_ids'][0][:20].tolist()}")

    # Generate with thinking enabled (if supported)
    print(f"\n{'='*80}")
    print("GENERATION PROCESS")
    print(f"{'='*80}")

    from transformers import GenerationConfig

    # Try generation with output_scores to see token probabilities
    gen_config = GenerationConfig(
        max_new_tokens=50,  # Generate more tokens to see thinking
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    print("Generating (max 50 tokens)...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            speech_values=inputs.get("speech_values"),
            speech_mask=inputs.get("speech_mask"),
            spk_emb=None,
            generation_config=gen_config,
        )

    # Extract generated tokens
    if hasattr(outputs, 'sequences'):
        generated_ids = outputs.sequences[0]
    else:
        generated_ids = outputs[0]

    print(f"\nGenerated {len(generated_ids)} tokens")

    # Decode step by step
    print(f"\n{'='*80}")
    print("TOKEN-BY-TOKEN GENERATION")
    print(f"{'='*80}")

    for i, token_id in enumerate(generated_ids[:30]):  # Show first 30 tokens
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"Token {i:2d}: id={token_id:6d} -> '{token_text}'")

    if len(generated_ids) > 30:
        print(f"... ({len(generated_ids) - 30} more tokens)")

    # Full decoded output
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n{'='*80}")
    print("FULL OUTPUT")
    print(f"{'='*80}")
    print(full_output)

    # Check if thinking tokens are present
    if '<think>' in full_output or 'thinking' in full_output.lower():
        print("\n✓ Model used thinking process")
    else:
        print("\n✗ No explicit thinking tokens found")

    # Show top-5 token probabilities for first generated token (if available)
    if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
        print(f"\n{'='*80}")
        print("TOP-5 TOKEN PROBABILITIES (First Generated Token)")
        print(f"{'='*80}")
        first_token_logits = outputs.scores[0][0]  # [vocab_size]
        probs = torch.softmax(first_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5)

        for prob, idx in zip(top_probs, top_indices):
            token_text = tokenizer.decode([idx], skip_special_tokens=False)
            print(f"  {prob:.4f} -> '{token_text}' (id={idx})")

def main():
    print("Loading model...")
    model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
    model.eval()

    # Get 5 sad audio files
    sad_dir = Path("/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad")
    audio_files = list(sad_dir.rglob("*.wav"))[:5]

    print(f"\n{'#'*80}")
    print(f"# Testing {len(audio_files)} Sad Audio Samples")
    print(f"# Config: sr={cfg.sample_rate}, n_fft={cfg.n_fft}, hop={cfg.hop_length}")
    print(f"{'#'*80}")

    for audio_path in audio_files:
        try:
            test_with_thinking(audio_path, model, tokenizer)
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'#'*80}")
    print("# Test Complete")
    print(f"{'#'*80}")

if __name__ == "__main__":
    main()
