#!/usr/bin/env python3
"""Test model's natural response to audio without any prompt."""

import sys
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, build_inputs

def test_no_prompt(audio_path: Path, model, tokenizer):
    """Test model's response to audio without prompt."""
    print(f"\n{'='*80}")
    print(f"Testing: {audio_path.name}")
    print(f"{'='*80}")

    # Load audio
    waveform, sr = torchaudio.load(str(audio_path))
    waveform = waveform.to(cfg.device)
    print(f"Audio: sr={sr}, duration={waveform.shape[1]/sr:.2f}s")

    # Build inputs with EMPTY prompt
    print(f"\n>>> NO PROMPT - Just audio <<<")
    inputs = build_inputs(waveform, sr, "", tokenizer, cfg.device)  # Empty prompt!

    print(f"\nInput shapes:")
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  speech_values: {inputs['speech_values'].shape}")

    # Generate with extended max tokens to see full thinking
    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=200,  # Allow longer generation
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    print(f"\n{'='*80}")
    print("MODEL'S NATURAL RESPONSE (No Prompt)")
    print(f"{'='*80}")
    print("Generating (max 200 tokens)...\n")

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

    print(f"Generated {len(generated_ids)} tokens\n")

    # Show token-by-token generation
    print(f"{'='*80}")
    print("TOKEN-BY-TOKEN GENERATION")
    print(f"{'='*80}")

    for i, token_id in enumerate(generated_ids[:50]):  # Show first 50 tokens
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        # Highlight special tokens
        if token_id in [151644, 151645, 151669, 151670, 151671, 151672]:
            print(f"Token {i:3d}: id={token_id:6d} -> '{token_text}' [SPECIAL]")
        else:
            print(f"Token {i:3d}: id={token_id:6d} -> '{token_text}'")

    if len(generated_ids) > 50:
        print(f"... ({len(generated_ids) - 50} more tokens)")

    # Full decoded output
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n{'='*80}")
    print("FULL OUTPUT (Special tokens removed)")
    print(f"{'='*80}")
    print(full_output)
    print(f"{'='*80}")

    # Show top-5 probabilities for first few tokens
    if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
        print(f"\n{'='*80}")
        print("TOP-5 TOKEN PROBABILITIES (First 3 Generated Tokens)")
        print(f"{'='*80}")

        for step in range(min(3, len(outputs.scores))):
            print(f"\n--- Token {step} ---")
            token_logits = outputs.scores[step][0]
            probs = torch.softmax(token_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5)

            for prob, idx in zip(top_probs, top_indices):
                token_text = tokenizer.decode([idx], skip_special_tokens=False)
                print(f"  {prob:.4f} -> '{token_text}' (id={idx})")

    return full_output

def main():
    print("Loading model...")
    model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
    model.eval()

    # Get 5 sad audio files
    sad_dir = Path("/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad")
    audio_files = list(sad_dir.rglob("*.wav"))[:5]

    print(f"\n{'#'*80}")
    print(f"# Testing Model's Natural Response (NO PROMPT)")
    print(f"# Samples: {len(audio_files)} Sad audio files")
    print(f"# Config: sr={cfg.sample_rate}, n_fft={cfg.n_fft}, hop={cfg.hop_length}")
    print(f"{'#'*80}")

    results = []
    for audio_path in audio_files:
        try:
            output = test_no_prompt(audio_path, model, tokenizer)
            results.append({
                'file': audio_path.name,
                'output': output
            })
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'#'*80}")
    print("# SUMMARY")
    print(f"{'#'*80}")
    for result in results:
        print(f"\n{result['file']}:")
        # Show first 100 chars
        output_preview = result['output'][:100] + "..." if len(result['output']) > 100 else result['output']
        print(f"  {output_preview}")

    print(f"\n{'#'*80}")
    print("# Test Complete")
    print(f"{'#'*80}")

if __name__ == "__main__":
    main()
