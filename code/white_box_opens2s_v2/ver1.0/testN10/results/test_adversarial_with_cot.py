#!/usr/bin/env python3
"""
Test adversarial samples with OpenS2S and capture Chain-of-Thought reasoning.
"""

import os
import sys
import json
import torch
import torchaudio
from pathlib import Path

# Add OpenS2S to path
sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from src.modeling_omnispeech import OmniSpeechForConditionalGeneration
from transformers import AutoTokenizer

def load_model(model_path, device='cuda:0'):
    """Load OpenS2S model and tokenizer."""
    print(f"Loading model from {model_path}...")
    model = OmniSpeechForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    return model, tokenizer

def test_audio_with_cot(model, tokenizer, audio_path, device='cuda:0'):
    """Test audio with Chain-of-Thought prompting."""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    # Test with multiple prompts to capture reasoning
    prompts = [
        # Prompt 1: Direct emotion classification
        "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",

        # Prompt 2: Encourage reasoning
        "Listen to this audio carefully. First, describe what you hear in terms of tone, pitch, and energy. Then classify the emotion as one of: happy, sad, angry, neutral, surprise.",

        # Prompt 3: Step-by-step analysis
        "Analyze this audio step by step:\n1. What is the speaker's tone?\n2. What is the emotional quality?\n3. Final emotion classification (happy/sad/angry/neutral/surprise):",
    ]

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Testing with Prompt {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        # Prepare messages in multimodal format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Prepare audio inputs
        audio_inputs = {
            'speech_values': waveform.unsqueeze(0),
            'speech_mask': torch.ones(1, waveform.shape[1], dtype=torch.long, device=device)
        }

        # Merge inputs
        inputs.update(audio_inputs)

        # Generate with higher max_new_tokens to capture full reasoning
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Allow longer responses
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (after the prompt)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()

        print(f"\nModel Response:")
        print(response)

        results.append({
            'prompt': prompt,
            'response': response
        })

    return results

def main():
    # Configuration
    model_path = '/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S'
    results_dir = Path('/data1/lixiang/lx_code/white_box_v2/codex/testN10/results')
    output_file = results_dir / 'cot_analysis.json'
    device = 'cuda:0'

    # Load model
    model, tokenizer = load_model(model_path, device)

    # Find all adversarial audio files
    audio_files = sorted(results_dir.glob('*.wav'))

    print(f"\nFound {len(audio_files)} adversarial audio files")

    all_results = {}

    # Test each audio file
    for audio_file in audio_files:
        sample_id = audio_file.stem
        print(f"\n{'#'*80}")
        print(f"Testing sample: {sample_id}")
        print(f"{'#'*80}")

        # Load metadata
        json_file = audio_file.with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            print(f"Original emotion: {metadata.get('path', '').split('/')[-2]}")
            print(f"Attack target: happy")
            print(f"Attack success: {metadata.get('success_emo', False)}")

        # Test with CoT prompts
        results = test_audio_with_cot(model, tokenizer, str(audio_file), device)

        all_results[sample_id] = {
            'audio_file': str(audio_file),
            'metadata': metadata if json_file.exists() else None,
            'cot_results': results
        }

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("Done!")

if __name__ == '__main__':
    main()
