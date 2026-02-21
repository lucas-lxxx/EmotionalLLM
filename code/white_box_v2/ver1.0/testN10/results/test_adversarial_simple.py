#!/usr/bin/env python3
"""
Test adversarial samples with OpenS2S - simplified version focusing on text output.
"""

import os
import sys
import json
import torch
from pathlib import Path
from copy import deepcopy

# Add OpenS2S to path
sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from src.modeling_omnispeech import OmniSpeechModel
from src.feature_extraction_audio import WhisperFeatureExtractor
from src.utils import get_waveform
from src.constants import (
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_TTS_START_TOKEN,
    DEFAULT_AUDIO_TOKEN,
    AUDIO_TOKEN_INDEX
)
from transformers import AutoTokenizer, GenerationConfig

def load_model(model_path):
    """Load OpenS2S model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    model = OmniSpeechModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.cuda()
    audio_extractor = WhisperFeatureExtractor.from_pretrained(os.path.join(model_path, "audio"))
    print("Model loaded.")
    return model, tokenizer, audio_extractor, generation_config

def prepare_inputs(messages, tokenizer, audio_extractor, system_prompt="You are a helpful assistant."):
    """Prepare inputs for the model."""
    new_messages = []
    audios = []

    if system_prompt:
        new_messages.append({"role": "system", "content": system_prompt})

    for turn in messages:
        role = turn["role"]
        content = turn["content"]
        new_content = ""

        if isinstance(content, list):
            for item in content:
                if "audio" in item and item["audio"]:
                    waveform = get_waveform(item["audio"])
                    audios.append(waveform)
                    new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                elif "text" in item and item["text"]:
                    new_content += item["text"]

        new_messages.append({"role": role, "content": new_content})

    prompt = tokenizer.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False)

    # Split and encode with audio tokens
    segments = prompt.split(f"{DEFAULT_AUDIO_TOKEN}")
    input_ids = []
    for idx, segment in enumerate(segments):
        if idx != 0:
            input_ids += [AUDIO_TOKEN_INDEX]
        input_ids += tokenizer.encode(segment)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0)

    if audios:
        speech_inputs = audio_extractor(
            audios,
            sampling_rate=audio_extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )
        speech_values = speech_inputs.input_features
        speech_mask = speech_inputs.attention_mask
    else:
        speech_values, speech_mask = None, None

    return input_ids, speech_values, speech_mask

@torch.inference_mode()
def generate_text_only(model, tokenizer, audio_extractor, messages, generation_config):
    """Generate text response only (no TTS)."""
    input_ids, speech_values, speech_mask = prepare_inputs(messages, tokenizer, audio_extractor)

    input_ids = input_ids.to(device='cuda', non_blocking=True)
    if speech_values is not None:
        speech_values = speech_values.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
        speech_mask = speech_mask.to(device='cuda', non_blocking=True)

    gen_config = deepcopy(generation_config)
    gen_config.update(
        max_new_tokens=512,
        do_sample=False,  # Greedy decoding for consistency
        temperature=1.0,
        top_p=1.0
    )

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=None,
        speech_values=speech_values,
        speech_mask=speech_mask,
        spk_emb=None,
        units_gen=False,  # Don't generate TTS units
        generation_config=gen_config,
        use_cache=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Configuration
    model_path = '/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S'
    results_dir = Path('/data1/lixiang/lx_code/white_box_v2/codex/testN10/results')
    output_file = results_dir / 'cot_analysis.json'

    # Load model
    model, tokenizer, audio_extractor, generation_config = load_model(model_path)

    # Find all adversarial audio files
    audio_files = sorted(results_dir.glob('*.wav'))
    print(f"\nFound {len(audio_files)} adversarial audio files\n")

    # Test prompts
    prompts = [
        "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",
        "Listen to this audio carefully. First, describe what you hear in terms of tone, pitch, and energy. Then classify the emotion as one of: happy, sad, angry, neutral, surprise.",
        "Analyze this audio step by step:\n1. What is the speaker's tone?\n2. What is the emotional quality?\n3. Final emotion classification (happy/sad/angry/neutral/surprise):",
    ]

    all_results = {}

    # Test each audio file
    for audio_file in audio_files:
        sample_id = audio_file.stem
        print(f"\n{'#'*80}")
        print(f"Testing sample: {sample_id}")
        print(f"{'#'*80}")

        # Load metadata
        json_file = audio_file.with_suffix('.json')
        metadata = None
        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            print(f"Original emotion: {metadata.get('path', '').split('/')[-2]}")
            print(f"Attack target: happy")
            print(f"Attack success: {metadata.get('success_emo', False)}")

        results = []

        for i, prompt in enumerate(prompts):
            print(f"\n{'-'*60}")
            print(f"Prompt {i+1}: {prompt[:50]}...")
            print(f"{'-'*60}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"audio": str(audio_file)},
                        {"text": prompt}
                    ]
                }
            ]

            try:
                response = generate_text_only(model, tokenizer, audio_extractor, messages, generation_config)

                # Extract assistant response
                if "<|im_start|>assistant" in response:
                    response = response.split("<|im_start|>assistant")[-1].strip()
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()

                print(f"Response: {response}")

                results.append({
                    'prompt': prompt,
                    'response': response
                })
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'prompt': prompt,
                    'response': f"ERROR: {str(e)}"
                })

        all_results[sample_id] = {
            'audio_file': str(audio_file),
            'metadata': metadata,
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
