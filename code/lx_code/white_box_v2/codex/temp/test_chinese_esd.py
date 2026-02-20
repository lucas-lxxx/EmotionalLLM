#!/usr/bin/env python3
"""Test OpenS2S on Chinese ESD dataset (Angry emotion)."""

import sys
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, '/data1/lixiang/Opens2s/OpenS2S')

from config import cfg
from opens2s_io import load_opens2s, build_inputs

# Ground truth labels from 0001.txt
GROUND_TRUTH = {
    '0001_000351': '打远一看，它们的确很是美丽，',
    '0001_000352': '英国的哲学家曾经说过"',
    '0001_000353': '我老家在北京，哇塞！太精彩了。',
    '0001_000354': '不管怎么说主队好象是志在夺魁。',
    '0001_000355': '我们乘船漂游了三峡，真是刺激。',
}

def test_chinese_audio(audio_path: Path, ground_truth: str, model, tokenizer):
    """Test model on Chinese audio without prompt."""
    print(f"\n{'='*80}")
    print(f"Testing: {audio_path.name}")
    print(f"{'='*80}")
    print(f"Ground Truth Text: {ground_truth}")
    print(f"Ground Truth Emotion: 生气 (Angry)")

    # Load audio
    waveform, sr = torchaudio.load(str(audio_path))
    waveform = waveform.to(cfg.device)
    print(f"\nAudio: sr={sr}, duration={waveform.shape[1]/sr:.2f}s")

    # Test 1: No prompt (natural response)
    print(f"\n{'='*80}")
    print("TEST 1: NO PROMPT - Natural Response")
    print(f"{'='*80}")

    inputs = build_inputs(waveform, sr, "", tokenizer, cfg.device)

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            speech_values=inputs.get("speech_values"),
            speech_mask=inputs.get("speech_mask"),
            spk_emb=None,
            generation_config=gen_config,
        )

    if hasattr(outputs, 'sequences'):
        generated_ids = outputs.sequences[0]
    else:
        generated_ids = outputs[0]

    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Model Output: {full_output}")

    # Show first few tokens
    print(f"\nFirst 10 tokens:")
    for i, token_id in enumerate(generated_ids[:10]):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"  Token {i}: '{token_text}'")

    # Test 2: ASR prompt
    print(f"\n{'='*80}")
    print("TEST 2: ASR PROMPT")
    print(f"{'='*80}")

    asr_prompt = "请转录这段语音的内容。"
    inputs_asr = build_inputs(waveform, sr, asr_prompt, tokenizer, cfg.device)

    with torch.no_grad():
        outputs_asr = model.generate(
            input_ids=inputs_asr["input_ids"],
            attention_mask=inputs_asr.get("attention_mask"),
            speech_values=inputs_asr.get("speech_values"),
            speech_mask=inputs_asr.get("speech_mask"),
            spk_emb=None,
            generation_config=gen_config,
        )

    if hasattr(outputs_asr, 'sequences'):
        generated_ids_asr = outputs_asr.sequences[0]
    else:
        generated_ids_asr = outputs_asr[0]

    asr_output = tokenizer.decode(generated_ids_asr, skip_special_tokens=True)
    print(f"Prompt: {asr_prompt}")
    print(f"Model Output: {asr_output}")

    # Test 3: Emotion prompt
    print(f"\n{'='*80}")
    print("TEST 3: EMOTION PROMPT")
    print(f"{'='*80}")

    emo_prompt = "这段语音的情绪是什么？请从以下选项中选择一个：高兴、生气、悲伤、中立。"
    inputs_emo = build_inputs(waveform, sr, emo_prompt, tokenizer, cfg.device)

    with torch.no_grad():
        outputs_emo = model.generate(
            input_ids=inputs_emo["input_ids"],
            attention_mask=inputs_emo.get("attention_mask"),
            speech_values=inputs_emo.get("speech_values"),
            speech_mask=inputs_emo.get("speech_mask"),
            spk_emb=None,
            generation_config=gen_config,
        )

    if hasattr(outputs_emo, 'sequences'):
        generated_ids_emo = outputs_emo.sequences[0]
    else:
        generated_ids_emo = outputs_emo[0]

    emo_output = tokenizer.decode(generated_ids_emo, skip_special_tokens=True)
    print(f"Prompt: {emo_prompt}")
    print(f"Model Output: {emo_output}")

    # Check if emotion is correct
    if '生气' in emo_output or 'angry' in emo_output.lower():
        print("✓ Emotion CORRECT!")
    else:
        print("✗ Emotion INCORRECT")

    return {
        'file': audio_path.name,
        'ground_truth': ground_truth,
        'no_prompt': full_output,
        'asr_output': asr_output,
        'emotion_output': emo_output,
    }

def main():
    print("Loading model...")
    model, tokenizer = load_opens2s(cfg.model_path, cfg.device, cfg.opens2s_root)
    model.eval()

    # Test first 5 Angry samples
    angry_dir = Path("/data1/lixiang/ESD/0001/Angry")
    test_files = [
        '0001_000351.wav',
        '0001_000352.wav',
        '0001_000353.wav',
        '0001_000354.wav',
        '0001_000355.wav',
    ]

    print(f"\n{'#'*80}")
    print(f"# Testing OpenS2S on Chinese ESD Dataset (Angry Emotion)")
    print(f"# Speaker: 0001")
    print(f"# Samples: {len(test_files)}")
    print(f"# Config: sr={cfg.sample_rate}, n_fft={cfg.n_fft}, hop={cfg.hop_length}")
    print(f"{'#'*80}")

    results = []
    for filename in test_files:
        audio_path = angry_dir / filename
        file_id = filename.replace('.wav', '')
        ground_truth = GROUND_TRUTH.get(file_id, "Unknown")

        try:
            result = test_chinese_audio(audio_path, ground_truth, model, tokenizer)
            results.append(result)
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
        print(f"  Ground Truth: {result['ground_truth']}")
        print(f"  ASR Output:   {result['asr_output'][:50]}...")
        print(f"  Emotion:      {result['emotion_output'][:50]}...")

    print(f"\n{'#'*80}")
    print("# Test Complete")
    print(f"{'#'*80}")

if __name__ == "__main__":
    main()
