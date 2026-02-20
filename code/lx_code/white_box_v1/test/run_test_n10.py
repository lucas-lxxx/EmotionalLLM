"""
测试脚本：对10个Sad音频样本进行白盒攻击
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils_audio import (
    load_model, load_audio_extractor, load_waveform,
    clean_inference, attack_inference, run_attack
)

# 配置
OMNISPEECH_PATH = "/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S"
CHECKPOINT_PATH = "checkpoints/sad_happy_classifier.pt"
SAMPLE_LIST_PATH = "test/sad_samples_10.txt"
OUTPUT_DIR = "test/results_n10"
DEVICE = "cuda:0"

# 攻击参数
EPSILON = 0.002
STEPS = 30
LAMBDA_EMO = 1.0
LAMBDA_SEM = 1e-2
LAMBDA_PER = 1e-4

# Prompt：只输出情绪标签
PROMPT = "What is the emotion of this audio? Please answer with only one word: the emotion label (happy, sad, angry, or neutral)."

def main():
    print("=" * 80)
    print("OpenS2S White-Box Attack Test (N=10)")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio" / "clean").mkdir(parents=True, exist_ok=True)
    (output_dir / "audio" / "adv").mkdir(parents=True, exist_ok=True)
    (output_dir / "text").mkdir(parents=True, exist_ok=True)

    # 加载样本列表
    with open(SAMPLE_LIST_PATH, 'r') as f:
        audio_paths = [line.strip() for line in f if line.strip()]

    print(f"\nLoaded {len(audio_paths)} audio samples")
    print(f"Output directory: {output_dir}")
    print(f"\nAttack parameters:")
    print(f"  - epsilon: {EPSILON}")
    print(f"  - steps: {STEPS}")
    print(f"  - lambda_emo: {LAMBDA_EMO}")
    print(f"  - lambda_sem: {LAMBDA_SEM}")
    print(f"  - lambda_per: {LAMBDA_PER}")
    print(f"\nPrompt: {PROMPT}")
    print("=" * 80)

    # 加载模型
    print("\nLoading OpenS2S model...")
    model, tokenizer = load_model(OMNISPEECH_PATH, device=DEVICE)
    audio_extractor = load_audio_extractor(OMNISPEECH_PATH)

    # 加载情绪分类器
    print(f"Loading emotion classifier from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    from utils.emotion_classifier import FrozenEmotionClassifier

    emotion_label_to_idx = checkpoint['label_to_idx']
    input_dim = checkpoint['svd_components'].shape[0] if 'svd_components' in checkpoint else checkpoint['input_dim']
    num_emotions = len(emotion_label_to_idx)

    emotion_classifier = FrozenEmotionClassifier(input_dim, num_emotions)
    emotion_classifier.load_state_dict(checkpoint['classifier'])
    emotion_classifier.to(DEVICE)
    emotion_classifier.freeze()

    print(f"  Emotion mapping: {emotion_label_to_idx}")
    print(f"  Input dim: {input_dim}, Num emotions: {num_emotions}")

    # 保存配置
    config = {
        "omnispeech_path": OMNISPEECH_PATH,
        "checkpoint_path": CHECKPOINT_PATH,
        "num_samples": len(audio_paths),
        "epsilon": EPSILON,
        "steps": STEPS,
        "lambda_emo": LAMBDA_EMO,
        "lambda_sem": LAMBDA_SEM,
        "lambda_per": LAMBDA_PER,
        "prompt": PROMPT,
        "device": DEVICE,
        "emotion_label_to_idx": emotion_label_to_idx
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # 运行攻击
    results = []
    print("\n" + "=" * 80)
    print("Running attacks...")
    print("=" * 80)

    for idx, audio_path in enumerate(tqdm(audio_paths, desc="Processing")):
        sample_id = f"sample_{idx+1:03d}"
        print(f"\n[{idx+1}/{len(audio_paths)}] Processing: {Path(audio_path).name}")

        try:
            # Clean inference
            print("  [1/3] Clean inference...")
            clean_text = clean_inference(
                model, tokenizer, audio_extractor,
                audio_path, PROMPT, DEVICE
            )
            print(f"    Clean output: {clean_text[:100]}...")

            # 运行攻击
            print(f"  [2/3] Running attack...")
            attack_start = time.time()
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
                checkpoint=checkpoint,
                epsilon=EPSILON,
                steps=STEPS,
                alpha=EPSILON / 10.0,
                lambda_emo=LAMBDA_EMO,
                lambda_sem=LAMBDA_SEM,
                lambda_per=LAMBDA_PER,
                device=DEVICE
            )

            # 保存对抗音频
            adv_audio_path = output_dir / "audio" / "adv" / f"{sample_id}.wav"
            sf.write(str(adv_audio_path), waveform_adv.cpu().numpy(), sample_rate)

            # 创建clean音频软链接
            clean_audio_link = output_dir / "audio" / "clean" / f"{sample_id}.wav"
            if not clean_audio_link.exists():
                os.symlink(os.path.abspath(audio_path), clean_audio_link)

            # Attack inference
            print(f"  [3/3] Attack inference...")
            attack_text = attack_inference(
                model, tokenizer, audio_extractor,
                str(adv_audio_path), PROMPT, DEVICE
            )
            print(f"    Attack output: {attack_text[:100]}...")

            # 记录结果
            result = {
                'sample_id': sample_id,
                'audio_path': audio_path,
                'clean_text': clean_text,
                'attack_text': attack_text,
                'attack_time': attack_time,
                'metrics': metrics
            }
            results.append(result)

            # 保存文本
            with open(output_dir / "text" / f"{sample_id}_clean.txt", 'w') as f:
                f.write(clean_text)
            with open(output_dir / "text" / f"{sample_id}_attack.txt", 'w') as f:
                f.write(attack_text)

            print(f"  ✓ Completed in {attack_time:.2f}s")
            print(f"    Metrics: linf={metrics.get('linf', 'N/A'):.6f}, "
                  f"l2={metrics.get('l2', 'N/A'):.6f}, "
                  f"snr={metrics.get('snr_db', 'N/A'):.2f}dB")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'sample_id': sample_id,
                'audio_path': audio_path,
                'error': str(e)
            })

    # 保存结果
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)

    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # 统计
    success_count = sum(1 for r in results if 'error' not in r)
    print(f"\nSummary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(results) - success_count}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
