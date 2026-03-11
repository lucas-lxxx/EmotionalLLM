#!/usr/bin/env python3
"""
Analyze Chain-of-Thought results from adversarial sample testing.
"""

import json
import re
from pathlib import Path
from collections import Counter

def extract_emotion(response):
    """Extract emotion from model response."""
    response_lower = response.lower().strip()

    # Direct emotion words
    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise']

    # Check for direct emotion word at the end or as standalone
    for emotion in emotions:
        if response_lower.endswith(emotion) or response_lower == emotion:
            return emotion

    # Check for emotion word after </think>
    if '</think>' in response_lower:
        after_think = response_lower.split('</think>')[-1].strip()
        for emotion in emotions:
            if emotion in after_think:
                return emotion

    # Check anywhere in response
    for emotion in emotions:
        if emotion in response_lower:
            return emotion

    return "unknown"

def main():
    results_file = Path('cot_analysis.json')

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("="*80)
    print("对抗样本情绪识别分析报告")
    print("="*80)

    total_samples = len(data)
    print(f"\n总样本数: {total_samples}")

    # Analyze each prompt type
    prompt_names = [
        "Prompt 1: 直接情绪分类",
        "Prompt 2: 描述+分类",
        "Prompt 3: 逐步分析"
    ]

    for prompt_idx in range(3):
        print(f"\n{'='*80}")
        print(f"{prompt_names[prompt_idx]}")
        print(f"{'='*80}")

        emotion_counts = Counter()
        happy_count = 0
        attack_success_count = 0

        for sample_id, sample_data in data.items():
            metadata = sample_data.get('metadata', {})
            attack_success = metadata.get('success_emo', False)

            if attack_success:
                attack_success_count += 1

            cot_results = sample_data.get('cot_results', [])
            if prompt_idx < len(cot_results):
                response = cot_results[prompt_idx]['response']
                emotion = extract_emotion(response)
                emotion_counts[emotion] += 1

                if emotion == 'happy':
                    happy_count += 1

                # Print individual result
                original_emotion = metadata.get('path', '').split('/')[-2] if metadata else 'Unknown'
                print(f"\n样本 {sample_id}:")
                print(f"  原始情绪: {original_emotion}")
                print(f"  攻击成功(训练时): {attack_success}")
                print(f"  识别情绪: {emotion}")
                if len(response) > 100:
                    print(f"  响应摘要: {response[:100]}...")
                else:
                    print(f"  响应: {response}")

        print(f"\n{'-'*80}")
        print(f"统计结果:")
        print(f"  攻击成功样本数(训练时): {attack_success_count}/{total_samples}")
        print(f"  识别为 happy: {happy_count}/{total_samples} ({happy_count/total_samples*100:.1f}%)")
        print(f"\n  情绪分布:")
        for emotion, count in emotion_counts.most_common():
            print(f"    {emotion}: {count} ({count/total_samples*100:.1f}%)")

    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print(f"训练时标记为攻击成功的样本: {attack_success_count}/{total_samples}")
    print(f"\n关键发现:")
    print(f"1. 对抗样本在不同提示下的识别结果存在差异")
    print(f"2. 模型的 <think> 标签显示了其推理过程")
    print(f"3. 部分样本被识别为 happy，部分被识别为其他情绪（如 neutral, surprise）")

if __name__ == '__main__':
    main()
