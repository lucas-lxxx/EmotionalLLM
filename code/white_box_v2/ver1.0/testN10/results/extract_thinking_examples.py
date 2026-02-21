#!/usr/bin/env python3
"""
Extract specific thinking process examples from CoT analysis.
"""

import json
from pathlib import Path

def main():
    results_file = Path('cot_analysis.json')

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("="*80)
    print("模型思考过程示例分析")
    print("="*80)

    # Example 1: Sample identified as "surprise" instead of "happy"
    sample_id = "00000_0001_000351"
    print(f"\n示例 1: {sample_id}")
    print(f"原始情绪: Angry → 攻击目标: Happy → 实际识别: Surprise")
    print("-"*80)

    response = data[sample_id]['cot_results'][0]['response']

    # Extract thinking part
    if '<think>' in response and '</think>' in response:
        think_part = response.split('<think>')[1].split('</think>')[0]
        print("模型的思考过程:")
        print(think_part[:500] + "..." if len(think_part) > 500 else think_part)

    # Example 2: Sample identified as "happy"
    sample_id = "00001_0001_000352"
    print(f"\n{'='*80}")
    print(f"示例 2: {sample_id}")
    print(f"原始情绪: Angry → 攻击目标: Happy → 实际识别: Happy ✓")
    print("-"*80)

    response = data[sample_id]['cot_results'][0]['response']

    if '<think>' in response and '</think>' in response:
        think_part = response.split('<think>')[1].split('</think>')[0]
        print("模型的思考过程:")
        print(think_part[:500] + "..." if len(think_part) > 500 else think_part)

    # Example 3: Sample identified as "neutral"
    sample_id = "00006_0004_000351"
    print(f"\n{'='*80}")
    print(f"示例 3: {sample_id}")
    print(f"原始情绪: Angry → 攻击目标: Happy → 实际识别: Neutral")
    print("-"*80)

    response = data[sample_id]['cot_results'][0]['response']

    if '<think>' in response and '</think>' in response:
        think_part = response.split('<think>')[1].split('</think>')[0]
        print("模型的思考过程:")
        print(think_part[:500] + "..." if len(think_part) > 500 else think_part)

if __name__ == '__main__':
    main()
