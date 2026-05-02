#!/usr/bin/env python3
"""测试 OpenS2S 模型是否正常工作

该脚本从数据集中随机选择音频文件，使用 OpenS2S 模型进行推理，
验证模型是否能够正常加载和运行。
"""

import argparse
import os
import sys
from pathlib import Path
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GenerationConfig
from tqdm import tqdm

from utils_audio import load_model, load_audio_extractor, load_waveform
from constants import (
    DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_TOKEN,
    DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN, AUDIO_TOKEN_INDEX
)


def collect_audio_files(data_root: str, max_files: int = 10) -> list:
    """从数据集中收集音频文件"""
    audio_files = []
    data_path = Path(data_root)

    if not data_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"Scanning for audio files in {data_root}...")

    # 遍历所有情绪目录
    for emotion_dir in data_path.glob("*"):
        if not emotion_dir.is_dir():
            continue

        # 遍历 age/gender 子目录
        for subdir in emotion_dir.rglob("*.wav"):
            audio_files.append(str(subdir))

            if len(audio_files) >= max_files:
                return audio_files

    print(f"Found {len(audio_files)} audio files")
    return audio_files


def run_inference(
    model,
    tokenizer,
    audio_extractor,
    audio_path: str,
    prompt: str = "What is the emotion of this audio? Please answer with only the emotion label (e.g., happy, sad, neutral).",
    device: str = "cuda:0"
) -> str:
    """对单个音频进行推理"""
    # 准备输入
    waveform, sample_rate = load_waveform(audio_path)
    wave_np = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform

    inputs = audio_extractor(
        [wave_np],
        sampling_rate=audio_extractor.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )
    speech_values = inputs.input_features.to(device)
    speech_mask = inputs.attention_mask.to(device)

    # 构建 prompt
    prompt_with_audio = DEFAULT_AUDIO_START_TOKEN + DEFAULT_AUDIO_TOKEN
    if prompt:
        prompt_with_audio += prompt
    prompt_with_audio += DEFAULT_AUDIO_END_TOKEN + DEFAULT_TTS_START_TOKEN

    # 编码输入
    segments = prompt_with_audio.split(DEFAULT_AUDIO_TOKEN)
    ids = []
    for idx, seg in enumerate(segments):
        if idx != 0:
            ids.append(AUDIO_TOKEN_INDEX)
        ids.extend(tokenizer.encode(seg))
    input_ids = torch.LongTensor(ids).unsqueeze(0).to(device)

    # 生成配置
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # 推理
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            audio_values=speech_values,
            audio_mask=speech_mask,
            generation_config=generation_config,
        )

    # 解码输出
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取 TTS token 之后的内容
    if DEFAULT_TTS_START_TOKEN in output_text:
        output_text = output_text.split(DEFAULT_TTS_START_TOKEN)[1]

    # 移除特殊的音频 token
    if DEFAULT_AUDIO_START_TOKEN in output_text:
        output_text = output_text.split(DEFAULT_AUDIO_START_TOKEN)[0]
    if DEFAULT_AUDIO_END_TOKEN in output_text:
        output_text = output_text.split(DEFAULT_AUDIO_END_TOKEN)[0]

    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Test OpenS2S model with random audio samples")
    parser.add_argument(
        "--omnispeech-path",
        type=str,
        default="/data1/lixiang/Opens2s/models/OpenS2S_ckpt",
        help="Path to OpenS2S model checkpoint"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data1/lixiang/OpenS2S_dataset/data/en_query_wav",
        help="Path to audio dataset root"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random audio samples to test"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the emotion of this audio? Please answer with only the emotion label (e.g., happy, sad, neutral).",
        help="Prompt for the model"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OpenS2S Model Test")
    print("=" * 80)

    # 检查设备
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"

    print(f"\nDevice: {args.device}")
    print(f"Model path: {args.omnispeech_path}")
    print(f"Data root: {args.data_root}")
    print(f"Number of samples: {args.num_samples}")

    # 加载模型
    print("\n" + "=" * 80)
    print("Loading OpenS2S model...")
    print("=" * 80)

    try:
        model, tokenizer = load_model(
            omnipath=args.omnispeech_path,
            device=args.device,
        )
        print(f"✓ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    # 加载音频提取器
    print("\nLoading audio extractor...")
    try:
        audio_extractor = load_audio_extractor(args.omnispeech_path)
        print("✓ Audio extractor loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load audio extractor: {e}")
        sys.exit(1)

    # 收集音频文件
    print("\n" + "=" * 80)
    print("Collecting audio files...")
    print("=" * 80)

    try:
        audio_files = collect_audio_files(args.data_root, max_files=args.num_samples * 2)

        if len(audio_files) == 0:
            print("✗ No audio files found in the dataset!")
            sys.exit(1)

        # 随机选择指定数量的文件
        selected_files = random.sample(audio_files, min(args.num_samples, len(audio_files)))
        print(f"✓ Selected {len(selected_files)} audio files for testing")
    except Exception as e:
        print(f"✗ Failed to collect audio files: {e}")
        sys.exit(1)

    # 运行推理测试
    print("\n" + "=" * 80)
    print("Running inference tests...")
    print("=" * 80)
    print(f"\nPrompt: {args.prompt}\n")

    results = []
    success_count = 0

    for idx, audio_path in enumerate(tqdm(selected_files, desc="Processing")):
        print(f"\n[{idx + 1}/{len(selected_files)}] Testing: {Path(audio_path).name}")
        print(f"Full path: {audio_path}")

        try:
            # 运行推理
            output_text = run_inference(
                model=model,
                tokenizer=tokenizer,
                audio_extractor=audio_extractor,
                audio_path=audio_path,
                prompt=args.prompt,
                device=args.device,
            )

            print(f"Output: {output_text}")
            results.append({
                "audio_path": audio_path,
                "output": output_text,
                "status": "success"
            })
            success_count += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "audio_path": audio_path,
                "output": "",
                "error": str(e),
                "status": "failed"
            })

    # 输出测试结果摘要
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total samples tested: {len(selected_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(selected_files) - success_count}")

    if success_count == len(selected_files):
        print("\n✓ All tests passed! The OpenS2S model is working correctly.")
    else:
        print(f"\n⚠ Some tests failed. Success rate: {success_count/len(selected_files)*100:.1f}%")

    # 显示详细结果
    print("\n" + "=" * 80)
    print("Detailed Results")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        status = "✓" if result["status"] == "success" else "✗"
        print(f"\n{i}. {status} {Path(result['audio_path']).name}")
        print(f"   Output: {result.get('output', result.get('error', 'No output'))}")


if __name__ == "__main__":
    main()
