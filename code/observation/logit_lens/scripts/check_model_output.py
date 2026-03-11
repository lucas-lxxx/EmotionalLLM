#!/usr/bin/env python3
"""
预检脚本：确认模型输出格式

运行几个样本，打印模型生成的第一个 token，确认标签格式。
"""

import sys
import argparse
from pathlib import Path

import torch
import torchaudio
import yaml


def setup_paths(config: dict):
    """设置必要的路径"""
    src_path = Path(config['model']['src_path'])
    if str(src_path.parent) not in sys.path:
        sys.path.insert(0, str(src_path.parent))

    dataset_module = Path(config['data']['dataset_module'])
    dataset_src = dataset_module.parent.parent
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))


def load_model_and_tokenizer(config: dict):
    """加载模型和 tokenizer"""
    from src.modeling_omnispeech import OmniSpeechModel
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    model_path = Path(config['model']['model_path'])
    device = config['model']['device']

    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载模型
    model = OmniSpeechModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    # 加载音频特征提取器
    audio_extractor = WhisperFeatureExtractor.from_pretrained(model_path / "audio")

    return model, tokenizer, audio_extractor


def build_input(
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    tokenizer,
    audio_extractor,
    device: str,
    system_prompt: str = None
) -> dict:
    """构建模型输入"""
    # 音频特殊 token
    AUDIO_TOKEN_INDEX = -200
    DEFAULT_AUDIO_TOKEN = "<|im_audio|>"
    DEFAULT_AUDIO_START_TOKEN = "<|im_audio_start|>"
    DEFAULT_AUDIO_END_TOKEN = "<|im_audio_end|>"
    DEFAULT_TTS_START_TOKEN = "<|im_tts_start|>"

    # 构建 messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}{prompt}"
    })

    # 应用 chat template
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    prompt_text += DEFAULT_TTS_START_TOKEN

    # Tokenize，处理音频 token
    segments = prompt_text.split(DEFAULT_AUDIO_TOKEN)
    input_ids = []
    for idx, segment in enumerate(segments):
        if idx != 0:
            input_ids.append(AUDIO_TOKEN_INDEX)
        input_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)

    # 提取音频特征
    wav = waveform.detach().cpu().numpy().squeeze()
    outputs = audio_extractor(
        wav,
        sampling_rate=sr,
        return_attention_mask=True,
        return_tensors="pt",
    )
    speech_values = outputs.input_features.to(device)
    speech_mask = outputs.attention_mask.to(device)

    # 转换 dtype
    model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    speech_values = speech_values.to(dtype=model_dtype)

    attention_mask = torch.ones_like(input_ids, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "speech_values": speech_values,
        "speech_mask": speech_mask,
    }


def check_single_sample(
    model,
    tokenizer,
    audio_extractor,
    audio_path: str,
    prompt: str,
    device: str,
    system_prompt: str = None
):
    """检查单个样本的模型输出"""
    # 加载音频
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    waveform = waveform.to(device)

    # 构建输入
    inputs = build_input(
        waveform, sr, prompt, tokenizer, audio_extractor, device, system_prompt
    )

    # 生成
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        temperature=0,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speech_values=inputs["speech_values"],
            speech_mask=inputs["speech_mask"],
            spk_emb=None,
            generation_config=gen_config,
        )

    # 解码生成的 token
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = generated[0].tolist()
    if len(gen_tokens) == 0:
        gen_text = ""
        gen_token_id = None
    elif len(gen_tokens) <= input_len:
        gen_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        gen_token_id = gen_tokens[0]
    else:
        gen_text = tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()
        gen_token_id = gen_tokens[input_len]

    return {
        "generated_text": gen_text,
        "generated_tokens": gen_tokens,
        "first_token_id": gen_token_id,
        "first_token_str": tokenizer.decode([gen_token_id]) if gen_token_id is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="检查模型输出格式")
    parser.add_argument("--config", type=str,
                        default="/data1/lixiang/lx_code/logit_lens/configs/logit_lens_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--num_samples", type=int, default=5, help="检查的样本数量")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("模型输出格式预检")
    print("=" * 60)

    # 设置路径
    setup_paths(config)

    # 加载模型
    print("\n[1] 加载模型...")
    model, tokenizer, audio_extractor = load_model_and_tokenizer(config)
    device = config['model']['device']
    print(f"    模型加载完成，设备: {device}")

    # 加载数据集
    print("\n[2] 加载数据集...")
    dataset_module = Path(config['data']['dataset_module'])
    dataset_src = dataset_module.parent.parent
    sys.path.insert(0, str(dataset_src))
    from data.dataset import ModalConflictDataset

    dataset = ModalConflictDataset(
        text_jsonl=config['data']['text_jsonl'],
        audio_root=config['data']['audio_root'],
        emotions=config['data'].get('emotions'),
    )
    print(f"    数据集加载完成，共 {len(dataset)} 个样本")

    # 获取冲突样本
    conflict_samples = dataset.get_conflict_samples()
    print(f"    冲突样本数: {len(conflict_samples)}")

    # 检查样本
    print("\n[3] 检查模型输出...")
    prompt = config['prompt']
    system_prompt = config.get('system_prompt')

    results = []
    for i, sample in enumerate(conflict_samples[:args.num_samples]):
        print(f"\n--- 样本 {i+1}/{args.num_samples} ---")
        print(f"    音频: {sample.audio_path}")
        print(f"    语义情绪: {sample.semantic_emotion}")
        print(f"    韵律情绪: {sample.prosody_emotion}")

        result = check_single_sample(
            model, tokenizer, audio_extractor,
            sample.audio_path, prompt, device, system_prompt
        )
        results.append(result)

        print(f"    生成文本: '{result['generated_text']}'")
        print(f"    第一个 token ID: {result['first_token_id']}")
        print(f"    第一个 token 字符串: '{result['first_token_str']}'")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)

    # 检查标签 tokenization
    print("\n[4] 标签 tokenization 检查...")
    emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised']
    for emotion in emotions:
        token_ids = tokenizer.encode(emotion, add_special_tokens=False)
        token_str = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"    {emotion}: ids={token_ids}, tokens={token_str}")

    print("\n预检完成！")


if __name__ == "__main__":
    main()
