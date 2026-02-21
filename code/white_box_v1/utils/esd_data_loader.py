"""ESD 数据加载工具"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


# ESD_BASE_PATH should be set by user or environment variable
ESD_BASE_PATH = os.environ.get("ESD_BASE_PATH", "/data3/xuzhenyu/ESD/OpenDataLab___ESD/raw/ESD/Emotional Speech Dataset (ESD)")


def get_esd_speakers() -> List[str]:
    """获取所有说话人ID"""
    speakers = []
    base = Path(ESD_BASE_PATH)
    if not base.exists():
        print(f"Warning: ESD_BASE_PATH does not exist: {ESD_BASE_PATH}")
        return []
    for item in base.iterdir():
        if item.is_dir() and item.name.isdigit():
            speakers.append(item.name)
    return sorted(speakers)


def get_emotions() -> List[str]:
    """获取所有情绪类型"""
    return ["Angry", "Happy", "Neutral", "Sad", "Surprise"]


def get_splits() -> List[str]:
    """获取所有数据分割"""
    return ["train", "evaluation", "test"]


def load_text_file(speaker_id: str) -> Dict[str, str]:
    """
    加载说话人的文本文件
    
    Returns:
        {sample_id: text} 字典
    """
    text_file = Path(ESD_BASE_PATH) / speaker_id / f"{speaker_id}.txt"
    if not text_file.exists():
        return {}
    
    texts = {}
    try:
        with open(text_file, "r", encoding="gbk", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    sample_id = parts[0]
                    text = parts[1]
                    texts[sample_id] = text
    except (UnicodeDecodeError, LookupError):
        with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    sample_id = parts[0]
                    text = parts[1]
                    texts[sample_id] = text
    
    return texts


def get_audio_files(
    speaker_id: str,
    emotion: str,
    split: str = "train",
    max_files: Optional[int] = None
) -> List[str]:
    """
    获取音频文件列表
    
    Args:
        speaker_id: 说话人ID
        emotion: 情绪类型
        split: 数据分割（train/evaluation/test）
        max_files: 最大文件数（None表示全部）
    
    Returns:
        音频文件路径列表
    """
    audio_dir = Path(ESD_BASE_PATH) / speaker_id / emotion / split
    if not audio_dir.exists():
        return []
    
    audio_files = sorted([str(f) for f in audio_dir.glob("*.wav")])
    
    if max_files is not None and len(audio_files) > max_files:
        audio_files = audio_files[:max_files]
    
    return audio_files


def get_sample_with_text(
    speaker_id: str,
    emotion: str,
    split: str = "train",
    max_samples: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    获取音频文件及其对应的文本
    
    Returns:
        [(audio_path, text), ...] 列表
    """
    audio_files = get_audio_files(speaker_id, emotion, split, max_samples)
    texts = load_text_file(speaker_id)
    
    samples = []
    for audio_file in audio_files:
        sample_id = Path(audio_file).stem
        text = texts.get(sample_id, "")
        samples.append((audio_file, text))
    
    return samples

