"""
ESD/CN 数据集加载和采样模块

提供 ESD (Emotional Speech Dataset) 中文数据集的扫描、解析和采样功能。
"""

from __future__ import annotations

import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# 情绪标签映射（处理大小写不一致）
EMO_LABEL_MAP = {
    "happy": "happy",
    "Happy": "happy",
    "HAPPY": "happy",
    "sad": "sad",
    "Sad": "sad",
    "SAD": "sad",
    "angry": "angry",
    "Angry": "angry",
    "ANGRY": "angry",
    "neutral": "neutral",
    "Neutral": "neutral",
    "NEUTRAL": "neutral",
    "surprise": "surprise",
    "Surprise": "surprise",
    "SURPRISE": "surprise",
}


@dataclass
class AudioSample:
    """音频样本元数据"""

    path: Path  # 音频文件完整路径
    speaker_id: str  # 说话人ID (如 "0001")
    emotion: str  # 情绪标签 (如 "happy", "sad")
    filename: str  # 文件名 (如 "0001_000001.wav")

    def __str__(self) -> str:
        return f"{self.speaker_id}/{self.emotion}/{self.filename}"


@dataclass
class SpeakerData:
    """单个说话人的数据"""

    speaker_id: str
    samples_by_emotion: Dict[str, List[AudioSample]]  # emotion -> samples

    def get_emotion_count(self, emotion: str) -> int:
        """获取指定情绪的样本数量"""
        return len(self.samples_by_emotion.get(emotion, []))

    def get_total_count(self) -> int:
        """获取总样本数量"""
        return sum(len(samples) for samples in self.samples_by_emotion.values())


def normalize_emotion_label(label: str) -> str:
    """
    规范化情绪标签（处理大小写不一致）

    Args:
        label: 原始情绪标签

    Returns:
        规范化后的小写标签
    """
    normalized = EMO_LABEL_MAP.get(label)
    if normalized is None:
        warnings.warn(f"Unknown emotion label: {label}, using lowercase")
        return label.lower()
    return normalized


def parse_esd_path(path: Path, dataset_root: Path) -> tuple[str, str]:
    """
    从路径安全地解析说话人ID和情绪标签

    Args:
        path: 音频文件路径
        dataset_root: 数据集根目录

    Returns:
        (speaker_id, emotion) 元组

    Raises:
        ValueError: 如果路径结构不符合预期
    """
    try:
        rel_path = path.relative_to(dataset_root)
        parts = rel_path.parts

        if len(parts) < 3:
            raise ValueError(f"Path structure invalid: expected at least 3 parts, got {len(parts)}")

        speaker_id = parts[0]
        emotion_raw = parts[1]
        emotion = normalize_emotion_label(emotion_raw)

        return speaker_id, emotion

    except ValueError as e:
        raise ValueError(f"Cannot parse ESD path {path}: {e}") from e


def scan_esd_dataset(
    dataset_root: Path, emotions: List[str] | None = None
) -> Dict[str, SpeakerData]:
    """
    扫描 ESD/CN 数据集目录

    Args:
        dataset_root: ESD/CN 根目录路径
        emotions: 需要的情绪列表，None表示全部

    Returns:
        speaker_id -> SpeakerData 的字典

    目录结构: ESD/CN/{speaker_id}/{emotion}/xxx.wav
    """
    if not dataset_root.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    # 扫描所有 .wav 文件
    wav_files = list(dataset_root.glob("*/*/*.wav"))

    if not wav_files:
        warnings.warn(f"No .wav files found in {dataset_root}")
        return {}

    # 按说话人和情绪组织样本
    speaker_samples = defaultdict(lambda: defaultdict(list))

    for wav_path in wav_files:
        try:
            speaker_id, emotion = parse_esd_path(wav_path, dataset_root)

            # 如果指定了情绪列表，只保留指定的情绪
            if emotions is not None and emotion not in emotions:
                continue

            sample = AudioSample(
                path=wav_path,
                speaker_id=speaker_id,
                emotion=emotion,
                filename=wav_path.name,
            )

            speaker_samples[speaker_id][emotion].append(sample)

        except ValueError as e:
            warnings.warn(f"Skipping file {wav_path}: {e}")
            continue

    # 转换为 SpeakerData 对象
    result = {}
    for speaker_id, emotions_dict in speaker_samples.items():
        result[speaker_id] = SpeakerData(
            speaker_id=speaker_id, samples_by_emotion=dict(emotions_dict)
        )

    return result


def sample_speaker_data(
    speaker_data: SpeakerData,
    target_emotions: List[str],
    samples_per_emotion: int,
    seed: int | None = None,
) -> List[AudioSample]:
    """
    为单个说话人采样数据

    Args:
        speaker_data: 说话人数据
        target_emotions: 要采样的情绪列表 (如 ["sad", "angry", "neutral", "surprise"])
        samples_per_emotion: 每种情绪采样数量 (如 100)
        seed: 随机种子

    Returns:
        采样的样本列表
    """
    if seed is not None:
        random.seed(seed)

    sampled = []

    for emotion in target_emotions:
        available_samples = speaker_data.samples_by_emotion.get(emotion, [])
        available_count = len(available_samples)

        if available_count == 0:
            warnings.warn(
                f"Speaker {speaker_data.speaker_id} has no samples for emotion '{emotion}', skipping"
            )
            continue

        if samples_per_emotion <= 0:
            # 0 or negative = use all samples
            sampled.extend(available_samples)
        elif available_count < samples_per_emotion:
            warnings.warn(
                f"Speaker {speaker_data.speaker_id} has only {available_count} samples "
                f"for emotion '{emotion}' (requested {samples_per_emotion}), using all available"
            )
            sampled.extend(available_samples)
        else:
            # 随机采样
            sampled.extend(random.sample(available_samples, samples_per_emotion))

    return sampled


def create_experiment_samples(
    dataset_root: Path,
    exclude_emotion: str = "happy",
    samples_per_emotion: int = 100,
    seed: int = 1234,
) -> Dict[str, List[AudioSample]]:
    """
    为所有说话人创建实验样本

    Args:
        dataset_root: ESD/CN 根目录
        exclude_emotion: 要排除的情绪 (默认 "happy"，不作为源情绪)
        samples_per_emotion: 每种情绪采样数量
        seed: 随机种子

    Returns:
        speaker_id -> samples 的字典
    """
    # 扫描数据集
    print(f"Scanning dataset: {dataset_root}")
    all_speaker_data = scan_esd_dataset(dataset_root)

    if not all_speaker_data:
        raise ValueError(f"No speakers found in {dataset_root}")

    print(f"Found {len(all_speaker_data)} speakers")

    # 确定要采样的情绪（排除指定情绪）
    all_emotions = set()
    for speaker_data in all_speaker_data.values():
        all_emotions.update(speaker_data.samples_by_emotion.keys())

    target_emotions = sorted(all_emotions - {exclude_emotion})

    if not target_emotions:
        raise ValueError(f"No emotions left after excluding '{exclude_emotion}'")

    print(f"Target emotions: {target_emotions}")
    print(f"Samples per emotion: {samples_per_emotion}")

    # 为每个说话人采样
    result = {}
    for speaker_id, speaker_data in all_speaker_data.items():
        samples = sample_speaker_data(
            speaker_data, target_emotions, samples_per_emotion, seed
        )

        if samples:
            result[speaker_id] = samples
            print(
                f"Speaker {speaker_id}: {len(samples)} samples "
                f"({', '.join(f'{emotion}: {speaker_data.get_emotion_count(emotion)}' for emotion in target_emotions)})"
            )
        else:
            warnings.warn(f"Speaker {speaker_id} has no valid samples, skipping")

    return result


