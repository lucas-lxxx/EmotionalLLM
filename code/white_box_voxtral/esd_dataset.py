"""
ESD/CN 数据集加载和采样模块
"""

from __future__ import annotations

import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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
    path: Path
    speaker_id: str
    emotion: str
    filename: str

    def __str__(self) -> str:
        return f"{self.speaker_id}/{self.emotion}/{self.filename}"


@dataclass
class SpeakerData:
    speaker_id: str
    samples_by_emotion: Dict[str, List[AudioSample]]

    def get_emotion_count(self, emotion: str) -> int:
        return len(self.samples_by_emotion.get(emotion, []))

    def get_total_count(self) -> int:
        return sum(len(samples) for samples in self.samples_by_emotion.values())


def normalize_emotion_label(label: str) -> str:
    normalized = EMO_LABEL_MAP.get(label)
    if normalized is None:
        warnings.warn(f"Unknown emotion label: {label}, using lowercase")
        return label.lower()
    return normalized


def parse_esd_path(path: Path, dataset_root: Path) -> tuple[str, str]:
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
    if not dataset_root.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    wav_files = list(dataset_root.glob("*/*/*.wav"))

    if not wav_files:
        warnings.warn(f"No .wav files found in {dataset_root}")
        return {}

    speaker_samples = defaultdict(lambda: defaultdict(list))

    for wav_path in wav_files:
        try:
            speaker_id, emotion = parse_esd_path(wav_path, dataset_root)

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
            sampled.extend(available_samples)
        elif available_count < samples_per_emotion:
            warnings.warn(
                f"Speaker {speaker_data.speaker_id} has only {available_count} samples "
                f"for emotion '{emotion}' (requested {samples_per_emotion}), using all available"
            )
            sampled.extend(available_samples)
        else:
            sampled.extend(random.sample(available_samples, samples_per_emotion))

    return sampled


def create_experiment_samples(
    dataset_root: Path,
    exclude_emotion: str = "happy",
    samples_per_emotion: int = 100,
    seed: int = 1234,
) -> Dict[str, List[AudioSample]]:
    print(f"Scanning dataset: {dataset_root}")
    all_speaker_data = scan_esd_dataset(dataset_root)

    if not all_speaker_data:
        raise ValueError(f"No speakers found in {dataset_root}")

    print(f"Found {len(all_speaker_data)} speakers")

    all_emotions = set()
    for speaker_data in all_speaker_data.values():
        all_emotions.update(speaker_data.samples_by_emotion.keys())

    target_emotions = sorted(all_emotions - {exclude_emotion})

    if not target_emotions:
        raise ValueError(f"No emotions left after excluding '{exclude_emotion}'")

    print(f"Target emotions: {target_emotions}")
    print(f"Samples per emotion: {samples_per_emotion}")

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
