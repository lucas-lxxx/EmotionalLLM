"""ESD English 数据集加载

支持两种目录结构：
  A) speaker/Emotion/split/*.wav   （标准 ESD）
  B) speaker/emotion/*.wav          （扁平化）
自动检测并适配。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from config import cfg


EMO_NORMALIZE = {
    "angry": "angry", "Angry": "angry", "ANGRY": "angry",
    "happy": "happy", "Happy": "happy", "HAPPY": "happy",
    "neutral": "neutral", "Neutral": "neutral", "NEUTRAL": "neutral",
    "sad": "sad", "Sad": "sad", "SAD": "sad",
    "surprise": "surprise", "Surprise": "surprise", "SURPRISE": "surprise",
}


@dataclass
class AudioEntry:
    path: Path
    speaker_id: str
    emotion: str
    split: str  # "train", "evaluation", "test"


def _detect_structure(esd_root: Path, speakers: list[str]) -> str:
    """检测 ESD 目录结构：'standard' (有 train/eval/test) 或 'flat'"""
    for sp in speakers:
        sp_dir = esd_root / sp
        if not sp_dir.exists():
            continue
        for emo_dir in sp_dir.iterdir():
            if not emo_dir.is_dir():
                continue
            if (emo_dir / "train").is_dir():
                return "standard"
            if list(emo_dir.glob("*.wav")):
                return "flat"
    raise FileNotFoundError(f"无法在 {esd_root} 中找到 ESD 数据")


def scan_esd_en(
    esd_root: Path,
    speakers: list[str],
    emotions: list[str],
) -> List[AudioEntry]:
    """扫描 ESD 英文部分，返回所有条目"""
    structure = _detect_structure(esd_root, speakers)
    print(f"ESD 目录结构: {structure}")

    entries = []
    for sp in speakers:
        sp_dir = esd_root / sp
        if not sp_dir.exists():
            print(f"  跳过不存在的 speaker: {sp}")
            continue

        for emo_dir in sp_dir.iterdir():
            if not emo_dir.is_dir():
                continue
            emo_raw = emo_dir.name
            emo = EMO_NORMALIZE.get(emo_raw)
            if emo is None or emo not in emotions:
                continue

            if structure == "standard":
                for split_name in ["train", "evaluation", "test"]:
                    split_dir = emo_dir / split_name
                    if not split_dir.exists():
                        continue
                    for wav in sorted(split_dir.glob("*.wav")):
                        entries.append(AudioEntry(wav, sp, emo, split_name))
            else:
                for wav in sorted(emo_dir.glob("*.wav")):
                    entries.append(AudioEntry(wav, sp, emo, "train"))

    print(f"扫描到 {len(entries)} 个音频文件")
    return entries


def split_entries(
    entries: List[AudioEntry], split: str
) -> List[AudioEntry]:
    """按 split 筛选。如果全是 'train'（flat 结构），随机划分 80/10/10"""
    splits_found = set(e.split for e in entries)

    if splits_found == {"train"}:
        random.seed(42)
        shuffled = list(entries)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.8)
        n_eval = int(n * 0.1)
        if split == "train":
            return shuffled[:n_train]
        elif split in ("evaluation", "eval"):
            return shuffled[n_train:n_train + n_eval]
        else:  # test
            return shuffled[n_train + n_eval:]
    else:
        target = "evaluation" if split == "eval" else split
        return [e for e in entries if e.split == target]


class ESDDataset(Dataset):
    """ESD English PyTorch Dataset"""

    def __init__(
        self,
        esd_root: Path,
        speakers: list[str],
        emotions: list[str],
        split: str = "train",
        sample_rate: int = 16000,
        max_len: int = 96000,
        emotion2idx: dict[str, int] | None = None,
    ):
        all_entries = scan_esd_en(esd_root, speakers, emotions)
        self.entries = split_entries(all_entries, split)
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.emotion2idx = emotion2idx or cfg.emotion2idx

        emo_counts = {}
        for e in self.entries:
            emo_counts[e.emotion] = emo_counts.get(e.emotion, 0) + 1
        print(f"  [{split}] {len(self.entries)} 样本: {emo_counts}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        entry = self.entries[idx]
        waveform, sr = torchaudio.load(str(entry.path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if waveform.shape[0] > self.max_len:
            waveform = waveform[:self.max_len]
        elif waveform.shape[0] < self.max_len:
            pad = torch.zeros(self.max_len - waveform.shape[0])
            waveform = torch.cat([waveform, pad])

        label = self.emotion2idx[entry.emotion]
        return waveform, label, entry.emotion, str(entry.path)


def collate_fn(batch):
    waveforms, labels, emotions, paths = zip(*batch)
    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels, list(emotions), list(paths)
