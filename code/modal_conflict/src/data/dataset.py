"""
Modal Conflict Dataset Module

加载text.jsonl和音频文件，构建模态冲突样本数据集。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from torch.utils.data import Dataset


@dataclass
class ModalConflictSample:
    """模态冲突样本数据类"""
    sample_id: str           # 唯一标识符
    text_id: str             # 文本ID (e.g., "t000")
    text: str                # 文本内容
    semantic_emotion: str    # 语义情绪 (文本标注)
    prosody_emotion: str     # 韵律情绪 (音频情绪)
    audio_path: str          # 音频文件路径
    is_conflict: bool        # 是否为冲突样本 (semantic != prosody)

    def __post_init__(self):
        self.is_conflict = self.semantic_emotion != self.prosody_emotion

    @property
    def semantic_label(self) -> int:
        """语义情绪标签 (用于分类)"""
        return EMOTION_TO_IDX[self.semantic_emotion]

    @property
    def prosody_label(self) -> int:
        """韵律情绪标签 (用于分类)"""
        return EMOTION_TO_IDX[self.prosody_emotion]


# 情绪标签映射
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprised']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}


class ModalConflictDataset(Dataset):
    """模态冲突数据集类"""

    def __init__(
        self,
        text_jsonl: str,
        audio_root: str,
        emotions: Optional[List[str]] = None
    ):
        """
        Args:
            text_jsonl: text.jsonl文件路径
            audio_root: 音频根目录
            emotions: 情绪类别列表，默认使用所有5类
        """
        self.text_jsonl = Path(text_jsonl)
        self.audio_root = Path(audio_root)
        self.emotions = emotions or EMOTIONS

        self.samples: List[ModalConflictSample] = []
        self._load_data()

    def _load_data(self):
        """加载数据"""
        # 加载text.jsonl
        text_data = []
        with open(self.text_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    text_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    print(f"Line content: {repr(line)}")
                    continue

        # 遍历音频目录，匹配文本
        for text_item in text_data:
            text_id = text_item['id']
            text = text_item['text']
            semantic_emotion = text_item.get('text_emotion', text_item.get('emotion'))
            if semantic_emotion is None:
                raise KeyError("Missing 'text_emotion' in text.jsonl entry.")

            # 查找该文本对应的所有韵律版本音频
            for prosody_emotion in self.emotions:
                # 音频文件名格式: {emotion}/{id}.wav
                audio_path = self.audio_root / prosody_emotion / f"{text_id}.wav"

                if audio_path.exists():
                    sample = ModalConflictSample(
                        sample_id=f"{text_id}_{prosody_emotion}",
                        text_id=text_id,
                        text=text,
                        semantic_emotion=semantic_emotion,
                        prosody_emotion=prosody_emotion,
                        audio_path=str(audio_path),
                        is_conflict=(semantic_emotion != prosody_emotion)
                    )
                    self.samples.append(sample)

        # 统计信息
        self._compute_stats()

    def _compute_stats(self):
        """计算数据集统计信息"""
        self.n_samples = len(self.samples)
        self.n_conflict = sum(1 for s in self.samples if s.is_conflict)
        self.n_consistent = self.n_samples - self.n_conflict

        # 按情绪统计
        self.semantic_counts = {}
        self.prosody_counts = {}
        for s in self.samples:
            self.semantic_counts[s.semantic_emotion] = \
                self.semantic_counts.get(s.semantic_emotion, 0) + 1
            self.prosody_counts[s.prosody_emotion] = \
                self.prosody_counts.get(s.prosody_emotion, 0) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ModalConflictSample:
        return self.samples[idx]

    def get_conflict_samples(self) -> List[ModalConflictSample]:
        """获取所有冲突样本"""
        return [s for s in self.samples if s.is_conflict]

    def get_consistent_samples(self) -> List[ModalConflictSample]:
        """获取所有一致样本"""
        return [s for s in self.samples if not s.is_conflict]

    def get_samples_by_text_id(self, text_id: str) -> List[ModalConflictSample]:
        """获取指定文本ID的所有样本"""
        return [s for s in self.samples if s.text_id == text_id]

    def get_text_ids(self) -> List[str]:
        """获取所有唯一的文本ID"""
        return sorted(set(s.text_id for s in self.samples))

    def get_group_labels(self) -> List[str]:
        """获取分组标签 (用于GroupKFold)"""
        return [s.text_id for s in self.samples]

    def summary(self) -> Dict[str, Any]:
        """返回数据集摘要"""
        return {
            'total_samples': self.n_samples,
            'conflict_samples': self.n_conflict,
            'consistent_samples': self.n_consistent,
            'unique_texts': len(self.get_text_ids()),
            'semantic_distribution': self.semantic_counts,
            'prosody_distribution': self.prosody_counts,
        }

    def __repr__(self) -> str:
        return (
            f"ModalConflictDataset(\n"
            f"  total_samples={self.n_samples},\n"
            f"  conflict_samples={self.n_conflict},\n"
            f"  consistent_samples={self.n_consistent},\n"
            f"  unique_texts={len(self.get_text_ids())}\n"
            f")"
        )


def load_dataset(config: Dict[str, Any]) -> ModalConflictDataset:
    """从配置加载数据集"""
    return ModalConflictDataset(
        text_jsonl=config['data']['text_jsonl'],
        audio_root=config['data']['audio_root'],
        emotions=config['data'].get('emotions', EMOTIONS)
    )
