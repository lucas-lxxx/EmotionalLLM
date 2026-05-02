"""Surrogate SER 模型：wav2vec2-base + 线性分类头"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from config import cfg


class SurrogateSER(nn.Module):
    def __init__(
        self,
        model_name: str = cfg.wav2vec_model,
        num_classes: int = cfg.ser_num_classes,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        if hasattr(self.wav2vec, 'gradient_checkpointing_disable'):
            self.wav2vec.gradient_checkpointing_disable()
        if freeze_feature_extractor:
            self.wav2vec.feature_extractor._freeze_parameters()
        hidden_size = self.wav2vec.config.hidden_size  # 768 for base
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, time) 原始波形
        Returns:
            logits: (batch, num_classes)
        """
        outputs = self.wav2vec(waveform)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled = hidden.mean(dim=1)         # (batch, hidden)
        logits = self.classifier(pooled)    # (batch, num_classes)
        return logits

    def predict(self, waveform: torch.Tensor) -> torch.Tensor:
        """返回预测类别索引"""
        logits = self.forward(waveform)
        return logits.argmax(dim=-1)
