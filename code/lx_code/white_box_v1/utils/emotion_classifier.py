"""情绪分类器：用于边界感知几何"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional


class FrozenEmotionClassifier(nn.Module):
    """
    冻结的情绪分类器（用于计算决策边界）
    """
    
    def __init__(self, input_dim: int, num_emotions: int = 5):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_emotions)
        self.num_emotions = num_emotions
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: [B, R] 子空间表示
        
        Returns:
            logits: [B, C] 情绪 logits
        """
        return self.linear(z)
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True


def train_emotion_classifier(
    Z_train: np.ndarray,  # [N, R] 训练样本
    y_train: np.ndarray,  # [N] 情绪标签
    device: str = "cuda:0",
    epochs: int = 100,
    lr: float = 0.01
) -> FrozenEmotionClassifier:
    """
    训练情绪分类器
    
    Args:
        Z_train: [N, R] 训练样本
        y_train: [N] 情绪标签
        device: 设备
        epochs: 训练轮数
        lr: 学习率
    
    Returns:
        classifier: 训练好的分类器（已冻结）
    """
    R = Z_train.shape[1]
    num_emotions = len(np.unique(y_train))
    
    # 创建分类器
    classifier = FrozenEmotionClassifier(R, num_emotions).to(device)
    
    # 转换为 tensor
    Z_tensor = torch.from_numpy(Z_train).float().to(device)
    y_tensor = torch.from_numpy(y_train).long().to(device)
    
    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier(Z_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_tensor).float().mean()
                print(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    # 冻结参数
    classifier.freeze()
    classifier.eval()
    
    return classifier


def compute_boundary_direction(
    z: torch.Tensor,  # [B, R] 当前子空间表示
    classifier: FrozenEmotionClassifier,
    source_emotion_idx: int,
    target_emotion_idx: int
) -> torch.Tensor:
    """
    计算从源情绪到目标情绪的决策边界方向
    
    Args:
        z: [B, R] 当前子空间表示
        classifier: 冻结的分类器
        source_emotion_idx: 源情绪索引
        target_emotion_idx: 目标情绪索引
    
    Returns:
        boundary_direction: [B, R] 边界方向（归一化）
    """
    z.requires_grad_(True)
    
    # 计算 logits
    logits = classifier(z)  # [B, C]
    
    # 计算 logit 差异（目标 - 源）
    logit_diff = logits[:, target_emotion_idx] - logits[:, source_emotion_idx]  # [B]
    
    # 计算梯度方向（指向决策边界）
    logit_diff.sum().backward(retain_graph=True)
    
    # 获取梯度（即边界方向）
    boundary_direction = z.grad.clone()  # [B, R]
    
    # 清除梯度
    z.grad = None
    z.requires_grad_(False)
    
    # 归一化
    norm = torch.norm(boundary_direction, dim=1, keepdim=True) + 1e-8
    boundary_direction = boundary_direction / norm
    
    return boundary_direction






