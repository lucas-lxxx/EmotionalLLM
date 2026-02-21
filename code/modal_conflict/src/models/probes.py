"""
Emotion Probe Module

线性Probe（Logistic Regression）与可选MLP分类器，用于从hidden states预测情绪类别。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression


class EmotionProbe(nn.Module):
    """2层MLP情绪分类器"""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [512, 128],
        n_classes: int = 5,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 输入维度 (hidden_size)
            hidden_dims: 隐藏层维度列表
            n_classes: 情绪类别数
            dropout: Dropout概率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            logits: [batch_size, n_classes]
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class LinearProbeTrainer:
    """线性Probe训练器（sklearn LogisticRegression）"""

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        solver: str = "lbfgs",
        n_classes: int = 5,
        random_state: int = 42
    ):
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.n_classes = n_classes
        self.random_state = random_state
        self.model = None

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            # 转换为float32以支持numpy（bfloat16不被numpy支持）
            if x.dtype == torch.bfloat16:
                x = x.float()
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> LogisticRegression:
        """训练线性Probe"""
        X_train_np = self._to_numpy(X_train)
        y_train_np = self._to_numpy(y_train)

        self.model = LogisticRegression(
            multi_class="multinomial",
            max_iter=self.max_iter,
            C=self.C,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.model.fit(X_train_np, y_train_np)
        return self.model

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """评估线性Probe"""
        if self.model is None:
            raise ValueError("Probe not trained yet")

        X_np = self._to_numpy(X)
        y_np = self._to_numpy(y)
        preds = self.model.predict(X_np)

        return {
            "accuracy": accuracy_score(y_np, preds),
            "f1_macro": f1_score(y_np, preds, average='macro'),
            "f1_weighted": f1_score(y_np, preds, average='weighted')
        }

    def get_predictions(
        self,
        X: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取预测结果"""
        if self.model is None:
            raise ValueError("Probe not trained yet")

        X_np = self._to_numpy(X)
        preds = self.model.predict(X_np)
        probs = self.model.predict_proba(X_np)
        return preds, probs


class ProbeTrainer:
    """Probe训练器（含早停）"""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [512, 128],
        n_classes: int = 5,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        device: str = "cuda"
    ):
        """
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            n_classes: 类别数
            dropout: Dropout概率
            learning_rate: 学习率
            weight_decay: 权重衰减
            epochs: 最大训练轮数
            patience: 早停耐心值
            batch_size: 批大小
            device: 计算设备
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device

        self.probe = None
        self.best_state = None
        self.train_losses = []
        self.val_losses = []

    def _create_probe(self) -> EmotionProbe:
        """创建新的Probe实例"""
        return EmotionProbe(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            n_classes=self.n_classes,
            dropout=self.dropout
        ).to(self.device)

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> EmotionProbe:
        """
        训练Probe

        Args:
            X_train: 训练特征 [N, input_dim]
            y_train: 训练标签 [N]
            X_val: 验证特征 (可选)
            y_val: 验证标签 (可选)
            verbose: 是否打印训练信息

        Returns:
            训练好的Probe
        """
        self.probe = self._create_probe()
        optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # 创建DataLoader
        train_dataset = TensorDataset(
            X_train.to(self.device),
            y_train.to(self.device)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 早停相关
        best_val_loss = float('inf')
        patience_counter = 0
        self.best_state = None
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            # 训练
            self.probe.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.probe(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(train_dataset)
            self.train_losses.append(train_loss)

            # 验证
            if X_val is not None and y_val is not None:
                self.probe.eval()
                with torch.no_grad():
                    val_logits = self.probe(X_val.to(self.device))
                    val_loss = criterion(val_logits, y_val.to(self.device)).item()
                self.val_losses.append(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.probe.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

        # 恢复最佳模型
        if self.best_state is not None:
            self.probe.load_state_dict(self.best_state)

        self.probe.eval()
        return self.probe

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        评估Probe

        Args:
            X: 特征 [N, input_dim]
            y: 标签 [N]

        Returns:
            评估指标字典
        """
        if self.probe is None:
            raise ValueError("Probe not trained yet")

        self.probe.eval()
        with torch.no_grad():
            preds = self.probe.predict(X.to(self.device)).cpu().numpy()

        y_np = y.numpy()

        return {
            "accuracy": accuracy_score(y_np, preds),
            "f1_macro": f1_score(y_np, preds, average='macro'),
            "f1_weighted": f1_score(y_np, preds, average='weighted')
        }

    def get_predictions(
        self,
        X: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取预测结果

        Args:
            X: 特征 [N, input_dim]

        Returns:
            (predictions, probabilities)
        """
        if self.probe is None:
            raise ValueError("Probe not trained yet")

        self.probe.eval()
        with torch.no_grad():
            preds = self.probe.predict(X.to(self.device)).cpu().numpy()
            probs = self.probe.predict_proba(X.to(self.device)).cpu().numpy()

        return preds, probs


class DualProbeEvaluator:
    """双Probe评估器，同时训练semantic和prosody probe"""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [512, 128],
        n_classes: int = 5,
        **trainer_kwargs
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.trainer_kwargs = trainer_kwargs

        self.semantic_trainer = None
        self.prosody_trainer = None

    def train_and_evaluate(
        self,
        X_train: torch.Tensor,
        y_semantic_train: torch.Tensor,
        y_prosody_train: torch.Tensor,
        X_val: torch.Tensor,
        y_semantic_val: torch.Tensor,
        y_prosody_val: torch.Tensor,
        X_test: torch.Tensor,
        y_semantic_test: torch.Tensor,
        y_prosody_test: torch.Tensor,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        训练并评估双Probe

        Returns:
            包含semantic和prosody评估结果的字典
        """
        # 训练Semantic Probe
        self.semantic_trainer = ProbeTrainer(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            n_classes=self.n_classes,
            **self.trainer_kwargs
        )
        self.semantic_trainer.train(
            X_train, y_semantic_train,
            X_val, y_semantic_val,
            verbose=verbose
        )
        semantic_metrics = self.semantic_trainer.evaluate(X_test, y_semantic_test)

        # 训练Prosody Probe
        self.prosody_trainer = ProbeTrainer(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            n_classes=self.n_classes,
            **self.trainer_kwargs
        )
        self.prosody_trainer.train(
            X_train, y_prosody_train,
            X_val, y_prosody_val,
            verbose=verbose
        )
        prosody_metrics = self.prosody_trainer.evaluate(X_test, y_prosody_test)

        # 计算主导性指标
        dominance = prosody_metrics["accuracy"] - semantic_metrics["accuracy"]

        return {
            "semantic": semantic_metrics,
            "prosody": prosody_metrics,
            "dominance": dominance
        }
