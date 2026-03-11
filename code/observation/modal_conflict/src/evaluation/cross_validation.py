"""
Cross Validation Module

使用GroupKFold进行逐层Probe评估，防止数据泄露。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import pandas as pd

from ..models.probes import ProbeTrainer, LinearProbeTrainer


class LayerWiseProbeEvaluator:
    """逐层Probe评估器，使用GroupKFold交叉验证"""

    def __init__(
        self,
        n_layers: int = 36,
        n_splits: int = 5,
        hidden_dims: List[int] = [512, 128],
        n_classes: int = 5,
        random_seed: int = 42,
        device: str = "cuda",
        probe_type: str = "mlp",
        linear_probe_kwargs: Optional[Dict[str, Any]] = None,
        **probe_kwargs
    ):
        """
        Args:
            n_layers: 模型层数
            n_splits: 交叉验证折数
            hidden_dims: Probe隐藏层维度
            n_classes: 情绪类别数
            random_seed: 随机种子
            device: 计算设备
            **probe_kwargs: ProbeTrainer的其他参数
        """
        self.n_layers = n_layers
        self.n_splits = n_splits
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.device = device
        self.probe_type = probe_type.lower()
        self.linear_probe_kwargs = linear_probe_kwargs or {}
        self.probe_kwargs = probe_kwargs

        self.results = None

    def evaluate(
        self,
        hidden_states: torch.Tensor,
        semantic_labels: torch.Tensor,
        prosody_labels: torch.Tensor,
        text_ids: torch.Tensor,
        is_conflict: torch.Tensor,
        layers_to_evaluate: Optional[List[int]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        逐层评估Probe

        Args:
            hidden_states: [N, n_layers, hidden_dim]
            semantic_labels: [N] 语义情绪标签
            prosody_labels: [N] 韵律情绪标签
            text_ids: [N] 文本ID (用于分组)
            is_conflict: [N] 是否为冲突样本
            layers_to_evaluate: 要评估的层索引列表，None表示所有层
            verbose: 是否显示进度

        Returns:
            DataFrame with per-layer metrics
        """
        n_samples, n_layers, hidden_dim = hidden_states.shape

        if layers_to_evaluate is None:
            layers_to_evaluate = list(range(n_layers))

        # GroupKFold
        gkf = GroupKFold(n_splits=self.n_splits)
        if hasattr(text_ids, "numpy"):
            groups = text_ids.numpy()
        else:
            groups = np.asarray(text_ids)

        results = []

        layer_iterator = tqdm(layers_to_evaluate, desc="Evaluating layers") if verbose else layers_to_evaluate

        for layer_idx in layer_iterator:
            layer_results = self._evaluate_layer(
                X=hidden_states[:, layer_idx, :],
                y_semantic=semantic_labels,
                y_prosody=prosody_labels,
                is_conflict=is_conflict,
                groups=groups,
                gkf=gkf,
                hidden_dim=hidden_dim
            )
            layer_results["layer"] = layer_idx
            results.append(layer_results)

        self.results = pd.DataFrame(results)
        return self.results

    def _evaluate_layer(
        self,
        X: torch.Tensor,
        y_semantic: torch.Tensor,
        y_prosody: torch.Tensor,
        is_conflict: torch.Tensor,
        groups: np.ndarray,
        gkf: GroupKFold,
        hidden_dim: int
    ) -> Dict[str, float]:
        """评估单层"""

        # 存储每折的结果
        semantic_accs_all = []
        semantic_f1s_all = []
        prosody_accs_all = []
        prosody_f1s_all = []

        # 冲突子集
        semantic_accs_conflict = []
        prosody_accs_conflict = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, groups=groups)):
            # 划分数据
            X_train = X[train_idx]
            X_test = X[test_idx]

            y_sem_train = y_semantic[train_idx]
            y_sem_test = y_semantic[test_idx]

            y_pro_train = y_prosody[train_idx]
            y_pro_test = y_prosody[test_idx]

            conflict_test = is_conflict[test_idx]

            # 进一步划分训练集和验证集 (80/20)
            n_train = len(train_idx)
            n_val = int(n_train * 0.2)
            perm = np.random.RandomState(self.random_seed + fold_idx).permutation(n_train)
            val_perm = perm[:n_val]
            train_perm = perm[n_val:]

            X_train_split = X_train[train_perm]
            X_val = X_train[val_perm]

            y_sem_train_split = y_sem_train[train_perm]
            y_sem_val = y_sem_train[val_perm]

            y_pro_train_split = y_pro_train[train_perm]
            y_pro_val = y_pro_train[val_perm]

            # 训练Semantic Probe
            if self.probe_type == "linear":
                sem_trainer = LinearProbeTrainer(
                    n_classes=self.n_classes,
                    random_state=self.random_seed + fold_idx,
                    **self.linear_probe_kwargs
                )
            else:
                sem_trainer = ProbeTrainer(
                    input_dim=hidden_dim,
                    hidden_dims=self.hidden_dims,
                    n_classes=self.n_classes,
                    device=self.device,
                    **self.probe_kwargs
                )
            sem_trainer.train(
                X_train_split, y_sem_train_split,
                X_val, y_sem_val,
                verbose=False
            )
            sem_metrics = sem_trainer.evaluate(X_test, y_sem_test)
            semantic_accs_all.append(sem_metrics["accuracy"])
            semantic_f1s_all.append(sem_metrics["f1_macro"])

            # 训练Prosody Probe
            if self.probe_type == "linear":
                pro_trainer = LinearProbeTrainer(
                    n_classes=self.n_classes,
                    random_state=self.random_seed + fold_idx,
                    **self.linear_probe_kwargs
                )
            else:
                pro_trainer = ProbeTrainer(
                    input_dim=hidden_dim,
                    hidden_dims=self.hidden_dims,
                    n_classes=self.n_classes,
                    device=self.device,
                    **self.probe_kwargs
                )
            pro_trainer.train(
                X_train_split, y_pro_train_split,
                X_val, y_pro_val,
                verbose=False
            )
            pro_metrics = pro_trainer.evaluate(X_test, y_pro_test)
            prosody_accs_all.append(pro_metrics["accuracy"])
            prosody_f1s_all.append(pro_metrics["f1_macro"])

            # 冲突子集评估
            if conflict_test.sum() > 0:
                X_test_conflict = X_test[conflict_test]
                y_sem_test_conflict = y_sem_test[conflict_test]
                y_pro_test_conflict = y_pro_test[conflict_test]

                sem_conflict_metrics = sem_trainer.evaluate(X_test_conflict, y_sem_test_conflict)
                pro_conflict_metrics = pro_trainer.evaluate(X_test_conflict, y_pro_test_conflict)

                semantic_accs_conflict.append(sem_conflict_metrics["accuracy"])
                prosody_accs_conflict.append(pro_conflict_metrics["accuracy"])

        # 计算平均指标
        return {
            # 全量数据
            "semantic_acc": np.mean(semantic_accs_all),
            "semantic_acc_std": np.std(semantic_accs_all),
            "semantic_f1": np.mean(semantic_f1s_all),
            "prosody_acc": np.mean(prosody_accs_all),
            "prosody_acc_std": np.std(prosody_accs_all),
            "prosody_f1": np.mean(prosody_f1s_all),
            "dominance": np.mean(prosody_accs_all) - np.mean(semantic_accs_all),

            # 冲突子集
            "semantic_acc_conflict": np.mean(semantic_accs_conflict) if semantic_accs_conflict else np.nan,
            "prosody_acc_conflict": np.mean(prosody_accs_conflict) if prosody_accs_conflict else np.nan,
            "dominance_conflict": (
                np.mean(prosody_accs_conflict) - np.mean(semantic_accs_conflict)
                if semantic_accs_conflict else np.nan
            )
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if self.results is None:
            raise ValueError("No results available. Run evaluate() first.")

        df = self.results

        # 找到各指标的极值层
        max_semantic_layer = df.loc[df["semantic_acc"].idxmax(), "layer"]
        max_prosody_layer = df.loc[df["prosody_acc"].idxmax(), "layer"]
        max_dominance_layer = df.loc[df["dominance"].idxmax(), "layer"]
        min_dominance_layer = df.loc[df["dominance"].idxmin(), "layer"]

        # 确定主导模态
        avg_dominance = df["dominance"].mean()
        if avg_dominance > 0.05:
            overall_dominant = "prosody"
        elif avg_dominance < -0.05:
            overall_dominant = "semantic"
        else:
            overall_dominant = "balanced"

        # 分析层级趋势
        early_layers = df[df["layer"] < 12]
        middle_layers = df[(df["layer"] >= 12) & (df["layer"] < 24)]
        late_layers = df[df["layer"] >= 24]

        return {
            "overall_dominant_modality": overall_dominant,
            "average_dominance": avg_dominance,
            "max_semantic_acc": {
                "layer": int(max_semantic_layer),
                "accuracy": df.loc[df["layer"] == max_semantic_layer, "semantic_acc"].values[0]
            },
            "max_prosody_acc": {
                "layer": int(max_prosody_layer),
                "accuracy": df.loc[df["layer"] == max_prosody_layer, "prosody_acc"].values[0]
            },
            "max_prosody_dominance": {
                "layer": int(max_dominance_layer),
                "dominance": df.loc[df["layer"] == max_dominance_layer, "dominance"].values[0]
            },
            "max_semantic_dominance": {
                "layer": int(min_dominance_layer),
                "dominance": df.loc[df["layer"] == min_dominance_layer, "dominance"].values[0]
            },
            "layer_trends": {
                "early_layers_dominance": early_layers["dominance"].mean() if len(early_layers) > 0 else np.nan,
                "middle_layers_dominance": middle_layers["dominance"].mean() if len(middle_layers) > 0 else np.nan,
                "late_layers_dominance": late_layers["dominance"].mean() if len(late_layers) > 0 else np.nan
            },
            "conflict_subset": {
                "avg_semantic_acc": df["semantic_acc_conflict"].mean(),
                "avg_prosody_acc": df["prosody_acc_conflict"].mean(),
                "avg_dominance": df["dominance_conflict"].mean()
            }
        }


def run_evaluation(
    hidden_states: torch.Tensor,
    semantic_labels: torch.Tensor,
    prosody_labels: torch.Tensor,
    text_ids: torch.Tensor,
    is_conflict: torch.Tensor,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    运行完整评估流程

    Args:
        hidden_states: [N, n_layers, hidden_dim]
        semantic_labels: [N]
        prosody_labels: [N]
        text_ids: [N]
        is_conflict: [N]
        config: 配置字典

    Returns:
        (results_df, summary_dict)
    """
    evaluator = LayerWiseProbeEvaluator(
        n_layers=config["model"]["n_layers"],
        n_splits=config["evaluation"]["n_splits"],
        hidden_dims=config["probe"]["hidden_dims"],
        n_classes=5,
        random_seed=config["evaluation"]["random_seed"],
        device=config["model"]["device"],
        probe_type=config["probe"].get("type", "mlp"),
        linear_probe_kwargs=config["probe"].get("linear", {}),
        dropout=config["probe"]["dropout"],
        learning_rate=config["probe"]["learning_rate"],
        weight_decay=config["probe"]["weight_decay"],
        epochs=config["probe"]["epochs"],
        patience=config["probe"]["patience"],
        batch_size=config["probe"]["batch_size"]
    )

    results_df = evaluator.evaluate(
        hidden_states=hidden_states,
        semantic_labels=semantic_labels,
        prosody_labels=prosody_labels,
        text_ids=text_ids,
        is_conflict=is_conflict,
        layers_to_evaluate=config["evaluation"]["layers_to_evaluate"],
        verbose=True
    )

    summary = evaluator.get_summary()

    return results_df, summary
