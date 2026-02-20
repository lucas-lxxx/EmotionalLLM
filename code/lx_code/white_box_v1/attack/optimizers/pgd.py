"""PGD / MI-FGSM 优化器"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np


class PGD:
    """Projected Gradient Descent 优化器"""
    
    def __init__(
        self,
        objective_fn: Callable,
        eps: float = 0.01,
        alpha: float = 0.001,
        steps: int = 40,
        eot_k: int = 5,
        norm: str = "Linf",
        random_start: bool = True,
        device: str = "cuda:0"
    ):
        """
        Args:
            objective_fn: 目标函数，输入 waveform，返回 (loss, metrics)
            eps: 扰动上界（Linf 或 L2）
            alpha: 步长
            steps: 迭代步数
            eot_k: EOT 采样次数
            norm: 范数类型 ("Linf" 或 "L2")
            random_start: 是否随机初始化
            device: 设备
        """
        self.objective_fn = objective_fn
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.eot_k = eot_k
        self.norm = norm
        self.random_start = random_start
        self.device = device
    
    def attack(
        self,
        waveform_orig: torch.Tensor,
        eot_transform: Optional[Callable] = None
    ) -> tuple:
        """
        执行攻击
        
        Args:
            waveform_orig: [T] 原始音频波形
            eot_transform: EOT 变换函数（可选）
        
        Returns:
            waveform_adv: [T] 对抗音频
            metrics_history: 指标历史
        """
        waveform_orig = waveform_orig.to(self.device)
        
        # 初始化扰动
        if self.random_start:
            if self.norm == "Linf":
                delta = torch.empty_like(waveform_orig).uniform_(-self.eps, self.eps)
            else:  # L2
                delta = torch.randn_like(waveform_orig)
                delta = delta / torch.norm(delta, p=2) * self.eps * torch.rand(1).item()
        else:
            delta = torch.zeros_like(waveform_orig)
        
        waveform_adv = waveform_orig + delta
        
        metrics_history = []
        
        for step in range(self.steps):
            waveform_adv = waveform_adv.detach().requires_grad_(True)
            
            # EOT: 采样 k 次变换，梯度求平均
            if eot_transform is not None and self.eot_k > 1:
                gradients = []
                total_loss = 0.0
                total_metrics = {}
                
                for k in range(self.eot_k):
                    # 应用变换（每次采样一次，得到不同的随机变换）
                    transformed_list = eot_transform(waveform_adv, k=1)
                    transformed = transformed_list[0] if transformed_list else waveform_adv
                    
                    # 计算损失
                    loss, metrics = self.objective_fn(transformed, waveform_orig, compute_grad=True)
                    
                    # 反向传播
                    loss.backward()
                    gradients.append(waveform_adv.grad.clone())
                    waveform_adv.grad.zero_()
                    
                    total_loss += loss.item()
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            total_metrics[key] = total_metrics.get(key, 0.0) + value
                
                # 平均梯度
                avg_grad = torch.stack(gradients).mean(dim=0)
                waveform_adv.grad = avg_grad
                
                # 平均指标
                for key in total_metrics:
                    total_metrics[key] /= self.eot_k
                total_metrics['total_loss'] = total_loss / self.eot_k
            else:
                # 不使用 EOT
                loss, metrics = self.objective_fn(waveform_adv, waveform_orig, compute_grad=True)
                loss.backward()
                total_metrics = metrics
            
            # 梯度上升（最大化损失）
            with torch.no_grad():
                grad = waveform_adv.grad
                
                # 计算梯度范数（用于 debug）
                grad_norm = torch.norm(grad).item() if grad is not None else 0.0
                total_metrics['grad_norm_input'] = grad_norm
                
                # 更新
                if self.norm == "Linf":
                    waveform_adv = waveform_adv + self.alpha * torch.sign(grad)
                else:  # L2
                    grad_norm = torch.norm(grad, p=2)
                    if grad_norm > 0:
                        waveform_adv = waveform_adv + self.alpha * grad / grad_norm
                
                # 投影到约束集
                delta = waveform_adv - waveform_orig
                if self.norm == "Linf":
                    delta = torch.clamp(delta, -self.eps, self.eps)
                else:  # L2
                    delta_norm = torch.norm(delta, p=2)
                    if delta_norm > self.eps:
                        delta = delta / delta_norm * self.eps
                
                waveform_adv = waveform_orig + delta
            
            # 记录指标
            metrics_step = {
                'step': step,
                **total_metrics
            }
            metrics_history.append(metrics_step)
            
            if (step + 1) % 10 == 0:
                # Debug 打印：emo_prob_target, emo_loss, grad_norm_input
                emo_prob_target = total_metrics.get('emo_prob_target', 0.0)
                emo_loss = total_metrics.get('emo_text_loss', 0.0)
                grad_norm = total_metrics.get('grad_norm_input', 0.0)
                sem_loss = total_metrics.get('sem_loss', 0.0)
                sem_sim = total_metrics.get('sem_cosine_sim', 0.0)
                
                print(f"  Step {step+1}/{self.steps}: "
                      f"loss={total_metrics.get('total_loss', 0.0):.6f}, "
                      f"emo_loss={emo_loss:.6f}, "
                      f"emo_prob_target={emo_prob_target:.4f}, "
                      f"sem_loss={sem_loss:.6f}, "
                      f"sem_sim={sem_sim:.4f}, "
                      f"per={total_metrics.get('per_loss', 0.0):.6f}, "
                      f"grad_norm={grad_norm:.6f}")
                
                # 检查梯度是否为零（断图检测）
                if grad_norm < 1e-6:
                    print(f"    ⚠️  WARNING: grad_norm={grad_norm:.6e} is too small! Possible gradient break.")
        
        return waveform_adv, metrics_history


class MI_FGSM:
    """Momentum Iterative Fast Gradient Sign Method"""
    
    def __init__(
        self,
        objective_fn: Callable,
        eps: float = 0.01,
        alpha: float = 0.001,
        steps: int = 40,
        momentum: float = 0.9,
        eot_k: int = 5,
        norm: str = "Linf",
        random_start: bool = True,
        device: str = "cuda:0"
    ):
        """
        Args:
            momentum: 动量系数
            其他参数同 PGD
        """
        self.objective_fn = objective_fn
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        self.eot_k = eot_k
        self.norm = norm
        self.random_start = random_start
        self.device = device
    
    def attack(
        self,
        waveform_orig: torch.Tensor,
        eot_transform: Optional[Callable] = None
    ) -> tuple:
        """执行 MI-FGSM 攻击"""
        waveform_orig = waveform_orig.to(self.device)
        
        # 初始化
        if self.random_start:
            if self.norm == "Linf":
                delta = torch.empty_like(waveform_orig).uniform_(-self.eps, self.eps)
            else:
                delta = torch.randn_like(waveform_orig)
                delta = delta / torch.norm(delta, p=2) * self.eps * torch.rand(1).item()
        else:
            delta = torch.zeros_like(waveform_orig)
        
        waveform_adv = waveform_orig + delta
        accumulated_grad = torch.zeros_like(waveform_orig)
        
        metrics_history = []
        
        for step in range(self.steps):
            waveform_adv = waveform_adv.detach().requires_grad_(True)
            
            # EOT
            if eot_transform is not None and self.eot_k > 1:
                gradients = []
                total_loss = 0.0
                total_metrics = {}
                
                for k in range(self.eot_k):
                    transformed = eot_transform(waveform_adv, k=1)[0]
                    loss, metrics = self.objective_fn(transformed, waveform_orig, compute_grad=True)
                    loss.backward()
                    gradients.append(waveform_adv.grad.clone())
                    waveform_adv.grad.zero_()
                    
                    total_loss += loss.item()
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            total_metrics[key] = total_metrics.get(key, 0.0) + value
                
                avg_grad = torch.stack(gradients).mean(dim=0)
                waveform_adv.grad = avg_grad
                
                for key in total_metrics:
                    total_metrics[key] /= self.eot_k
                total_metrics['total_loss'] = total_loss / self.eot_k
            else:
                loss, metrics = self.objective_fn(waveform_adv, waveform_orig, compute_grad=True)
                loss.backward()
                total_metrics = metrics
            
            # 动量更新
            with torch.no_grad():
                grad = waveform_adv.grad
                
                # 累积动量
                accumulated_grad = self.momentum * accumulated_grad + grad
                
                # 更新
                if self.norm == "Linf":
                    waveform_adv = waveform_adv + self.alpha * torch.sign(accumulated_grad)
                else:
                    grad_norm = torch.norm(accumulated_grad, p=2)
                    if grad_norm > 0:
                        waveform_adv = waveform_adv + self.alpha * accumulated_grad / grad_norm
                
                # 投影
                delta = waveform_adv - waveform_orig
                if self.norm == "Linf":
                    delta = torch.clamp(delta, -self.eps, self.eps)
                else:
                    delta_norm = torch.norm(delta, p=2)
                    if delta_norm > self.eps:
                        delta = delta / delta_norm * self.eps
                
                waveform_adv = waveform_orig + delta
            
            metrics_step = {
                'step': step,
                **total_metrics
            }
            metrics_history.append(metrics_step)
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{self.steps}: loss={total_metrics.get('total_loss', 0.0):.6f}")
        
        return waveform_adv, metrics_history

