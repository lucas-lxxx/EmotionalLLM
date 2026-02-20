"""EOT (Expectation over Transformations) 变换集合"""

import torch
import torchaudio
import numpy as np
from typing import List, Callable, Optional
import random


class AudioTransform:
    """音频变换基类"""
    
    def __init__(self, prob: float = 1.0):
        """
        Args:
            prob: 应用该变换的概率
        """
        self.prob = prob
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Args:
            waveform: [1, T] 或 [T] 音频波形
            sample_rate: 采样率
        
        Returns:
            transformed_waveform: 变换后的波形
        """
        if random.random() > self.prob:
            return waveform
        return self.apply(waveform, sample_rate)
    
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """子类实现具体变换"""
        raise NotImplementedError


class RandomResample(AudioTransform):
    """随机重采样（16k <-> 24k）"""
    
    def __init__(self, target_rates: List[int] = [16000, 24000], prob: float = 1.0):
        super().__init__(prob)
        self.target_rates = target_rates
    
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        target_rate = random.choice(self.target_rates)
        if target_rate == sample_rate:
            return waveform
        
        # 确保是 [1, T] 格式
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # 移到 CPU 进行重采样（Resample 不支持 CUDA）
        device = waveform.device
        waveform_cpu = waveform.cpu()
        
        resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
        transformed = resampler(waveform_cpu)
        
        # 重采样回原始采样率
        resampler_back = torchaudio.transforms.Resample(target_rate, sample_rate)
        transformed = resampler_back(transformed)
        
        # 移回原设备
        return transformed.to(device)


class RandomGain(AudioTransform):
    """随机增益调整"""
    
    def __init__(self, gain_range: tuple = (-6.0, 6.0), prob: float = 1.0):
        """
        Args:
            gain_range: (min_gain_db, max_gain_db)
        """
        super().__init__(prob)
        self.gain_range = gain_range
    
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        gain_db = random.uniform(*self.gain_range)
        gain_linear = 10 ** (gain_db / 20.0)
        return waveform * gain_linear


class TimeShift(AudioTransform):
    """时间偏移（循环移位）"""
    
    def __init__(self, max_shift_ms: int = 100, prob: float = 1.0):
        """
        Args:
            max_shift_ms: 最大偏移毫秒数
        """
        super().__init__(prob)
        self.max_shift_ms = max_shift_ms
    
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        max_shift_samples = int(self.max_shift_ms * sample_rate / 1000)
        shift = random.randint(-max_shift_samples, max_shift_samples)
        
        if shift == 0:
            return waveform
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # 循环移位
        if shift > 0:
            return torch.cat([waveform[:, shift:], waveform[:, :shift]], dim=1)
        else:
            shift = abs(shift)
            return torch.cat([waveform[:, -shift:], waveform[:, :-shift]], dim=1)


class AddNoise(AudioTransform):
    """添加噪声（SNR 区间采样）"""
    
    def __init__(self, snr_range: tuple = (15.0, 30.0), prob: float = 1.0):
        """
        Args:
            snr_range: (min_snr_db, max_snr_db)
        """
        super().__init__(prob)
        self.snr_range = snr_range
    
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 10.0)
        
        # 计算信号功率
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / snr_linear
        
        # 生成高斯噪声
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        return waveform + noise


class RIRConvolution(AudioTransform):
    """RIR 卷积（简化指数衰减模拟）"""
    
    def __init__(self, prob: float = 1.0):
        super().__init__(prob)
    
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # 简化的指数衰减 RIR
        # 实际应用中可以使用真实 RIR 库
        rir_length = int(0.1 * sample_rate)  # 100ms
        t = torch.arange(rir_length, dtype=waveform.dtype, device=waveform.device) / sample_rate
        
        # 指数衰减 + 随机延迟
        delay = random.randint(0, int(0.02 * sample_rate))  # 0-20ms 延迟
        decay = random.uniform(0.3, 0.8)  # 衰减系数
        
        rir = torch.zeros(rir_length, dtype=waveform.dtype, device=waveform.device)
        rir[delay:delay+int(0.05*sample_rate)] = torch.exp(-decay * t[:int(0.05*sample_rate)-delay])
        rir = rir / torch.norm(rir)  # 归一化
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            rir = rir.unsqueeze(0)
        
        # 卷积
        transformed = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            rir.unsqueeze(0),
            padding=rir_length-1
        )
        
        # 裁剪到原始长度
        transformed = transformed[:, :, :waveform.shape[-1]]
        return transformed.squeeze(0)


class LowPassFilter(AudioTransform):
    """低通滤波"""
    
    def __init__(self, cutoff_range: tuple = (3000, 8000), prob: float = 1.0):
        """
        Args:
            cutoff_range: (min_cutoff_hz, max_cutoff_hz)
        """
        super().__init__(prob)
        self.cutoff_range = cutoff_range
    
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        cutoff = random.uniform(*self.cutoff_range)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # 使用简单的移动平均作为低通滤波（简化实现）
        # 实际可以使用 scipy.signal 或 torchaudio 的滤波器
        kernel_size = int(sample_rate / cutoff)
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = torch.ones(1, 1, kernel_size, device=waveform.device) / kernel_size
        filtered = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            kernel,
            padding=kernel_size//2
        )
        return filtered.squeeze(0)


class EOTTransformCompose:
    """EOT 变换组合"""
    
    def __init__(
        self,
        transforms: Optional[List[AudioTransform]] = None,
        sample_rate: int = 16000
    ):
        """
        Args:
            transforms: 变换列表（None 则使用默认组合）
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        if transforms is None:
            # 默认变换组合
            self.transforms = [
                RandomResample(prob=0.5),
                RandomGain(prob=0.7),
                TimeShift(prob=0.5),
                AddNoise(prob=0.6),
                RIRConvolution(prob=0.4),
                LowPassFilter(prob=0.3),
            ]
        else:
            self.transforms = transforms
    
    def __call__(self, waveform: torch.Tensor, k: int = 1) -> List[torch.Tensor]:
        """
        对波形应用 EOT 变换（采样 k 次）
        
        Args:
            waveform: [1, T] 或 [T] 音频波形
            k: 采样次数
        
        Returns:
            transformed_waveforms: List of [T] 变换后的波形列表
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        transformed_list = []
        for _ in range(k):
            transformed = waveform.clone()
            # 顺序应用所有变换
            for transform in self.transforms:
                transformed = transform(transformed, self.sample_rate)
            transformed_list.append(transformed.squeeze(0))
        
        return transformed_list
    
    def apply_single(self, waveform: torch.Tensor) -> torch.Tensor:
        """应用单次变换（用于确定性测试）"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        transformed = waveform.clone()
        for transform in self.transforms:
            transformed = transform(transformed, self.sample_rate)
        
        return transformed.squeeze(0)


def create_default_eot_transform(sample_rate: int = 16000) -> EOTTransformCompose:
    """创建默认 EOT 变换组合"""
    return EOTTransformCompose(sample_rate=sample_rate)

