"""STAA-Net Wave-U-Net Generator

参考 STAA-Net (arXiv 2402.01227) 官方实现。
架构：1D U-Net encoder-decoder + skip connections + 双输出头（magnitude + sparse mask）。
扰动：δ = v ⊙ m，x_adv = x + δ。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class DownBlock(nn.Module):
    """下采样块：Conv1d + LeakyReLU + Conv1d(stride)"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.act = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """上采样块：Upsample + Conv1d + Concat skip + Conv1d"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.stride = stride
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.act = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.stride, mode="linear", align_corners=False)
        # 对齐长度
        diff = skip.shape[2] - x.shape[2]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :skip.shape[2]]
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class WaveUNetGenerator(nn.Module):
    """STAA-Net Wave-U-Net generator

    输入：(batch, 1, time) 原始波形
    输出：(adv_waveform, magnitude_v, mask_m)
      - adv_waveform: x + v ⊙ m
      - magnitude_v: [-eps, eps] bounded
      - mask_m: [0, 1], 推理时 threshold=0.5 得到 binary mask
    """

    def __init__(
        self,
        channels: list[int] | None = None,
        kernel_size: int = cfg.unet_kernel_size,
        stride: int = cfg.unet_stride,
    ):
        super().__init__()
        if channels is None:
            channels = cfg.unet_channels  # [24, 48, 72, 96, 120, 144]

        self.num_levels = len(channels)
        pad = kernel_size // 2

        # Input conv
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size, padding=pad)
        self.input_act = nn.LeakyReLU(0.2)
        self.input_bn = nn.BatchNorm1d(channels[0])

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.encoders.append(DownBlock(channels[i], channels[i + 1], kernel_size, stride))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size, padding=pad),
            nn.BatchNorm1d(channels[-1]),
            nn.LeakyReLU(0.2),
        )

        # Decoder - magnitude path
        self.decoders_mag = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            self.decoders_mag.append(
                UpBlock(channels[i + 1], channels[i], channels[i], kernel_size, stride)
            )

        # Decoder - mask path (shared encoder, separate decoder)
        self.decoders_mask = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            self.decoders_mask.append(
                UpBlock(channels[i + 1], channels[i], channels[i], kernel_size, stride)
            )

        # Output heads
        self.mag_head = nn.Conv1d(channels[0], 1, kernel_size=1)   # → tanh → eps*
        self.mask_head = nn.Conv1d(channels[0], 1, kernel_size=1)  # → sigmoid → [0,1]

    def forward(
        self,
        x: torch.Tensor,
        eps: float,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, time) 或 (batch, time) 原始波形
            eps: 扰动幅值上界
            training: True=使用 soft mask，False=使用 hard threshold
        Returns:
            x_adv: (batch, 1, time) 对抗波形
            v: (batch, 1, time) magnitude [-eps, eps]
            m: (batch, 1, time) mask [0, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)

        input_len = x.shape[2]

        # Encoder
        h = self.input_act(self.input_bn(self.input_conv(x)))
        skips = [h]
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder - magnitude
        h_mag = h
        for i, dec in enumerate(self.decoders_mag):
            skip_idx = self.num_levels - 2 - i
            h_mag = dec(h_mag, skips[skip_idx])

        # Decoder - mask
        h_mask = h
        for i, dec in enumerate(self.decoders_mask):
            skip_idx = self.num_levels - 2 - i
            h_mask = dec(h_mask, skips[skip_idx])

        # Output heads
        v = eps * torch.tanh(self.mag_head(h_mag))  # [-eps, eps]

        mask_raw = self.mask_head(h_mask)
        m_soft = (torch.tanh(mask_raw) + 1.0) / 2.0  # [0, 1]

        if training:
            m = m_soft
        else:
            # 推理时 hard threshold
            m = (m_soft >= 0.5).float()

        # 对齐输出长度
        v = v[:, :, :input_len]
        m = m[:, :, :input_len]

        # 扰动
        delta = v * m
        x_adv = x + delta

        return x_adv, v, m
