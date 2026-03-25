"""1D Atrous CNN Generator

参考 Ren et al. 的 atrous CNN 架构（原文为 2D mel-spec 版本），
适配为 1D 波形版本，保留核心设计：
  - 4 层空洞卷积，channels {64, 128, 64, 1}
  - dilation {1, 2, 4, 8}
  - kernel size 5
  - 前 3 层 BatchNorm + ReLU，最后一层 Sigmoid
  - 输出扰动 η ∈ [0, 1]，最终 x' = x + ε * (2η - 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import cfg


class AtrousCNNGenerator(nn.Module):
    """1D Atrous CNN for generating adversarial perturbations on waveforms."""

    def __init__(
        self,
        channels: list[int] | None = None,
        dilations: list[int] | None = None,
        kernel_size: int = cfg.gen_kernel_size,
    ):
        super().__init__()
        channels = channels or cfg.gen_channels
        dilations = dilations or cfg.gen_dilations

        assert len(channels) == len(dilations), "channels and dilations must have same length"

        layers = []
        in_ch = 1  # mono waveform input
        for i, (out_ch, dil) in enumerate(zip(channels, dilations)):
            padding = dil * (kernel_size - 1) // 2  # same padding
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dil)
            )
            if i < len(channels) - 1:
                # 前 N-1 层: BatchNorm + ReLU
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            else:
                # 最后一层: Sigmoid（输出 [0, 1] 的扰动）
                layers.append(nn.Sigmoid())
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, epsilon: float = cfg.epsilon
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time) clean waveform
            epsilon: L-inf perturbation bound

        Returns:
            x_adv: (batch, time) adversarial waveform
            eta: (batch, 1, time) raw perturbation output ∈ [0, 1]
        """
        # (batch, time) → (batch, 1, time)
        x_in = x.unsqueeze(1)

        # Generate perturbation η ∈ [0, 1]
        eta = self.net(x_in)  # (batch, 1, time)

        # Scale to [-ε, ε]: perturbation = ε * (2η - 1)
        perturbation = epsilon * (2 * eta - 1)

        # Apply perturbation
        x_adv = x_in + perturbation
        x_adv = torch.clamp(x_adv, -1.0, 1.0)

        # (batch, 1, time) → (batch, time)
        x_adv = x_adv.squeeze(1)

        return x_adv, eta
