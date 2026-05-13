"""Архитектура остаточной нейронной сети (ResNet) для временных рядов.

Реализована одномерная свёрточная ResNet с skip-соединениями
для задачи классификации направления движения цены.

Автор: Миндрин Т.Д.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Остаточный блок с одномерными свёртками.

    Содержит две свёрточные группы (Conv1d + BatchNorm + ReLU)
    и skip-connection вокруг них. При изменении числа каналов
    или размера шага используется проекция identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через остаточный блок.

        Args:
            x: Тензор формы (B, C_in, L).

        Returns:
            Тензор формы (B, C_out, L').
        """
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNetTimeSeries(nn.Module):
    """Остаточная нейронная сеть для анализа финансовых временных рядов.

    Архитектура включает начальную свёртку, стек остаточных блоков
    с постепенным увеличением каналов и уменьшением длины,
    адаптивный пулинг и полносвязный классификатор.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        block_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, block_channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(block_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )

        layers = []
        in_ch = block_channels[0]
        for out_ch in block_channels:
            layers.append(
                ResidualBlock1D(in_ch, out_ch, kernel_size, stride=2, dropout=dropout)
            )
            layers.append(
                ResidualBlock1D(out_ch, out_ch, kernel_size, stride=1, dropout=dropout)
            )
            in_ch = out_ch

        self.residual_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход сети.

        Args:
            x: Входной тензор формы (B, T, F), где
               B — размер батча, T — длина окна, F — число признаков.

        Returns:
            Логиты формы (B, num_classes).
        """
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.initial(x)
        x = self.residual_layers(x)
        x = self.adaptive_pool(x).squeeze(-1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Возвращает число обучаемых параметров."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
