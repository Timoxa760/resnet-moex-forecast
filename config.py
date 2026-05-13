"""Конфигурация эксперимента по прогнозированию финансовой динамики.

Автор: Миндрин Т.Д.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Параметры загрузки и предобработки данных."""

    tickers: List[str]
    start_date: str
    end_date: str
    window_size: int
    forecast_horizon: int
    train_ratio: float
    val_ratio: float
    random_seed: int


@dataclass
class ModelConfig:
    """Архитектурные параметры остаточной сети."""

    in_channels: int
    num_classes: int
    block_channels: List[int]
    kernel_size: int
    dropout: float


@dataclass
class TrainConfig:
    """Гиперпараметры обучения."""

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    device: str


# Предустановленная конфигурация для российских акций (MOEX)
DEFAULT_DATA_CONFIG = DataConfig(
    tickers=["SBER", "GAZP", "YNDX"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    window_size=30,
    forecast_horizon=1,
    train_ratio=0.7,
    val_ratio=0.15,
    random_seed=42,
)

DEFAULT_MODEL_CONFIG = ModelConfig(
    in_channels=5,
    num_classes=2,
    block_channels=[64, 128, 256],
    kernel_size=3,
    dropout=0.3,
)

DEFAULT_TRAIN_CONFIG = TrainConfig(
    batch_size=64,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=15,
    device="cpu",
)
