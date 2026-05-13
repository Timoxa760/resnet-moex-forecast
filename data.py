"""Модуль загрузки и предобработки данных с Московской биржи (MOEX).

Автор: Миндрин Т.Д.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def fetch_moex_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Загружает исторические котировки с MOEX через ISS API с пагинацией.

    Args:
        ticker: Тикер ценной бумаги (например, 'SBER').
        start: Начальная дата в формате 'YYYY-MM-DD'.
        end: Конечная дата в формате 'YYYY-MM-DD'.

    Returns:
        DataFrame со столбцами [TRADEDATE, OPEN, HIGH, LOW, CLOSE, VOLUME].
    """
    url = (
        f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
    )
    all_data = []
    start_idx = 0
    while True:
        params = {
            "from": start,
            "till": end,
            "iss.meta": "off",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME",
            "start": start_idx,
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()["history"]
        data = payload["data"]
        if not data:
            break
        all_data.extend(data)
        start_idx += len(data)
        if len(data) < 100:
            break

    columns = payload["columns"]
    df = pd.DataFrame(all_data, columns=columns)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    numeric_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    logger.info("Загружено %d записей для %s", len(df), ticker)
    return df


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет технические индикаторы и признаки.

    Добавляет:
        - returns: логарифмическая доходность
        - rsi: индекс относительной силы (14 периодов)
        - sma: простое скользящее среднее (20 периодов)
        - volatility: стандартное отклонение доходности (20 периодов)

    Args:
        df: DataFrame с колонками OPEN, HIGH, LOW, CLOSE, VOLUME.

    Returns:
        DataFrame с добавленными признаками.
    """
    df = df.copy()
    df["returns"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))

    # RSI
    delta = df["CLOSE"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # SMA и волатильность
    df["sma"] = df["CLOSE"].rolling(window=20, min_periods=20).mean()
    df["volatility"] = df["returns"].rolling(window=20, min_periods=20).std()

    # Нормализованные цены
    df["close_norm"] = df["CLOSE"] / df["CLOSE"].iloc[0]
    df["volume_norm"] = df["VOLUME"] / df["VOLUME"].iloc[0]

    return df.dropna().reset_index(drop=True)


def build_classification_dataset(
    df: pd.DataFrame, window_size: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Формирует обучающую выборку для задачи классификации направления.

    Args:
        df: DataFrame с признаками.
        window_size: Размер окна истории.
        horizon: Горизонт прогнозирования (дней).

    Returns:
        X: Массив формы (N, window_size, n_features).
        y: Массив меток {0, 1} формы (N,).
    """
    feature_cols = ["close_norm", "returns", "rsi", "sma", "volatility"]
    data = df[feature_cols].values
    labels = (df["CLOSE"].shift(-horizon) > df["CLOSE"]).astype(int).values

    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i : i + window_size])
        y.append(labels[i + window_size - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class MoexDataset(Dataset):
    """PyTorch Dataset для временных рядов MOEX."""

    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: StandardScaler = None):
        """Инициализация датасета.

        Args:
            X: Массив признаков формы (N, T, F).
            y: Массив меток формы (N,).
            scaler: Опциональный скейлер для нормализации.
        """
        self.X = X
        self.y = y
        if scaler is not None:
            shape = X.shape
            flat = X.reshape(-1, shape[-1])
            flat = scaler.transform(flat)
            self.X = flat.reshape(shape)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)
