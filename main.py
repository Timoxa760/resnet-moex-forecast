"""Главный скрипт запуска эксперимента.

Последовательность действий:
    1. Загрузка данных с MOEX для тикеров SBER, GAZP, YNDX.
    2. Инжиниринг признаков и формирование выборок.
    3. Обучение ResNet и оценка на тестовой выборке.
    4. Сохранение метрик и визуализаций.

Автор: Миндрин Т.Д.
"""

import logging
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG
from data import build_classification_dataset, compute_technical_features, fetch_moex_data, MoexDataset
from evaluate import calculate_metrics, get_predictions, plot_confusion_matrix, plot_roc_curve, plot_training_history
from model import ResNetTimeSeries
from train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_experiment(ticker: str, out_dir: str) -> dict:
    """Проводит полный эксперимент для одного тикера.

    Args:
        ticker: Тикер ценной бумаги.
        out_dir: Директория для сохранения результатов.

    Returns:
        Словарь с метриками тестовой выборки.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== Эксперимент для %s ===", ticker)

    # 1. Загрузка данных
    dcfg = DEFAULT_DATA_CONFIG
    df = fetch_moex_data(ticker, dcfg.start_date, dcfg.end_date)
    df = compute_technical_features(df)

    # 2. Формирование выборки
    X, y = build_classification_dataset(df, dcfg.window_size, dcfg.forecast_horizon)
    logger.info("Размер выборки: X=%s, y=%s", X.shape, y.shape)
    logger.info("Распределение классов: %s", dict(zip(*np.unique(y, return_counts=True))))

    # 3. Разделение с учётом временной структуры
    n = len(X)
    n_train = int(n * dcfg.train_ratio)
    n_val = int(n * dcfg.val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    # 4. Нормализация
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    train_ds = MoexDataset(X_train, y_train, scaler)
    val_ds = MoexDataset(X_val, y_val, scaler)
    test_ds = MoexDataset(X_test, y_test, scaler)

    tcfg = DEFAULT_TRAIN_CONFIG
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False)

    # 5. Модель
    mcfg = DEFAULT_MODEL_CONFIG
    model = ResNetTimeSeries(
        in_channels=mcfg.in_channels,
        num_classes=mcfg.num_classes,
        block_channels=mcfg.block_channels,
        kernel_size=mcfg.kernel_size,
        dropout=mcfg.dropout,
    )
    logger.info("Параметров модели: %d", model.count_parameters())

    # 6. Обучение
    save_path = os.path.join(out_dir, f"{ticker}_best.pth")
    history = train_model(model, train_loader, val_loader, tcfg, save_path=save_path)

    # 7. Загрузка лучшей модели и оценка
    model.load_state_dict(torch.load(save_path, map_location=tcfg.device))
    y_true, y_pred, y_prob = get_predictions(model, test_loader, tcfg.device)
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    logger.info("Тестовые метрики %s: %s", ticker, metrics)

    # 8. Визуализация
    plot_training_history(history, os.path.join(out_dir, f"{ticker}_history.png"))
    plot_confusion_matrix(y_true, y_pred, os.path.join(out_dir, f"{ticker}_cm.png"))
    plot_roc_curve(y_true, y_prob, os.path.join(out_dir, f"{ticker}_roc.png"))

    return metrics


if __name__ == "__main__":
    results = {}
    for ticker in DEFAULT_DATA_CONFIG.tickers:
        results[ticker] = run_experiment(ticker, out_dir="results")

    logger.info("Итоговые результаты по всем тикерам:")
    for ticker, metrics in results.items():
        logger.info("%s -> %s", ticker, metrics)
