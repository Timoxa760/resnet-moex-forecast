"""Процедура обучения и валидации остаточной сети.

Автор: Миндрин Т.Д.
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from model import ResNetTimeSeries

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Ранняя остановка при отсутствии улучшения валидационной метрики."""

    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str
) -> Tuple[float, float]:
    """Одна эпоха обучения.

    Returns:
        Средние значения потерь и точности.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: str
) -> Tuple[float, float, float]:
    """Оценка модели на валидационной или тестовой выборке.

    Returns:
        Потери, точность и F1-мера.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = accuracy_score(all_labels, all_preds)
    avg_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, avg_acc, avg_f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    save_path: str = "best_model.pth",
) -> dict:
    """Полный цикл обучения с ранней остановкой.

    Args:
        model: Модель ResNet.
        train_loader: Загрузчик обучающей выборки.
        val_loader: Загрузчик валидационной выборки.
        config: Конфигурация обучения.
        save_path: Путь для сохранения лучшей модели.

    Returns:
        Словарь с историей метрик.
    """
    device = config.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stop = EarlyStopping(patience=config.patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        logger.info(
            "Эпоха %03d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f val_f1=%.3f",
            epoch, train_loss, train_acc, val_loss, val_acc, val_f1,
        )

        if early_stop(val_loss):
            logger.info("Ранняя остановка на эпохе %d", epoch)
            break

    return history
