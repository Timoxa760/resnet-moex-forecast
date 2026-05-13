"""Модуль оценки качества прогнозов и визуализации результатов.

Автор: Миндрин Т.Д.
"""

import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from model import ResNetTimeSeries

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Рассчитывает комплекс метрик классификации.

    Args:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.
        y_prob: Вероятности положительного класса.

    Returns:
        Словарь с метриками.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }


@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Получает предсказания модели для выборки.

    Returns:
        Кортеж (истинные метки, предсказанные метки, вероятности).
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Строит и сохраняет матрицу ошибок."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Вниз", "Вверх"],
        yticklabels=["Вниз", "Вверх"],
        title="Матрица ошибок",
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Матрица ошибок сохранена: %s", save_path)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """Строит и сохраняет ROC-кривую."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("ROC-кривая сохранена: %s", save_path)


def plot_training_history(history: Dict[str, list], save_path: str) -> None:
    """Визуализирует динамику обучения."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Потери (CrossEntropy)")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].plot(history["val_f1"], label="Val F1")
    axes[1].set_title("Точность и F1-мера")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("График обучения сохранен: %s", save_path)
