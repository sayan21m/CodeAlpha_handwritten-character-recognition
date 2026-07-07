"""Shared utilities for model training scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_history(
    history: tf.keras.callbacks.History,
    save_path: str,
    title_prefix: str,
) -> None:
    """Save training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"{title_prefix} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"{title_prefix} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    target_names: list[str] | None = None,
    report_title: str = "Model",
) -> None:
    """Print classification report and confusion matrix shape."""
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in test_ds:
        probabilities = model(images, training=False).numpy()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    print(f"\n{report_title} — Classification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_true, y_pred)
    print("Confusion matrix shape:", matrix.shape)
