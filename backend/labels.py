"""Label mapping and confidence helpers (no TensorFlow dependency)."""

from __future__ import annotations

import numpy as np

DIGITS = [str(i) for i in range(10)]
LETTERS = [chr(ord("A") + i) for i in range(26)]
TOP_K = 3
LOW_CONFIDENCE_THRESHOLD = 70.0


def index_to_digit(class_index: int) -> str:
    """Convert a class index to a digit label."""
    if class_index < 0 or class_index >= len(DIGITS):
        raise ValueError(f"Invalid digit class index: {class_index}")
    return DIGITS[class_index]


def index_to_letter(class_index: int) -> str:
    """Convert a class index to an uppercase letter label."""
    if class_index < 0 or class_index >= len(LETTERS):
        raise ValueError(f"Invalid character class index: {class_index}")
    return LETTERS[class_index]


def calculate_confidence(probabilities: np.ndarray) -> float:
    """Return the highest softmax probability as a percentage."""
    return float(np.max(probabilities) * 100.0)


def build_top_predictions(
    probabilities: np.ndarray,
    label_converter,
    top_k: int = TOP_K,
) -> list[dict]:
    """Build top-k prediction list with labels and confidence scores."""
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {
            "label": label_converter(int(index)),
            "confidence": round(float(probabilities[index] * 100.0), 2),
        }
        for index in indices
    ]
