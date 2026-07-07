"""Tests for label mapping and confidence helpers."""

import numpy as np
import pytest

from labels import (
    build_top_predictions,
    calculate_confidence,
    index_to_digit,
    index_to_letter,
)


def test_index_to_digit_valid():
    assert index_to_digit(0) == "0"
    assert index_to_digit(9) == "9"


def test_index_to_letter_valid():
    assert index_to_letter(0) == "A"
    assert index_to_letter(25) == "Z"


def test_calculate_confidence():
    probs = np.array([0.1, 0.7, 0.2])
    assert calculate_confidence(probs) == pytest.approx(70.0)


def test_build_top_predictions_digits():
    probs = np.zeros(10)
    probs[7] = 0.8
    probs[1] = 0.15
    probs[3] = 0.05
    top = build_top_predictions(probs, index_to_digit, top_k=3)
    assert top[0]["label"] == "7"
    assert len(top) == 3
