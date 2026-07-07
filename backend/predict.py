"""Prediction logic for digit and character recognition models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from model import (
    CHARACTER_MODEL_PATH,
    DIGIT_MODEL_PATH,
    load_saved_model,
    model_is_available,
)
from preprocess import preprocess_image_source

_digit_model: Optional[Any] = None
_character_model: Optional[Any] = None
LOW_CONFIDENCE_THRESHOLD = 70.0

DIGITS = [str(i) for i in range(10)]
LETTERS = [chr(ord("A") + i) for i in range(26)]


@dataclass
class PredictionResult:
    """Structured prediction output."""

    prediction: str
    confidence: float
    inference_time_ms: int
    top_predictions: list[dict] = field(default_factory=list)
    input_source: Optional[str] = None
    preprocessed_preview: Optional[str] = None
    low_confidence: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the result to a JSON-serializable dictionary."""
        payload = {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 2),
            "inference_time": f"{self.inference_time_ms} ms",
            "inference_time_ms": self.inference_time_ms,
            "top_predictions": self.top_predictions,
            "low_confidence": self.low_confidence,
        }
        if self.input_source:
            payload["input_source"] = self.input_source
        if self.preprocessed_preview:
            payload["preprocessed_preview"] = self.preprocessed_preview
        if self.warning:
            payload["warning"] = self.warning
        return payload


def load_digit_model() -> Any:
    """Load the digit model independently."""
    global _digit_model
    if _digit_model is None:
        _digit_model = load_saved_model(DIGIT_MODEL_PATH)
    return _digit_model


def load_character_model() -> Any:
    """Load the character model independently."""
    global _character_model
    if _character_model is None:
        _character_model = load_saved_model(CHARACTER_MODEL_PATH)
    return _character_model


def load_models() -> None:
    """Load both models when available."""
    if model_is_available(DIGIT_MODEL_PATH):
        load_digit_model()
    if model_is_available(CHARACTER_MODEL_PATH):
        load_character_model()


def get_model_status() -> dict:
    """Return availability status for health checks."""
    return {
        "digit_model": model_is_available(DIGIT_MODEL_PATH),
        "character_model": model_is_available(CHARACTER_MODEL_PATH),
        "digit_model_loaded": _digit_model is not None,
        "character_model_loaded": _character_model is not None,
    }


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
    top_k: int = 3,
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


def _run_inference(model: Any, tensor: np.ndarray) -> np.ndarray:
    """Run a single forward pass without predict() overhead."""
    output = model(tensor, training=False)
    return output.numpy()[0]


def _run_prediction(
    model: Any,
    image_source: Union[bytes, str],
    label_converter,
    input_source: Optional[str] = None,
) -> PredictionResult:
    """Shared prediction workflow for both models."""
    tensor, preview = preprocess_image_source(image_source, return_preview=True)

    start_time = time.perf_counter()
    probabilities = _run_inference(model, tensor)
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    class_index = int(np.argmax(probabilities))
    confidence = calculate_confidence(probabilities)
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

    warning = None
    if low_confidence:
        warning = (
            f"Low confidence ({confidence:.1f}%). "
            "Try redrawing larger and more centered."
        )

    return PredictionResult(
        prediction=label_converter(class_index),
        confidence=confidence,
        inference_time_ms=elapsed_ms,
        top_predictions=build_top_predictions(probabilities, label_converter),
        input_source=input_source,
        preprocessed_preview=preview,
        low_confidence=low_confidence,
        warning=warning,
    )


def predict_digit(
    image_source: Union[bytes, str],
    input_source: Optional[str] = None,
) -> PredictionResult:
    """Predict a handwritten digit from image bytes or base64 data."""
    return _run_prediction(
        load_digit_model(),
        image_source,
        index_to_digit,
        input_source=input_source,
    )


def predict_character(
    image_source: Union[bytes, str],
    input_source: Optional[str] = None,
) -> PredictionResult:
    """Predict a handwritten letter from image bytes or base64 data."""
    return _run_prediction(
        load_character_model(),
        image_source,
        index_to_letter,
        input_source=input_source,
    )
