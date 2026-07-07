"""Prediction logic for digit and character recognition models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from labels import (
    LOW_CONFIDENCE_THRESHOLD,
    build_top_predictions,
    calculate_confidence,
    index_to_digit,
    index_to_letter,
)
from model import (
    CHARACTER_MODEL_PATH,
    DIGIT_MODEL_PATH,
    load_saved_model,
    model_is_available,
)
from preprocess import preprocess_image_source

if TYPE_CHECKING:
    import tensorflow as tf

_digit_model: Optional[Any] = None
_character_model: Optional[Any] = None


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
