"""Tests for image preprocessing."""

import base64

import cv2
import numpy as np
import pytest

from preprocess import (
    center_content,
    expand_dimensions,
    invert_if_needed,
    normalize_pixels,
    preprocess_image_source,
    read_image_from_bytes,
    resize_image,
    tensor_to_preview_base64,
)


def _blank_png_bytes() -> bytes:
    image = np.full((28, 28), 255, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def _digit_png_bytes() -> bytes:
    image = np.full((56, 56), 255, dtype=np.uint8)
    cv2.rectangle(image, (18, 10), (38, 46), 0, thickness=-1)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_read_image_from_bytes_valid_png():
    image = read_image_from_bytes(_blank_png_bytes())
    assert image is not None
    assert image.ndim in (2, 3)


def test_read_image_from_bytes_empty_raises():
    with pytest.raises(ValueError, match="Empty"):
        read_image_from_bytes(b"")


def test_invert_if_needed_dark_background():
    dark = np.zeros((28, 28), dtype=np.uint8)
    inverted = invert_if_needed(dark)
    assert np.mean(inverted) > 200


def test_center_content_finds_ink():
    image = np.full((100, 100), 255, dtype=np.uint8)
    cv2.circle(image, (60, 40), 10, 0, thickness=-1)
    centered = center_content(image)
    assert centered.shape[0] == centered.shape[1]


def test_preprocess_pipeline_output_shape():
    tensor = preprocess_image_source(_digit_png_bytes())
    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32


def test_preprocess_base64_round_trip():
    encoded = base64.b64encode(_digit_png_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    tensor = preprocess_image_source(data_url)
    assert tensor.shape == (1, 28, 28, 1)


def test_tensor_preview_base64():
    gray = normalize_pixels(resize_image(np.zeros((28, 28), dtype=np.uint8)))
    tensor = expand_dimensions(gray)
    preview = tensor_to_preview_base64(tensor)
    assert preview.startswith("data:image/png;base64,")
