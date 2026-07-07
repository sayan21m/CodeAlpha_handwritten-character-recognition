"""Image preprocessing for canvas, upload, and camera inputs."""

from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (28, 28)
WHITE_BACKGROUND_THRESHOLD = 127
INK_THRESHOLD = 250
PADDING_RATIO = 0.20


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR NumPy array using OpenCV or Pillow."""
    if not image_bytes:
        raise ValueError("Empty image data received.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        return read_image_with_pillow(image_bytes)

    return image


def read_image_from_base64(image_data: str) -> np.ndarray:
    """Decode a base64 or data-URL string into a BGR NumPy array."""
    if not image_data:
        raise ValueError("Empty base64 image data received.")

    payload = image_data.strip()
    if payload.startswith("data:"):
        match = re.match(r"^data:image/[^;]+;base64,(.+)$", payload)
        if not match:
            raise ValueError("Invalid data URL format.")
        payload = match.group(1)

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image data.") from exc

    return read_image_from_bytes(image_bytes)


def read_image_from_pil(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to a BGR NumPy array."""
    rgb_image = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)


def read_image_with_pillow(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes with Pillow when OpenCV decoding fails."""
    with Image.open(BytesIO(image_bytes)) as pil_image:
        return read_image_from_pil(pil_image)


def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """Composite RGBA images onto a white background."""
    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        alpha = image[:, :, 3:4] / 255.0
        rgb = image[:, :, :3]
        white_background = np.ones_like(rgb, dtype=np.uint8) * 255
        composited = (rgb * alpha + white_background * (1.0 - alpha)).astype(np.uint8)
        return composited

    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a color image to a single-channel grayscale image."""
    if image.ndim == 2:
        return image

    if image.shape[2] == 1:
        return image[:, :, 0]

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_ink_bounding_box(image: np.ndarray, ink_threshold: int = INK_THRESHOLD) -> Optional[tuple[int, int, int, int]]:
    """Return bounding box (x, y, w, h) around dark ink pixels."""
    if image.ndim == 3:
        gray = convert_to_grayscale(image)
    else:
        gray = image

    ink_mask = gray < ink_threshold
    if not np.any(ink_mask):
        return None

    coords = np.argwhere(ink_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def crop_to_bounding_box(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop an image to the given bounding box."""
    x, y, width, height = bbox
    return image[y : y + height, x : x + width]


def pad_to_square(image: np.ndarray, padding_ratio: float = PADDING_RATIO) -> np.ndarray:
    """Pad a grayscale image to a square with proportional margin around the content."""
    height, width = image.shape[:2]
    max_side = max(height, width)
    pad = int(max_side * padding_ratio)
    square_size = max_side + pad * 2

    square = np.full((square_size, square_size), 255, dtype=image.dtype)
    offset_y = (square_size - height) // 2
    offset_x = (square_size - width) // 2
    square[offset_y : offset_y + height, offset_x : offset_x + width] = image
    return square


def center_content(image: np.ndarray) -> np.ndarray:
    """Crop to ink bounding box, pad to square, matching MNIST-style centering."""
    bbox = find_ink_bounding_box(image)
    if bbox is None:
        return image

    cropped = crop_to_bounding_box(image, bbox)
    return pad_to_square(cropped)


def resize_image(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize a grayscale image to the target shape using area interpolation."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to the range [0, 1] as float32."""
    return image.astype(np.float32) / 255.0


def invert_if_needed(image: np.ndarray) -> np.ndarray:
    """Invert colors when the background is darker than the foreground."""
    if np.mean(image) < WHITE_BACKGROUND_THRESHOLD:
        return 255 - image
    return image


def expand_dimensions(image: np.ndarray) -> np.ndarray:
    """Add batch and channel dimensions for CNN input: (1, 28, 28, 1)."""
    if image.ndim == 2:
        return image.reshape(1, image.shape[0], image.shape[1], 1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.expand_dims(image, axis=0)
    raise ValueError("Expected a 2D grayscale image before expanding dimensions.")


def tensor_to_preview_base64(tensor: np.ndarray) -> str:
    """Convert a model input tensor to a base64 PNG data URL for UI preview."""
    preview = (tensor[0, :, :, 0] * 255).astype(np.uint8)
    success, buffer = cv2.imencode(".png", preview)
    if not success:
        raise ValueError("Failed to encode preprocessed preview.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def preprocess_image_source(
    image_source: Union[bytes, str],
    return_preview: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, str]]:
    """
    Run the full preprocessing pipeline and return a model-ready tensor.

    Supports raw bytes (upload/camera) and base64 strings (canvas JSON payload).
    """
    if isinstance(image_source, str):
        image = read_image_from_base64(image_source)
    else:
        image = read_image_from_bytes(image_source)

    image = remove_alpha_channel(image)
    image = convert_to_grayscale(image)
    image = center_content(image)
    image = resize_image(image)
    image = invert_if_needed(image)
    image = normalize_pixels(image)
    tensor = expand_dimensions(image)

    if return_preview:
        return tensor, tensor_to_preview_base64(tensor)
    return tensor
