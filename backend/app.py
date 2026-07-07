"""Flask application for the Handwritten Character Recognition System."""

from __future__ import annotations

import base64
import binascii
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from model import CHARACTER_MODEL_PATH, DIGIT_MODEL_PATH, model_is_available
from predict import (
    get_model_status,
    load_character_model,
    load_digit_model,
    predict_character,
    predict_digit,
)

app = Flask(__name__)
CORS(app)

PROJECT_NAME = "Handwritten Character Recognition"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


def create_app() -> Flask:
    """Application factory used for testing or deployment."""
    return app


def extract_image_from_request():
    """
    Extract image data from multipart/form-data or JSON requests.

    Supports:
      - multipart/form-data field: image
      - application/json field: image (base64 or data URL)
    """
    if "image" in request.files:
        file_storage = request.files["image"]
        if file_storage:
            image_bytes = file_storage.read()
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Image exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                )
            if image_bytes:
                return image_bytes

    json_payload = request.get_json(silent=True)
    if json_payload and json_payload.get("image"):
        image_value = json_payload["image"]
        if isinstance(image_value, str) and len(image_value) > MAX_UPLOAD_BYTES * 2:
            raise ValueError("Image payload is too large.")
        return image_value

    if request.form.get("image"):
        image_value = request.form["image"]
        if image_value.startswith("data:"):
            return image_value
        try:
            return base64.b64decode(image_value)
        except (ValueError, binascii.Error):
            raise ValueError("Invalid base64 image data in form field.")

    return None


def get_input_source() -> str | None:
    """Read optional input source metadata from the request."""
    json_payload = request.get_json(silent=True) or {}
    return json_payload.get("input_source") or request.form.get("input_source")


@app.route("/", methods=["GET"])
def root():
    """Return basic service status."""
    return jsonify(
        {
            "status": "running",
            "project": PROJECT_NAME,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint with model availability."""
    status = get_model_status()
    overall = "healthy" if any(
        [status["digit_model"], status["character_model"]]
    ) else "degraded"

    return jsonify(
        {
            "status": overall,
            **status,
        }
    )


@app.route("/predict-digit", methods=["POST"])
def predict_digit_endpoint():
    """Predict a handwritten digit from an uploaded or JSON-encoded image."""
    if not model_is_available(DIGIT_MODEL_PATH):
        return jsonify({"error": "Digit model is not available. Train it first."}), 503

    image_source = extract_image_from_request()
    if image_source is None:
        return jsonify({"error": "Invalid image. Provide an image file or base64 data."}), 400

    try:
        load_digit_model()
        result = predict_digit(image_source, input_source=get_input_source())
        return jsonify(result.to_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Digit prediction failed.")
        return jsonify({"error": "Internal server error."}), 500


@app.route("/predict-character", methods=["POST"])
def predict_character_endpoint():
    """Predict a handwritten character from an uploaded or JSON-encoded image."""
    if not model_is_available(CHARACTER_MODEL_PATH):
        return jsonify(
            {"error": "Character model is not available. Train it first."}
        ), 503

    image_source = extract_image_from_request()
    if image_source is None:
        return jsonify({"error": "Invalid image. Provide an image file or base64 data."}), 400

    try:
        load_character_model()
        result = predict_character(image_source, input_source=get_input_source())
        return jsonify(result.to_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Character prediction failed.")
        return jsonify({"error": "Internal server error."}), 500


@app.errorhandler(404)
def not_found(error):
    """Return JSON for unknown endpoints."""
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException):
    """Return JSON for HTTP errors."""
    return jsonify({"error": error.description}), error.code


@app.errorhandler(Exception)
def handle_unexpected_exception(error: Exception):
    """Return JSON for unexpected server errors."""
    app.logger.exception("Unhandled server error.")
    return jsonify({"error": "Internal server error."}), 500


def main() -> None:
    """Start the development server."""
    os.makedirs(os.path.join(os.path.dirname(__file__), "saved_models"), exist_ok=True)

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
