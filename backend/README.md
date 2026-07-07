# Backend — Handwritten Character Recognition API

Flask REST API for digit and character inference.

**Requires Python 3.11+**

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train Models

```bash
python train.py --model both      # digit + character
python train.py --model digit     # MNIST only
python train.py --model character # EMNIST only
```

Saved models:

| File | Dataset | Classes |
|------|---------|---------|
| `saved_models/digit_model.keras` | MNIST | 0–9 |
| `saved_models/character_model.keras` | EMNIST Letters | A–Z |

## Run Server

```bash
python app.py
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `0.0.0.0` | Bind host |
| `FLASK_PORT` | `5000` | Bind port |
| `FLASK_DEBUG` | `false` | Debug mode |

## API

### `GET /health`

```json
{
  "status": "healthy",
  "digit_model": true,
  "character_model": true,
  "digit_model_loaded": true,
  "character_model_loaded": true
}
```

### `POST /predict-digit` / `POST /predict-character`

**multipart/form-data**

```
image: <file>
input_source: canvas | upload | camera  (optional)
```

**application/json**

```json
{
  "image": "data:image/png;base64,...",
  "input_source": "canvas"
}
```

Max upload size: **5 MB**

## Module Overview

| Module | Purpose |
|--------|---------|
| `app.py` | Flask routes, CORS, error handling |
| `model.py` | CNN architectures, model loading |
| `preprocess.py` | Image preprocessing pipeline |
| `predict.py` | Inference logic |
| `training_utils.py` | Shared training helpers |
| `train_digit.py` | MNIST training |
| `train_character.py` | EMNIST training |
| `train.py` | Unified training CLI |

## Tests

```bash
pytest tests/ -v
```

## Preprocessing Pipeline

1. Decode image (OpenCV → Pillow fallback)
2. Remove alpha channel
3. Convert to grayscale
4. **Center content** (bounding box + square padding)
5. Resize to 28×28
6. Invert if dark background
7. Normalize to [0, 1]
8. Expand to (1, 28, 28, 1)
