# Handwritten Character Recognition System

Clean and simple project structure with three main parts:

- `analysis/` - Jupyter notebooks for training and experimentation
- `backend/` - Flask API for inference
- `docs/` - frontend (HTML/CSS/Vanilla JS)

## Project Structure

```text
handwritten_character_recognition/
├── analysis/
│   ├── model_1.ipynb
│   └── model_2.ipynb
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── requirements.txt
│   └── saved_models/
└── docs/
```

## Run Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend URL: `http://localhost:5000`

## Run Frontend

```bash
python -m http.server 8080 --directory docs
```

Frontend URL: `http://localhost:8080`

## Models

## Model Source

- Training happens in `analysis/` notebooks only.
- `backend/` is inference-only and does not train models.


Put pretrained models in `backend/saved_models/`:

- `digit_model.keras`
- `character_model.keras`

Legacy names are also supported:

- `mnist_cnn.keras`
- `emnist_cnn.keras`

## API Endpoints

- `GET /`
- `GET /health`
- `POST /predict-digit`
- `POST /predict-character`
