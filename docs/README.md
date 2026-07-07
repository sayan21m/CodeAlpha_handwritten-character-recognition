# Handwritten Character Recognition — Frontend

Modern, responsive web UI for recognizing handwritten digits (MNIST) and English letters (EMNIST Letters).

## Tech Stack

- HTML5 (semantic markup)
- CSS3 (custom properties, flexbox, grid)
- Vanilla JavaScript (no frameworks)

## Running Locally

```bash
# From project root
./run.sh

# Or manually
python -m http.server 8080 --directory docs
```

> API calls require the Flask backend on port 5000. Configure `docs/config.js` if needed.

## Configuration

Edit `docs/config.js`:

```javascript
window.HCR_CONFIG = {
  API_BASE_URL: "http://localhost:5000",
  LOW_CONFIDENCE_THRESHOLD: 70,
};
```

The API URL auto-detects `localhost` when served locally.

## Features

- Model selection (Digit / Character)
- **Three input methods:** canvas, upload, camera (one active at a time)
- Camera center-crop guide and square capture region
- **Top-3 predictions** and **28×28 preprocessed preview**
- **Low-confidence warnings** (&lt; 70%)
- API error messages shown in toasts
- Responsive, keyboard-accessible layout

## API Integration

See the root [README.md](../README.md) for full API documentation.

## License

CodeAlpha Internship Project.


## Live URLs

- Frontend: https://sayan21m.github.io/CodeAlpha_handwritten-character-recognition/
- Backend API: https://codealpha-handwritten-character-3jhq.onrender.com
