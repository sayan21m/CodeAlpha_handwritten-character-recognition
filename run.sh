#!/usr/bin/env bash
# Start backend and frontend for local development (Python 3.11+)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
DOCS_DIR="$ROOT_DIR/docs"

echo "=== Handwritten Character Recognition System ==="

if [ ! -d "$BACKEND_DIR/saved_models" ] || \
   { [ ! -f "$BACKEND_DIR/saved_models/digit_model.keras" ] && \
     [ ! -f "$BACKEND_DIR/saved_models/mnist_cnn.keras" ]; }; then
  echo "Warning: No trained digit model found."
  echo "Run: cd backend && python train.py --model digit"
fi

echo "Starting Flask backend on http://localhost:5000"
(cd "$BACKEND_DIR" && python app.py) &
BACKEND_PID=$!

sleep 2

echo "Starting frontend on http://localhost:8080"
(cd "$ROOT_DIR" && python -m http.server 8080 --directory docs) &
FRONTEND_PID=$!

cleanup() {
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Press Ctrl+C to stop both servers."
wait
