"""Tests for Flask API routes."""

import pytest

pytest.importorskip("tensorflow")

from app import app  # noqa: E402


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "running"
    assert "Handwritten Character Recognition" in data["project"]


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "digit_model" in data
    assert "character_model" in data


def test_predict_digit_without_image(client):
    response = client.post("/predict-digit")
    assert response.status_code in (400, 503)
    data = response.get_json()
    assert "error" in data


def test_not_found_returns_json(client):
    response = client.get("/unknown-route")
    assert response.status_code == 404
    assert response.get_json()["error"]
