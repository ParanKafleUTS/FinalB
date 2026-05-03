"""Automated tests for the Banana Ripeness Classifier FastAPI backend."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient for the FastAPI app."""
    from main import app  # imported here so the module is only loaded once
    return TestClient(app)


def _make_jpeg_bytes(color: tuple = (255, 230, 0), size: tuple = (224, 224)) -> bytes:
    """Create a minimal JPEG image in memory and return its bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json().get("status") == "ok"


# ---------------------------------------------------------------------------
# /predict endpoint
# ---------------------------------------------------------------------------

VALID_MODELS = ["efficientnet", "efficientnetv2", "mobilenet", "resnet"]


class TestPredictEndpoint:
    def test_predict_default_model(self, client):
        """POST /predict with a valid image and default model returns a prediction."""
        resp = client.post(
            "/predict",
            files={"file": ("banana.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert data["prediction"] in {"unripe", "ripe", "overripe", "rotten"}

    @pytest.mark.parametrize("model_name", VALID_MODELS)
    def test_predict_each_model(self, client, model_name):
        """Each supported model should return a valid ripeness label."""
        resp = client.post(
            "/predict",
            data={"model": model_name},
            files={"file": ("banana.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert data["prediction"] in {"unripe", "ripe", "overripe", "rotten"}

    def test_predict_returns_confidence(self, client):
        """Response should contain a confidence score between 0 and 1."""
        resp = client.post(
            "/predict",
            files={"file": ("banana.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_no_file_returns_422(self, client):
        """Missing file should return HTTP 422 Unprocessable Entity."""
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_predict_invalid_model_returns_error(self, client):
        """An unknown model name should return an error response."""
        resp = client.post(
            "/predict",
            data={"model": "nonexistent_model"},
            files={"file": ("banana.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        # Accept either 400 Bad Request or 422 Unprocessable Entity
        assert resp.status_code in {400, 422}

    def test_predict_non_image_file_returns_error(self, client):
        """Uploading a non-image file should return an error response."""
        resp = client.post(
            "/predict",
            files={"file": ("data.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code in {400, 415, 422}
