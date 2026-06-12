import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

# ✅ Mockear el modelo ANTES de importar la app
mock_model = MagicMock()
mock_model.predict.return_value = [0]
mock_model.predict_proba.return_value = [np.array([0.9, 0.1])]

with patch("src.api.main.load_model"):
    with patch("src.api.main.model", mock_model):
        from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid():
    with patch("src.api.main.model", mock_model):
        response = client.post("/predict", json={"text": "The player scored an amazing goal"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_empty():
    with patch("src.api.main.model", mock_model):
        response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400

def test_predict_response_schema():
    with patch("src.api.main.model", mock_model):
        response = client.post("/predict", json={"text": "surgery and medicine advances"})
    data = response.json()
    assert set(data.keys()) == {"text", "label", "label_name", "confidence"}