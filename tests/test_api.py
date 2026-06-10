from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid():
    response = client.post("/predict", json={"text": "The player scored an amazing goal"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400

def test_predict_response_schema():
    response = client.post("/predict", json={"text": "surgery and medicine advances"})
    data = response.json()
    assert set(data.keys()) == {"text", "label", "label_name", "confidence"}