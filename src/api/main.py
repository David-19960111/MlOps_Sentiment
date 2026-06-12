import pickle
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

app = FastAPI(title="Sentiment Classifier API", version="1.0.0")

# ── Métricas custom del modelo ──────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total de predicciones realizadas",
    ["label_name"]
)

CONFIDENCE_HISTOGRAM = Histogram(
    "model_confidence_score",
    "Distribución del confidence score",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Latencia de inferencia del modelo"
)

MODEL_CONFIDENCE_AVG = Gauge(
    "model_confidence_avg",
    "Confidence promedio de las últimas predicciones"
)

# ── Prometheus ──────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

# ── Modelo ──────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
model = None  # ✅ No cargar al importar

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# ✅ Cargar solo cuando la app levanta, no al importar
@app.on_event("startup")
async def startup_event():
    load_model()

# ── Schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: int
    label_name: str
    confidence: float

LABEL_NAMES = {0: "hockey", 1: "medicina"}

# ── Endpoints ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    start = time.time()

    label = int(model.predict([request.text])[0])
    proba = model.predict_proba([request.text])[0]
    confidence = float(proba[label])

    latency = time.time() - start

    PREDICTIONS_TOTAL.labels(label_name=LABEL_NAMES[label]).inc()
    CONFIDENCE_HISTOGRAM.observe(confidence)
    PREDICTION_LATENCY.observe(latency)
    MODEL_CONFIDENCE_AVG.set(confidence)

    return PredictResponse(
        text=request.text,
        label=label,
        label_name=LABEL_NAMES[label],
        confidence=round(confidence, 4)
    )

@app.get("/")
def root():
    return {"message": "Sentiment Classifier API", "docs": "/docs"}