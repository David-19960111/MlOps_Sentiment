import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
import uvicorn

app = FastAPI(title="Sentiment Classifier API", version="1.0.0")

#Cargar modelo al iniciar
MODEL_PATH = os.getenv("MODEL_PATH","models/model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
    
model = load_model()

#Schemas
class PredictRequest(BaseModel):
    text: str 

class PredictResponse(BaseModel):
    text: str 
    label: int 
    label_name: str 
    confidence: float 

LABEL_NAME = {0: "hockey", 1: "medicina"}

@app.get("/health")
def health():
    return {"status":"ok","model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacio")
    
    label = int(model.predict([request.text])[0])
    proba = model.predict_proba([request.text])[0]
    confidence = float(proba[label])

    return PredictResponse(
        text=request.text,
        label=label,
        label_name=LABEL_NAME[label],
        confidence=round(confidence, 4)
    )

@app.get("/")
def root():
    return {"message":"Sentiment Classifier API","docs":"/docs"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
