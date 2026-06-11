import pandas as pd
import pickle
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import *

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

def generate_reference_data():
    """Datos de entrenamiento = referencia (lo que el modelo 'espera')"""
    df = pd.read_csv("data/processed/train.csv").dropna()
    return df.sample(200, random_state=42)  # muestra representativa

def simulate_production_data(drift: bool = False):
    """
    Simula datos que llegan en producción.
    drift=True  → inyecta textos raros para simular degradación
    drift=False → datos normales
    """
    if drift:
        # Textos fuera de distribución (simula drift real)
        texts = [
            "cryptocurrency bitcoin blockchain investment",
            "cooking recipe pasta italian food",
            "weather forecast rain temperature",
            "stock market trading nasdaq dow jones",
            "fashion clothing shoes accessories",
        ] * 40
        labels = [0, 1, 0, 1, 0] * 40
    else:
        df = pd.read_csv("data/processed/test.csv").dropna()
        return df.sample(200, random_state=99)

    return pd.DataFrame({"text": texts, "label": labels})

def add_predictions(df, model):
    """Agrega predicciones del modelo al dataframe"""
    df = df.copy()
    df["prediction"] = model.predict(df["text"].fillna(""))
    df["text_length"] = df["text"].str.len().fillna(0)
    df["word_count"]  = df["text"].str.split().str.len().fillna(0)
    return df

def run_monitoring(drift_simulation: bool = False):
    print(f"🔍 Ejecutando monitoreo (drift={'simulado' if drift_simulation else 'normal'})...")

    model = load_model()

    reference = add_predictions(generate_reference_data(), model)
    current   = add_predictions(simulate_production_data(drift=drift_simulation), model)

    os.makedirs("reports", exist_ok=True)

    # ── Reporte de Data Drift ──────────────────────────────
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference[["text_length", "word_count", "label"]],
        current_data=current[["text_length", "word_count", "label"]]
    )
    drift_report.save_html("reports/drift_report.html")
    print("✅ Reporte de drift guardado en reports/drift_report.html")

    # ── Reporte de Calidad del Modelo ──────────────────────
    model_report = Report(metrics=[ClassificationPreset()])
    model_report.run(
        reference_data=reference[["text_length", "word_count", "label", "prediction"]]\
                       .rename(columns={"label": "target"}),
        current_data=current[["text_length", "word_count", "label", "prediction"]]\
                     .rename(columns={"label": "target"})
    )
    model_report.save_html("reports/model_report.html")
    print("✅ Reporte de modelo guardado en reports/model_report.html")

    # ── Resumen en consola ─────────────────────────────────
    drift_result = drift_report.as_dict()
    drifted = drift_result["metrics"][0]["result"]["number_of_drifted_columns"]
    total   = drift_result["metrics"][0]["result"]["number_of_columns"]
    print(f"\n📊 Columnas con drift: {drifted}/{total}")

    if drifted > 0:
        print("⚠️  ALERTA: Se detectó data drift — revisar el modelo")
    else:
        print("✅  Sin drift detectado — modelo estable")

if __name__ == "__main__":
    import sys
    drift = "--drift" in sys.argv
    run_monitoring(drift_simulation=drift)