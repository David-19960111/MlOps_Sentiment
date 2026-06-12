import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import os

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train():
    params = load_params()
    
    # Cargar datos
    train_df = pd.read_csv("data/processed/train.csv").dropna()
    test_df  = pd.read_csv("data/processed/test.csv").dropna()

    X_train, y_train = train_df["text"], train_df["label"]
    X_test,  y_test  = test_df["text"],  test_df["label"]

    # MLflow: configurar experimento
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("sentiment-classifier")

    with mlflow.start_run():
        # Loggear parámetros
        mlflow.log_params(params["model"])

        # Pipeline: TF-IDF + Logistic Regression
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=params["model"]["max_features"],
                ngram_range=tuple(params["model"]["ngram_range"])
            )),
            ("clf", LogisticRegression(
                C=params["model"]["C"],
                max_iter=params["model"]["max_iter"]
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Métricas
        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
        }

        # Loggear métricas
        mlflow.log_metrics(metrics)

        # Loggear modelo
        mlflow.sklearn.log_model(pipeline, "model")

        # Guardar localmente también
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        print("\n📊 Resultados:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        
        print(f"\n✅ Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train()