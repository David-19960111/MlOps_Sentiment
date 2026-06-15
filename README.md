# 🚀 MLOps Sentiment Analysis API

API de análisis de sentimientos desarrollada con FastAPI y desplegada siguiendo prácticas de MLOps y DevSecOps.

## 📋 Descripción

Este proyecto implementa un modelo de Machine Learning capaz de clasificar texto según su sentimiento.

La solución incluye:

- API REST desarrollada con FastAPI
- Modelo de Machine Learning serializado con Pickle
- Contenerización con Docker
- Pipeline CI/CD con GitHub Actions
- Escaneo SAST con Bandit
- Escaneo SCA con pip-audit
- Escaneo de imágenes Docker con Trivy
- Escaneo DAST con OWASP ZAP
- Observabilidad con Prometheus
- Tracking de experimentos con MLflow

---

## 🏗️ Arquitectura

```text
Usuario
   │
   ▼
FastAPI
   │
   ▼
Modelo ML (Pickle)
   │
   ├── Prometheus Metrics
   ├── MLflow Tracking
   └── Docker Container

GitHub Actions
   │
   ├── Unit Tests
   ├── SAST (Bandit)
   ├── SCA (pip-audit)
   ├── Trivy Image Scan
   ├── OWASP ZAP Scan
   └── Docker Hub Push
```

---

## 📂 Estructura del proyecto

```text
.
├── app/
│   ├── api.py
│   ├── train.py
│   └── requirements.txt
│
├── models/
│   └── model.pkl
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── Dockerfile
├── README.md
└── .gitignore
```

---

## ⚙️ Tecnologías utilizadas

- Python 3.11
- FastAPI
- Scikit-Learn
- Docker
- GitHub Actions
- MLflow
- Prometheus
- Trivy
- Bandit
- pip-audit
- OWASP ZAP

---

## 🚀 Instalación local

### Clonar repositorio

```bash
git clone https://github.com/usuario/mlops-sentiment.git

cd mlops-sentiment
```

### Crear entorno virtual

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🧠 Entrenar modelo

```bash
python train.py
```

El modelo generado se almacenará en:

```text
models/model.pkl
```

---

## ▶️ Ejecutar API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Documentación Swagger:

```text
http://localhost:8000/docs
```

---

## 🐳 Docker

### Construir imagen

```bash
docker build -t sentiment-api .
```

### Ejecutar contenedor

```bash
docker run -p 8000:8000 sentiment-api
```

---

## 📡 Endpoint principal

### POST /predict

Request:

```json
{
  "text": "I love this product"
}
```

Response:

```json
{
  "prediction": "positive"
}
```

---

## 📊 Métricas Prometheus

Endpoint:

```text
/metrics
```

Métricas disponibles:

- api_requests_total
- prediction_requests_total
- prediction_latency_seconds
- model_loaded
- prediction_errors_total

---

## 🔬 MLflow

Ejecutar servidor:

```bash
mlflow ui
```

Acceso:

```text
http://localhost:5000
```

Permite visualizar:

- Experimentos
- Parámetros
- Métricas
- Artefactos

---

## 🔒 Seguridad

### SAST

Bandit

```bash
bandit -r .
```

### SCA

pip-audit

```bash
pip-audit
```

### Container Security

Trivy

```bash
trivy image sentiment-api
```

### DAST

OWASP ZAP

Ejecutado automáticamente desde GitHub Actions.

---

## 🧪 Testing

Ejecutar pruebas:

```bash
pytest -v
```

---

## 🔄 Pipeline CI/CD

El pipeline ejecuta automáticamente:

1. Unit Tests
2. SAST (Bandit)
3. SCA (pip-audit)
4. Build Docker
5. Trivy Scan
6. OWASP ZAP Scan
7. Push a Docker Hub

---

## 📈 Resultados

- API REST funcional
- Pipeline DevSecOps automatizado
- Seguridad integrada en CI/CD
- Observabilidad con Prometheus
- Gestión de experimentos con MLflow

---

## 👨‍💻 Autor

David Rojas

DevOps | DevSecOps | Cloud Engineer

AWS • Docker • Kubernetes • Terraform • GitHub Actions • Python
