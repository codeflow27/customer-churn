# Customer Churn MLOps Pipeline

End-to-end ML pipeline to predict telecom customer churn and serve predictions through an API.

## Stack

Python • Scikit-learn • MLflow • FastAPI • Git

## Pipeline

```text
Data → Preprocessing → Training → MLflow → Save Model → API → Prediction
```

## Model

Algorithm:

```text
RandomForestClassifier
```

Performance:

```text
ROC AUC = 0.835
```

## Features

- Experiment tracking with MLflow
- Model persistence using Joblib
- FastAPI prediction endpoint
- Swagger API docs
- Git version control

## Run

Train:

```bash
python src/train.py
```

Start API:

```bash
uvicorn src.api:app --reload
```

MLflow:

```bash
mlflow ui
```

## Screenshots

(Add MLflow + FastAPI screenshots)

## Improvements planned

- Docker
- CI/CD
- Deployment
- Monitoring