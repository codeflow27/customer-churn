import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Customer_Churn")
import pandas as pd
import mlflow
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Convert target to numeric
df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})

# Remove customerID if present
if "customerID" in df.columns:
    df = df.drop("customerID",axis=1)

X = pd.get_dummies(
        df.drop("Churn",axis=1)
)

y = df["Churn"]

Xtr,Xte,ytr,yte = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(Xtr,ytr)

    probs = model.predict_proba(Xte)[:,1]

    auc = roc_auc_score(
        yte,
        probs
    )

    mlflow.log_metric(
        "roc_auc",
        auc
    )

    mlflow.log_param(
        "model",
        "RandomForest"
    )

    joblib.dump(
        (model,X.columns.tolist()),
        "models/model.pkl"
    )

print("Training complete")
print("ROC_AUC =",auc)