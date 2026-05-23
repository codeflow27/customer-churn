from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model,cols = joblib.load(
    "models/model.pkl"
)

@app.get("/")
def home():

    return {
        "message":"Customer Churn API"
    }


@app.post("/predict")

def predict(x:dict):

    df = pd.DataFrame([x])

    df = df.reindex(
        columns=cols,
        fill_value=0
    )

    pred = model.predict(df)[0]

    return {
        "prediction":int(pred)
    }