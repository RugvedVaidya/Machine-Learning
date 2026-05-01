from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

model = joblib.load("churn_model.joblib")

app = FastAPI()

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Churn API Running"}

@app.post("/predict")
def predict(data: Customer):
    try:
        input_df = pd.DataFrame([data.dict()])
        input_df = input_df.replace({None: np.nan})

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return {
            "churn_prediction": int(prediction),
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}