import numpy as np 
import pandas as pd
import joblib
from fastapi import FastAPI

model = joblib.load('customer_churn_model.joblib')

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Customer Churn Prediction API!"}

@app.post("/predict")
def predict(data: dict):
    
    input_df = pd.DataFrame([data])
    input_df = input_df.replace({None: np.nan})
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    return {"prediction": str(prediction), "probability_of_churn": float(prob)}