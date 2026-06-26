from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="API Credit Scoring", version="1.0")

# Chargement de la pipeline
PKL_FILENAME = "credit_scoring.pkl"
model_pipeline = None
if os.path.exists(PKL_FILENAME):
    with open(PKL_FILENAME, "rb") as f:
        model_pipeline = pickle.load(f)

class CreditInput(BaseModel):
    Income: float
    Seniority: float
    Price: float
    Amount: float
    Age: float
    Assets: float
    Expenses: float
    Records: float
    Time: float
    Job: float
    Debt: float
    Home: float
    Marital: float

@app.post("/predict/")
def predict(data: CreditInput):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    features = [[
        data.Seniority,
        data.Home,     
        data.Time,     
        data.Age,      
        data.Marital,  
        data.Records,  
        data.Job,      
        data.Expenses, 
        data.Income,   
        data.Assets,   
        data.Debt,     
        data.Amount,   
        data.Price     
    ]]

    try:
        prediction = model_pipeline.predict(features)
        proba = model_pipeline.predict_proba(features)
        result = int(prediction[0])
        
        return {
            "prediction": result,
            "message": "Crédit Accordé" if result == 1 else "Crédit Refusé",
            "probabilite_refus": round(proba[0][0], 3),
            "probabilite_accord": round(proba[0][1], 3)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))