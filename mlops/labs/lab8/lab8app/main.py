from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="Currency Classifier API",
    description="API for classifying currency using MLflow model",
    version="0.1",
)

class CurrencyInput(BaseModel):
    values: list[float]

def load_model():
    global model
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri = "models:/currency_classifier-2025-04-21/2"
    model = mlflow.pyfunc.load_model(model_uri)

app.add_event_handler("startup", load_model)

@app.get("/")
def root():
    return {"message": "Welcome to the currency classification API!"}

@app.post("/predict")
def predict(input: CurrencyInput):
    # Turn input list into a DataFrame with the expected column names
    columns = ["variance", "skewness", "curtosis", "entropy"]
    df = pd.DataFrame([input.values], columns=columns)
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}

