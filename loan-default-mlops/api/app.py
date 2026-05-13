import os
import sys

# Add the parent directory (loan-default-mlops) to sys.path so 'api.schema' can be imported when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from api.schema import LoanApplication

# Load the best model
# Define the path to the model relative to the execution directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pkl")

# We will load the model when the app starts
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Note: We don't raise an exception here so the app can still start 
        # and show an error when the prediction endpoint is called, 
        # if the user hasn't trained the model yet.
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="A simple API to predict loan defaults using a trained machine learning model.",
    version="1.0",
    lifespan=lifespan
)

@app.get("/")
def home():
    """
    Root endpoint.
    """
    return {"message": "Welcome to the Loan Default Prediction API. Visit /docs for documentation."}

@app.post("/predict")
def predict_default(application: LoanApplication):
    """
    Endpoint to make predictions.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please train and save the model first.")
    
    try:
        # Convert input data to a dictionary and then to a DataFrame
        input_data = application.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] # Probability of class 1 (default)
        
        # Determine result
        result = "Default" if prediction[0] == 1 else "No Default"
        
        return {
            "prediction": int(prediction[0]),
            "result_text": result,
            "probability_of_default": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {e}")
