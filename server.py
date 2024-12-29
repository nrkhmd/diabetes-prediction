from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model dynamically
model_path = os.path.join(os.path.dirname(__file__), "diabetes_model_fixed.sav")
try:
    scaler, diabetes_model = pickle.load(open(model_path, "rb"))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", e)
    raise

# Initialize FastAPI app
app = FastAPI()

# Route for index.html
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        with open(html_path, "r") as file:
            return file.read()
    except Exception as e:
        logging.error("Error reading index.html: %s", e)
        return {"error": "index.html not found"}

# Route for prediction
@app.post("/predict/")
async def predict(
    pregnancies: int = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetes_pedigree_function: float = Form(...),
    age: int = Form(...),
):
    try:
        # Scale the input values
        scaled_data = scaler.transform(
            [
                [
                    pregnancies,
                    glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    bmi,
                    diabetes_pedigree_function,
                    age,
                ]
            ]
        )
        # Perform prediction
        prediction = diabetes_model.predict(scaled_data)

        return {"prediction": "Positive" if prediction[0] == 1 else "Negative"}
    except Exception as e:
        logging.error("Prediction error: %s", e)
        return {"error": "Prediction failed"}
