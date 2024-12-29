from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import os
import numpy as np
import logging
import gdown  # Untuk mengunduh file dari Google Drive

# Configure logging
logging.basicConfig(level=logging.INFO)

# Google Drive file ID dari URL baru
MODEL_FILE_ID = "1XF9Z8r6JH6N0vEPxmZgK9UfHa3V_PSub"
MODEL_FILE_NAME = "diabetes_model_fixed.sav"

def download_model():
    """Download model file from Google Drive if not exists locally."""
    if not os.path.exists(MODEL_FILE_NAME):
        try:
            # Membuat URL unduhan langsung dari Google Drive
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_FILE_NAME, quiet=False)
            logging.info("Model downloaded successfully.")
        except Exception as e:
            logging.error("Error downloading model: %s", e)
            raise
    else:
        logging.info("Model already exists locally.")

# Download the model
download_model()

# Load the model dynamically
try:
    scaler, diabetes_model = pickle.load(open(MODEL_FILE_NAME, "rb"))
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
