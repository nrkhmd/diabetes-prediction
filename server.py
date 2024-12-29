from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import numpy as np

# Load the model
scaler, diabetes_model = pickle.load(open('diabetes_model_fixed.sav', 'rb'))

app = FastAPI()

@app.get('/', response_class=HTMLResponse)
async def read_root():
    return open('index.html').read()

@app.post('/predict/')
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
    # Scale the input values
    scaled_data = scaler.transform([
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
    ])
    
    # Perform prediction
    prediction = diabetes_model.predict(scaled_data)

    return {"prediction": "Positive" if prediction[0] == 1 else "Negative"}
