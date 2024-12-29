from fastapi import FastAPI
import joblib
import os
import numpy as np
from pydantic import BaseModel

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Muat model dari file terkompresi
model_path = os.path.join(os.path.dirname(__file__), "diabetes_model_compressed.sav")
with open(model_path, "rb") as model_file:
    model = joblib.load(model_file)

# Definisikan schema data input
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

# Endpoint default untuk test
@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API"}

# Endpoint untuk prediksi
@app.post("/predict")
def predict(input_data: DiabetesInput):
    data_array = np.array([
        input_data.pregnancies,
        input_data.glucose,
        input_data.blood_pressure,
        input_data.skin_thickness,
        input_data.insulin,
        input_data.bmi,
        input_data.diabetes_pedigree,
        input_data.age
    ]).reshape(1, -1)

    prediction = model.predict(data_array)
    result = "Positive" if prediction[0] == 1 else "Negative"
    return {"prediction": result}
