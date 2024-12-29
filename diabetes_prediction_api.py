from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import logging
import numpy as np

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load the model
scaler, diabetes_model = pickle.load(open('diabetes_model_fixed.sav', 'rb'))


app = FastAPI()

# Home route with form and prediction in one page
@app.get('/', response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diabetes Prediction</title>
        <style>
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f9;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}
.container {
    background: #ffffff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    text-align: center;
    width: 100%;
    max-width: 400px;
}
h1 {
    color: #333;
    font-size: 1.5em;
    margin-bottom: 20px;
}
form {
    display: flex;
    flex-direction: column;
}
input {
    margin-bottom: 15px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 1em;
}
button {
    background-color: #4CAF50;
    color: white;
    padding: 10px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1em;
}
button:hover {
    background-color: #45a049;
}
#result {
    margin-top: 20px;
    font-size: 1.2em;
    color: #333;
}
</style>
    </head>
    <body>
        <div class="container">
            <h1>Implementasi Model Single Layer Perceptron untuk Klasifikasi Status Gejala Diabetes</h1>
            <form id="prediction-form">
                <input type="number" name="Pregnancies" placeholder="Jumlah Kehamilan (Pregnancies)" step="any" required>
                <input type="number" name="Glucose" placeholder="Glukosa (Glucose)" step="any" required>
                <input type="number" name="BloodPressure" placeholder="Tekanan Darah (Blood Pressure)" step="any" required>
                <input type="number" name="SkinThickness" placeholder="Ketebalan Kulit (Skin Thickness)" step="any" required>
                <input type="number" name="Insulin" placeholder="Insulin" step="any" required>
                <input type="number" name="BMI" placeholder="Indeks Masa Tubuh (BMI)" step="any" required>
                <input type="number" name="DiabetesPedigreeFunction" placeholder="Diabetes Pedigree Function" step="any" required>
                <input type="number" name="Age" placeholder="Usia (Age)" step="any" required>
                <button type="submit">Prediksi</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById('prediction-form').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (result.error) {
                    document.getElementById('result').innerHTML = `<h2>Error: ${result.error}</h2>`;
                } else {
                    document.getElementById('result').innerHTML = `<h2>Prediction: ${result.prediction}</h2>`;
                }
            };
        </script>
    </body>
    </html>
    """

# Prediction route
@app.post('/predict')
async def predict(
    Pregnancies: float = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: float = Form(...),
):
    try:
        # Prepare the input for the model
        input_data = np.array([[
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]])

        # Log input data
        logging.info(f"Input data: {input_data}")

        # Make prediction
        prediction = diabetes_model.predict(input_data)

        # Log hasil prediksi
        logging.info(f"Model prediction: {prediction}")

        # Logika prediksi
        if prediction[0] == 1:
            result = "Positif Diabetes"  # Jika prediksi 1
        elif prediction[0] == 0:
            result = "Negatif Diabetes"  # Jika prediksi 0
        else:
            result = "Prediksi tidak valid"  # Jika model mengembalikan nilai selain 0 atau 1

        return {"prediction": result}

    except Exception as e:
        # Tangani error jika terjadi masalah
        logging.error(f"Error during prediction: {e}")
        return {"error": str(e)}
