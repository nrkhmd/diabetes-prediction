import joblib

# Load model dari file asli
with open("diabetes_model_fixed.sav", "rb") as file:
    model = joblib.load(file)

# Simpan model dengan kompresi (level 3 adalah pilihan umum, bisa diatur 1-9)
joblib.dump(model, "diabetes_model_compressed.sav", compress=3)

print("Model berhasil dikompres menjadi 'diabetes_model_compressed.sav'")
