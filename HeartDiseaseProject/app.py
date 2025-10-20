from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

# إنشاء التطبيق
app = FastAPI(title="Heart Disease Prediction API")

# ربط ملفات القوالب
templates = Jinja2Templates(directory="templates")

# تحميل الموديل والسكايلر
from tensorflow.keras.models import load_model

model = load_model(r"D:\Moataz\ITI\HeartDiseaseProject\heart_disease_fcnn.keras")

scaler = joblib.load("D:\Moataz\ITI\HeartDiseaseProject\scaler.pkl")

# صفحة الواجهة
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# نقطة التنبؤ
@app.post("/predict")
def predict(data: dict):
    values = np.array([list(data.values())]).astype(float)
    scaled = scaler.transform(values)
    prediction = model.predict(scaled)
    result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    return {"prediction": result}
