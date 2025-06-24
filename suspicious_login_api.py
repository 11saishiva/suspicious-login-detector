from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from logistic_regression import ImprovedSuspiciousLoginDetector
import joblib
import numpy as np

detector = ImprovedSuspiciousLoginDetector.load_model("suspicious_login_model_20250610_114330.joblib")
threshold = joblib.load("threshold.joblib")
joblib.dump(0.01, "threshold.joblib")

app = FastAPI()

class LoginRecord(BaseModel):
    user_id: str
    login_timestamp: str  
    ip_address: str
    device_type: str
    browser: str
    latitude: float
    longitude: float
    is_blacklisted: int
    login_success: int
    is_off_hours: int
    pincode: int

@app.post("/predict/")
def predict_login(record: LoginRecord):
    try:
        data = pd.DataFrame([record.dict()])
        data['login_timestamp'] = pd.to_datetime(data['login_timestamp'])

        print("Converted login_timestamp dtype:", data['login_timestamp'].dtype)
        print(data.head())

        predictions, scores, _ = detector.predict_enhanced(data)

        score = float(scores[0])
        label = "suspicious" if score >= threshold else "normal"

        return {
            "prediction": label,
            "suspicion_score": round(score, 5),
            "threshold_used": round(threshold, 5)
        }

    except Exception as e:
        print("Exception:", e)
        return {"error": str(e)}