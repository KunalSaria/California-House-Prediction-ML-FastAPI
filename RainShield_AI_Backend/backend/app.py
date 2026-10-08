from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title="RainShield AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    state: Literal["Assam", "Delhi"]
    horizon: Literal[1, 3, 6] = Field(default=6)

# Demo values are aligned with the SIH prototype/PPT.
# Replace this function with the trained XGBoost model when the model file is available.
DEMO = {
    "Assam": {
        1: {"rainfall_mm": 28, "flood_probability": 42, "peak_mm_h": 28},
        3: {"rainfall_mm": 71, "flood_probability": 67, "peak_mm_h": 31},
        6: {"rainfall_mm": 134, "flood_probability": 82, "peak_mm_h": 34},
    },
    "Delhi": {
        1: {"rainfall_mm": 18, "flood_probability": 24, "peak_mm_h": 18},
        3: {"rainfall_mm": 43, "flood_probability": 48, "peak_mm_h": 24},
        6: {"rainfall_mm": 78, "flood_probability": 63, "peak_mm_h": 29},
    },
}

WHY = {
    "Assam": [
        ("6h cumulative rainfall", "High impact"),
        ("River / drainage condition", "High impact"),
        ("Soil moisture", "Medium"),
        ("Low elevation", "Medium"),
    ],
    "Delhi": [
        ("3–6h cumulative rainfall", "High impact"),
        ("Urban drainage stress", "High impact"),
        ("Low-lying pockets", "Medium"),
        ("Soil moisture", "Medium"),
    ],
}

def risk_level(p):
    if p >= 75:
        return "EXTREME"
    if p >= 55:
        return "HIGH"
    if p >= 30:
        return "MODERATE"
    return "LOW"

@app.get("/")
def root():
    return {"service": "RainShield AI", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictionRequest):
    d = DEMO[req.state][req.horizon]
    p = d["flood_probability"]
    risk = risk_level(p)

    return {
        "state": req.state,
        "location": "Guwahati, Assam" if req.state == "Assam" else "Delhi",
        "horizon_hours": req.horizon,
        "rainfall_mm": d["rainfall_mm"],
        "peak_intensity_mm_h": d["peak_mm_h"],
        "flood_probability": p,
        "risk_level": risk,
        "alert_priority": "HIGH" if p >= 55 else "WATCH",
        "explainability": [
            {"feature": x, "impact": y} for x, y in WHY[req.state]
        ],
        "model": {
            "rainfall": "XGBoost Regression (prototype/demo)",
            "flood": "XGBoost Classification (prototype/demo)"
        }
    }
