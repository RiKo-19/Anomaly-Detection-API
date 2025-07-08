from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import joblib
import pandas as pd
import traceback

# Load model and preprocessing tools
try:
    model = joblib.load("isolation_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except Exception as e:
    raise RuntimeError("Error loading model or preprocessing files") from e

# Define FastAPI app
app = FastAPI()

# Enable CORS to connect with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data schema
class UserInput(BaseModel):
    financial_loss: Annotated[float, Field(..., description='Financial Loss for the cyber attack in Millions')]
    affected_users: Annotated[int, Field(..., description='Number of affected users')]
    resolution_time: Annotated[float, Field(..., description='Resolution time of the incident in hours')]
    country: Annotated[str, Field(..., description='The country where the cyber attack happened')]
    attack_type: Annotated[str, Field(..., description='Type of the cyber attack')]
    attack_source: Annotated[str, Field(..., description='Source of the cyber attack')]
    vulnerability_type: Annotated[str, Field(..., description='Security Vulnerability Type')]

# Risk scoring function
high_risk_countries = ["Russia", "China", "Iran", "North Korea"]

def compute_risk_score(data: UserInput) -> int:
    score = 0

    # Financial loss
    if data.financial_loss > 500:
        score += 30
    elif data.financial_loss > 100:
        score += 20
    else:
        score += 10

    # Affected users
    if data.affected_users > 100000:
        score += 20
    elif data.affected_users > 10000:
        score += 10

    # Resolution time
    if data.resolution_time > 72:
        score += 15
    elif data.resolution_time > 24:
        score += 10

    # Country
    if data.country in high_risk_countries:
        score += 15

    # Attack type
    if data.attack_type in ["Ransomware", "Zero-Day", "APT"]:
        score += 20
    elif data.attack_type in ["Phishing", "Malware"]:
        score += 10

    return min(score, 100)

@app.get('/')
def home():
    return {'message': 'Anomaly Detection API'}

@app.post("/predict")
def predict_threat(data: UserInput):
    try:
        df = pd.DataFrame([{
            "Financial Loss (in Million $)": data.financial_loss,
            "Number of Affected Users": data.affected_users,
            "Incident Resolution Time (in Hours)": data.resolution_time,
            "Country": data.country,
            "Attack Type": data.attack_type,
            "Attack Source": data.attack_source,
            "Security Vulnerability Type": data.vulnerability_type
        }])

        # Apply label encoding
        for col in ['Country', 'Attack Type', 'Attack Source', 'Security Vulnerability Type']:
            encoder = label_encoders[col]
            if df[col].iloc[0] in encoder.classes_:
                df[col] = encoder.transform(df[col])
            else:
                df[col] = -1  # unseen category

        # Predict with Isolation Forest
        X_array = scaler.transform(df[feature_cols])
        X = pd.DataFrame(X_array, columns=feature_cols)

        prediction = model.predict(X)[0]
        result = "Anomaly" if prediction == -1 else "Normal"

        return {"prediction": result}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Input error: {str(ve)}")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/score")
def risk_score(data: UserInput):
    try:
        score = compute_risk_score(data)
        level = "Low"
        if score > 70:
            level = "High"
        elif score > 40:
            level = "Medium"

        return {
            "risk_score": score,
            "risk_level": level
        }

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error computing risk score")
