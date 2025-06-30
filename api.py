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
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
