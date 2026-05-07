import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Global State for Model & Scalers (Cached via Lifespan)
# ---------------------------------------------------------
ml_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On Startup: Load the heavy CNN model, the Data Scalers, and 
    calculate the Dataset Mean/Std to be kept in memory.
    """
    print("Booting AI Engine...")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    transformer_path = os.path.join(data_dir, "best_transformer_model.keras")
    cnn_path = os.path.join(data_dir, "best_cnn_model.keras")
    feature_scaler_path = os.path.join(data_dir, "feature_scaler.save")
    target_scaler_path = os.path.join(data_dir, "target_scaler.save")
    xtest_path = os.path.join(data_dir, "X_test.npy")

    if not all(os.path.exists(p) for p in [feature_scaler_path, target_scaler_path, xtest_path]):
        raise RuntimeError("Critical ML assets are missing. Please complete layers 1-6 first.")

    # Load Transformer (primary) or CNN (fallback)
    if os.path.exists(transformer_path):
        ml_components["model"] = load_model(transformer_path)
        ml_components["model_name"] = "Transformer"
    elif os.path.exists(cnn_path):
        ml_components["model"] = load_model(cnn_path)
        ml_components["model_name"] = "CNN"
    else:
        raise RuntimeError("No trained model found. Run Layer 5/5b first.")
    ml_components["feature_scaler"] = joblib.load(feature_scaler_path)
    ml_components["target_scaler"] = joblib.load(target_scaler_path)
    
    # Calculate baseline dataset prediction mean and std to evaluate confidence later
    X_test = np.load(xtest_path)
    background_preds = ml_components["model"].predict(X_test[:500], verbose=0)
    background_true = ml_components["target_scaler"].inverse_transform(background_preds).flatten()
    ml_components["mean_emission"] = float(np.mean(background_true))
    ml_components["std_emission"] = float(np.std(background_true))
    
    print(f"AI Engine Loaded and Ready! (Model: {ml_components['model_name']})")
    
    yield
    
    # On Shutdown: Clean up memory
    print("Shutting down AI Engine...")
    ml_components.clear()

# ---------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Carbon Emission Forecasting API",
    description="REST API serving the Carbon Emission Transformer/CNN model.",
    version="1.0.0",
    lifespan=lifespan
)

# Secure API against cross-origin browser scripts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Pydantic Schemas (Strict Input Validation)
# ---------------------------------------------------------
class EmissionRequest(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50.0, le=60.0)
    humidity: float = Field(..., description="Humidity percentage", ge=0.0, le=100.0)
    wind_speed: float = Field(..., description="Wind speed in km/h", ge=0.0)
    industry_level: int = Field(..., description="Industry categorical level", ge=1, le=5)
    vehicle_count: int = Field(..., description="Daily vehicle traffic", ge=0)
    energy_usage: float = Field(..., description="Total Energy used in MWh", ge=0.0)

class EmissionResponse(BaseModel):
    predicted_emission: float
    status_level: str
    confidence: str

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    """DevOps ping to verify the server and model are online."""
    return {"status": "online", "model_loaded": "model" in ml_components, "model_name": ml_components.get("model_name", "none")}

@app.post("/predict", response_model=EmissionResponse)
def predict_emission(request: EmissionRequest):
    """Core Inference Pipeline endpoint."""
    if "model" not in ml_components:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # 1. Transform JSON into Raw Feature Array
        raw_features = np.array([[
            request.temperature,
            request.humidity,
            request.wind_speed,
            request.industry_level,
            request.vehicle_count,
            request.energy_usage
        ]])

        # 2. Preprocess & Scale (Using pre-fit scaler)
        scaler_X = ml_components["feature_scaler"]
        scaled_features = scaler_X.transform(raw_features)

        # 3. Simulate Sequential Window
        # The model expects shape (samples, 7_time_steps, 6_features)
        time_steps = 7
        sequence = np.tile(scaled_features, (time_steps, 1))
        sequence = np.expand_dims(sequence, axis=0) # Shape: (1, 7, 6)
        
        # 4. Neural Network Inference
        model = ml_components["model"]
        pred_scaled = model.predict(sequence, verbose=0)
        
        # 5. Post-Process (Inverse Transform)
        scaler_y = ml_components["target_scaler"]
        pred_true = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
        
        # 6. Safety Status Logic
        if pred_true < 2000:
            status = "Safe"
        elif pred_true <= 6000:
            status = "Warning"
        else:
            status = "Danger"
            
        # 7. Confidence Logic (Anomaly Detection based on structural deviation)
        dev = abs(pred_true - ml_components["mean_emission"])
        if dev > (3 * ml_components["std_emission"]):
            confidence = "Low (Anomaly Detected / Out of Distribution)"
        elif dev > (2 * ml_components["std_emission"]):
            confidence = "Medium"
        else:
            confidence = "High"

        # 8. Return strictly typed JSON object
        return EmissionResponse(
            predicted_emission=round(pred_true, 2),
            status_level=status,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
