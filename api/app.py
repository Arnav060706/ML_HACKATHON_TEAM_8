"""
PathSnap Backend API (FastAPI)
Hosts the trained candidate scoring classifier, HMM/Viterbi route decoder,
and serves route reconstruction endpoints for the evaluation UI.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure PATHSNAP_DATA fallback
DEFAULT_DATA = ROOT.parent / "pathsnap" / "data"
if DEFAULT_DATA.exists() and "PATHSNAP_DATA" not in os.environ:
    os.environ["PATHSNAP_DATA"] = str(DEFAULT_DATA)

app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Distributed real-time map-matching service using ML scoring and Viterbi decoding.",
    version="1.0.0"
)

# Enable CORS for Laptop 2 Evaluator UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Student testing trips
EVAL_TRIPS = [
    "00000000", "00000007", "00000009", "00000040", "00000052", 
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

class PredictRequest(BaseModel):
    trip_id: str
    use_viterbi: bool = True

@app.get("/")
def root():
    return {
        "service": "PathSnap Map-Matching API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": ["/health", "/trips", "/predict"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": True
    }

@app.get("/trips")
def list_student_test_trips():
    """List available test trips for Laptop 2 evaluation UI."""
    return {
        "count": len(EVAL_TRIPS),
        "trips": EVAL_TRIPS
    }

@app.post("/predict")
def predict_trip_endpoint(req: PredictRequest):
    """
    Given a trip_id, score candidate road segments and reconstruct
    the most probable path using the Viterbi algorithm.
    """
    return run_prediction(req.trip_id, req.use_viterbi)

@app.get("/predict/{trip_id}")
def predict_trip_get(trip_id: str, use_viterbi: bool = Query(True)):
    """Convenience GET endpoint for easy browser/curl testing."""
    return run_prediction(trip_id, use_viterbi)

def run_prediction(trip_id: str, use_viterbi: bool = True):
    try:
        from inference.predict import predict_trip
        res = predict_trip(trip_id, include_points=True)
        return res
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed for trip {trip_id}: {str(e)}\n{tb}")

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
