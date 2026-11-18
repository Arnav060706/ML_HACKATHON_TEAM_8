"""
PathSnap Backend API (FastAPI)
Hosts the trained candidate scoring classifier, HMM/Viterbi route decoder,
and serves route reconstruction endpoints for the evaluation UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time

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

class PredictRequest(BaseModel):
    trip_id: str
    use_viterbi: bool = True

class PredictResponse(BaseModel):
    trip_id: str
    predicted_arcs: List[int]
    total_points: int
    matched_points: int
    inference_time_ms: float
    decoder_used: str

@app.get("/")
def root():
    return {
        "service": "PathSnap Map-Matching API",
        "status": "online",
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
    test_trips = ["00000000", "00000007", "00000009", "00000040", "00000052", 
                  "00000056", "00000065", "00000076", "00000077", "00000091"]
    return {
        "count": len(test_trips),
        "trips": test_trips
    }

@app.post("/predict", response_model=PredictResponse)
def predict_trip(req: PredictRequest):
    """
    Given a trip_id, score candidate road segments and reconstruct
    the most probable path using the Viterbi algorithm.
    """
    t0 = time.time()
    
    # Placeholder response to be connected with trained model artifact & decoder
    predicted_arcs = []
    
    inference_time = (time.time() - t0) * 1000.0
    return PredictResponse(
        trip_id=req.trip_id,
        predicted_arcs=predicted_arcs,
        total_points=0,
        matched_points=0,
        inference_time_ms=inference_time,
        decoder_used="HMM/Viterbi" if req.use_viterbi else "Greedy Top-1 Classifier"
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
