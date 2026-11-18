# PathSnap Backend API

FastAPI service designed for the two-laptop distributed architecture.

## Deployment Setup (Laptop 1: Model & API)

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic
```

### 2. Run API Server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints
* `GET /health`: Health check and status verification.
* `GET /trips`: Returns the list of 10 student-test trips for evaluation.
* `POST /predict`: Accepts `{ "trip_id": "00000000", "use_viterbi": true }` and returns reconstructed road arcs, point match stats, and inference latency.
