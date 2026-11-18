"""
PathSnap Backend API (FastAPI) -- runs on Laptop 1.

    uvicorn api.app:app --host 0.0.0.0 --port 8000        (run from the repo root)

Endpoints
    GET  /health              model / data status
    GET  /trips               trip ids found in the evaluation folder (scanned, never hard-coded)
    POST /predict             {"trip_id": "...", "use_viterbi": true}
    GET  /predict/{trip_id}   same, for browser / curl testing

Pipeline per request
    predict_trip()  (inference/predict.py)  raw files -> candidates -> features -> RF -> HMM/Viterbi
                                            never reads the reference .route file
    evaluate()      (evaluation/evaluate.py) server-side only: compares the prediction with the
                                            reference route and returns ONLY metrics (no labels)

Configuration (environment variables)
    PATHSNAP_DATA             folder containing the partition folders (default: see load_data.py)
    PATHSNAP_TRIPS_PARTITION  partition folder to serve (default "student_test";
                              set to e.g. "evaluator_test" for the final evaluation)
    PATHSNAP_WARMUP           "0" to skip the warm-up run at startup
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# optional fallback data location used on one team laptop (must be set before importing load_data)
_ALT_DATA = ROOT.parent / "pathsnap" / "data"
if _ALT_DATA.exists() and "PATHSNAP_DATA" not in os.environ:
    os.environ["PATHSNAP_DATA"] = str(_ALT_DATA)

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from data_processing.load_data import DATA_DIR, list_trips  # noqa: E402
from evaluation.evaluate import evaluate  # noqa: E402
from inference.predict import load_config, load_model, predict_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

PARTITION = os.environ.get("PATHSNAP_TRIPS_PARTITION", "student_test")
STATE: dict = {"model": None, "config": None, "warmup": None}

# fields of predict_trip() that the UI does not need (kept out of the response to keep it small)
_DROP = ("pred_arc_classifier", "route_arcs_classifier", "route_latlon_classifier", "skipped_points",
         "segment_id", "timestamps")


def trip_folder() -> Path:
    return Path(DATA_DIR) / PARTITION


def available_trips() -> list[str]:
    folder = trip_folder()
    if not folder.is_dir():
        return []
    return list_trips(PARTITION, DATA_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.perf_counter()
    STATE["config"] = load_config()
    STATE["model"] = load_model(STATE["config"])
    log.info("model loaded once at startup: %s (%.1fs)", STATE["config"]["model_path"],
             time.perf_counter() - t0)
    log.info("data dir: %s | serving partition '%s' (%d trips) | beta=%s gap_s=%s", DATA_DIR, PARTITION,
             len(available_trips()), STATE["config"]["beta"], STATE["config"]["gap_s"])
    if os.environ.get("PATHSNAP_WARMUP", "1") != "0":
        tw = time.perf_counter()
        ok = 0
        for tid in available_trips():
            try:
                predict_trip(trip_folder() / tid, model=STATE["model"], config=STATE["config"],
                             include_points=False)
                ok += 1
            except Exception as e:  # warm-up must never stop the server
                log.warning("warm-up failed for trip %s: %s", tid, e)
        STATE["warmup"] = {"trips_ok": ok, "seconds": round(time.perf_counter() - tw, 2)}
        log.info("warm-up: predicted %d trips in %.1fs", ok, time.perf_counter() - tw)
    log.info("API ready")
    yield


app = FastAPI(title="PathSnap Map-Matching & Route Reconstruction API",
              description="Random Forest candidate scoring + HMM/Viterbi route reconstruction.",
              version="1.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path, client,
             response.status_code, 1000 * (time.perf_counter() - t0))
    return response


class PredictRequest(BaseModel):
    trip_id: str
    use_viterbi: bool = True


@app.get("/")
def root():
    return {"service": "PathSnap Map-Matching API", "status": "online", "version": "1.1.0",
            "endpoints": ["/health", "/trips", "/predict", "/predict/{trip_id}"]}


@app.get("/health")
def health_check():
    cfg = STATE["config"] or {}
    return {"status": "healthy" if STATE["model"] is not None else "starting",
            "model_loaded": STATE["model"] is not None,
            "model_path": cfg.get("model_path"),
            "decoder": {"beta": cfg.get("beta"), "gap_s": cfg.get("gap_s")},
            "data_dir": str(DATA_DIR), "partition": PARTITION,
            "n_trips": len(available_trips()), "warmup": STATE["warmup"], "timestamp": time.time()}


@app.get("/trips")
def list_eval_trips():
    """Trip ids found by scanning <PATHSNAP_DATA>/<PATHSNAP_TRIPS_PARTITION>/."""
    trips = available_trips()
    return {"partition": PARTITION, "count": len(trips), "trips": trips}


@app.post("/predict")
def predict_trip_endpoint(req: PredictRequest):
    return run_prediction(req.trip_id, req.use_viterbi)


@app.get("/predict/{trip_id}")
def predict_trip_get(trip_id: str, use_viterbi: bool = Query(True)):
    return run_prediction(trip_id, use_viterbi)


def run_prediction(trip_id: str, use_viterbi: bool = True) -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if trip_id not in available_trips():
        raise HTTPException(status_code=404, detail=f"trip {trip_id} not found in partition '{PARTITION}'")
    tdir = trip_folder() / trip_id
    try:
        t0 = time.perf_counter()
        pred = predict_trip(tdir, model=STATE["model"], config=STATE["config"], include_points=True)
        t_pred = time.perf_counter() - t0
        # server-side evaluation against the reference route; only metrics leave this function
        scores = evaluate(pred, trip_dir=tdir)
    except Exception as e:
        log.exception("prediction failed for trip %s", trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed for trip {trip_id}: {e}")

    decoder = "hmm" if use_viterbi else "classifier_only"
    m = scores[decoder]
    other = scores["classifier_only" if use_viterbi else "hmm"]
    if not use_viterbi:  # show the classifier-only route instead of the HMM route
        pred["route_latlon"] = pred["route_latlon_classifier"]
        pred["route_arcs"] = pred["route_arcs_classifier"]
        pred["matched_latlon"] = None
    resp = {k: v for k, v in pred.items() if k not in _DROP}
    resp["decoder_used"] = "HMM/Viterbi" if use_viterbi else "Classifier only (RF argmax)"
    if m is None:  # trip has no reference route: prediction only, no accuracy
        resp["metrics"] = {"decoder": decoder, "reference_available": False, "top1_accuracy": None,
                           "top1_accuracy_reliable": None, "route_continuity": None, "route_f1": None,
                           "route_precision": None, "route_recall": None, "route_mismatch": None,
                           "n_points": pred["n_points"], "comparison": None}
        log.info("trip %s | %s | points=%d skipped=%d | no reference route: metrics unavailable | "
                 "predict %.2fs", trip_id, decoder, pred["n_points"], pred["n_skipped_points"], t_pred)
        return resp
    resp["metrics"] = {
        "decoder": decoder,
        "reference_available": True,
        "top1_accuracy": m["top1_all"],
        "top1_accuracy_reliable": m["top1_reliable"],
        "route_continuity": m["route_continuity"],
        "route_f1": m["route_f1_len"],
        "route_precision": m["route_precision_len"],
        "route_recall": m["route_recall_len"],
        "route_mismatch": m["route_mismatch_frac"],
        "n_points": m["n_points"],
        "comparison": {"hmm": scores["hmm"], "classifier_only": scores["classifier_only"]},
    }
    log.info("trip %s | %s | points=%d skipped=%d | Top-1=%.4f continuity=%.4f routeF1=%.4f "
             "(other decoder Top-1=%.4f) | predict %.2fs", trip_id, decoder, pred["n_points"],
             pred["n_skipped_points"], m["top1_all"], m["route_continuity"] or 0, m["route_f1_len"] or 0,
             other["top1_all"], t_pred)
    return resp


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000)
