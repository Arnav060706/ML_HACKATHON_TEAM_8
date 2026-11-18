"""
evaluate(): scores a prediction from inference/predict.py against the reference route.

This is the ONLY module of the prediction/evaluation pipeline that reads a .route file.
It is kept separate from predict_trip() so reference labels can never reach the model,
the decoder or the UI: the API calls predict_trip() first and evaluate() afterwards,
and returns only the resulting metrics.

Metrics (per trip)
  top1_all            fraction of ALL GPS fixes whose predicted arc equals the reference arc
                      (reference arc per fix = monotone DP alignment, features/labels.py);
                      skipped fixes (no candidate) count as wrong
  top1_reliable       same, over fixes within 100 m of their aligned reference arc
  route_continuity    fraction of consecutive predicted fixes whose arcs are equal or
                      directly connected (to(a) == from(b)), same definition as round 1
  route_precision_len, route_recall_len, route_f1_len
                      whole-sequence measure: length-weighted overlap between the set of
                      arcs in the reconstructed predicted route and in the reference route
  route_mismatch_frac Newson & Krumm route mismatch: (len(pred \\ ref) + len(ref \\ pred)) / len(ref)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import DATA_DIR, find_trip_dir, load_trip_dir  # noqa: E402
from features.labels import assign_route_arcs  # noqa: E402

RELIABLE_M = 100.0


def read_route(trip_dir: Path) -> np.ndarray:
    f = Path(trip_dir) / f"{Path(trip_dir).name}.route"
    if not f.exists() or f.stat().st_size == 0:
        return np.empty(0, dtype=np.int64)
    return pd.read_csv(f, sep=r"\s+", header=None, dtype=np.int64).to_numpy().ravel()


def reference_labels(trip_dir: Path):
    """(trip with route, reference arc per fix, reliable mask)."""
    trip = load_trip_dir(trip_dir, load_route=False)
    trip.route = read_route(trip_dir)
    true_arc, _, ldist = assign_route_arcs(trip)
    return trip, true_arc, ldist <= RELIABLE_M


def continuity(arcs_seq: np.ndarray, arc_from: np.ndarray, arc_to: np.ndarray) -> float:
    a = arcs_seq[arcs_seq >= 0]
    if len(a) < 2:
        return float("nan")
    ok = (a[:-1] == a[1:]) | (arc_to[a[:-1]] == arc_from[a[1:]])
    return float(ok.mean())


def route_overlap(pred_route: list[int], ref_route: np.ndarray, arc_len: np.ndarray) -> dict:
    pred = {a for a in pred_route if a >= 0}
    ref = set(int(a) for a in ref_route)
    lp = arc_len[list(pred)].sum() if pred else 0.0
    lr = arc_len[list(ref)].sum() if ref else 0.0
    li = arc_len[list(pred & ref)].sum() if pred & ref else 0.0
    prec = li / lp if lp else 0.0
    rec = li / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mismatch = ((lp - li) + (lr - li)) / lr if lr else float("nan")
    return {"route_precision_len": prec, "route_recall_len": rec, "route_f1_len": f1,
            "route_mismatch_frac": mismatch}


def score(pred_arcs: np.ndarray, route_arcs: list[int], true_arc, reliable, trip) -> dict:
    pred_arcs = np.asarray(pred_arcs)
    a, b = trip.arc_xy
    arc_len = np.linalg.norm(b - a, axis=1)
    hit = pred_arcs == true_arc
    out = {"n_points": int(len(pred_arcs)),
           "top1_all": float(hit.mean()),
           "top1_reliable": float(hit[reliable].mean()) if reliable.any() else float("nan"),
           "route_continuity": continuity(pred_arcs, trip.arcs[:, 0], trip.arcs[:, 1])}
    out.update(route_overlap(route_arcs, trip.route, arc_len))
    return out


def evaluate(prediction: dict, data_dir: Path = DATA_DIR, trip_dir: Path | None = None) -> dict:
    """Metrics for a predict_trip() result, for both the HMM and the classifier-only output."""
    trip_dir = Path(trip_dir) if trip_dir else find_trip_dir(prediction["trip_id"], data_dir)
    trip, true_arc, reliable = reference_labels(trip_dir)
    hmm = score(np.array(prediction["pred_arc_hmm"]), prediction["route_arcs"], true_arc, reliable, trip)
    clf_route = prediction.get("route_arcs_classifier")
    clf = score(np.array(prediction["pred_arc_classifier"]), clf_route if clf_route is not None else [],
                true_arc, reliable, trip)
    if clf_route is None:
        for k in ("route_precision_len", "route_recall_len", "route_f1_len", "route_mismatch_frac"):
            clf[k] = float("nan")
    # plain floats, NaN -> None, so the result is valid JSON for the API
    clean = lambda d: {k: (None if isinstance(v, float) and np.isnan(v) else  # noqa: E731
                           (float(v) if isinstance(v, (float, np.floating)) else v)) for k, v in d.items()}
    return {"trip_id": prediction["trip_id"], "hmm": clean(hmm), "classifier_only": clean(clf)}
