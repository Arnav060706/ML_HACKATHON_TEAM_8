"""
predict_trip(): the complete prediction pipeline for ONE trip.

    raw .track/.nodes/.arcs  ->  per-trip UTM projection  ->  candidates (R, k)
    ->  17 features  ->  Random Forest probabilities  ->  HMM / Viterbi  ->  route

This module NEVER reads the .route file (load_trip_dir(..., load_route=False)); the
reference route is only read by evaluation/evaluate.py. predict_trip() is what the
FastAPI server calls.

Configuration (model file, beta, gap threshold, R, k) lives in inference/config.json.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import DATA_DIR, find_trip_dir, load_trip_dir  # noqa: E402
from features.build_features import FEATURES, trip_tables  # noqa: E402
from inference.hmm import (build_lattice, classifier_only, reconstruct_route,  # noqa: E402
                           state_arcs, viterbi)
from inference.road_graph import RoadGraph  # noqa: E402

CONFIG_PATH = ROOT / "inference" / "config.json"
log = logging.getLogger("pathsnap.predict")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_model(config: dict | None = None):
    config = config or load_config()
    path = ROOT / config["model_path"]
    if not path.exists() and config.get("fallback_model_path"):
        log.warning("model %s not found, falling back to %s", path, config["fallback_model_path"])
        path = ROOT / config["fallback_model_path"]
    return joblib.load(path)


def predict_trip(trip: str | Path, model=None, config: dict | None = None,
                 data_dir: Path = DATA_DIR, include_points: bool = True) -> dict:
    """Predict the road arcs for every GPS fix of a trip.

    trip  : trip id (searched under data_dir/<any partition>/) or a trip folder path.
    Returns a JSON-serialisable dict with per-fix HMM and classifier-only arcs, the
    reconstructed route (arc ids + lat/lon polyline), skipped fixes and timings.
    """
    config = config or load_config()
    model = model if model is not None else load_model(config)
    t0 = time.perf_counter()
    trip_dir = Path(trip) if Path(str(trip)).is_dir() else find_trip_dir(str(trip), data_dir)
    tr = load_trip_dir(trip_dir, load_route=False)          # .route is never opened here
    assert len(tr.route) == 0
    t_load = time.perf_counter()

    feats, _ = trip_tables(tr, radius=config["radius_m"], k=config["top_k"])
    t_feat = time.perf_counter()
    prob = model.predict_proba(feats[FEATURES].to_numpy(np.float32))[:, 1] if len(feats) else np.empty(0)
    t_model = time.perf_counter()

    graph = RoadGraph(tr)
    lat = build_lattice(graph, tr.track, feats[["point_idx", "arc_id", "proj_t"]], prob)
    hmm_states, seg_id, vinfo = viterbi(lat, config["beta"], config["gap_s"])
    hmm_arcs = state_arcs(lat, hmm_states)
    clf_arcs = state_arcs(lat, classifier_only(lat))
    route, rinfo = reconstruct_route(graph, lat, hmm_arcs, seg_id)
    t_hmm = time.perf_counter()
    clf_route, _ = reconstruct_route(graph, lat, clf_arcs, seg_id)  # same reconstruction, for comparison

    if len(lat.skipped):
        log.info("trip %s: %d fixes without candidates skipped: %s", tr.trip_id, len(lat.skipped),
                 lat.skipped[:20].tolist())

    n = len(tr.track)
    pred_hmm = np.full(n, -1, dtype=np.int64)
    pred_clf = np.full(n, -1, dtype=np.int64)
    pred_hmm[lat.point_idx] = hmm_arcs
    pred_clf[lat.point_idx] = clf_arcs
    seg_full = np.full(n, -1, dtype=np.int64)
    seg_full[lat.point_idx] = seg_id

    out = {
        "trip_id": tr.trip_id,
        "partition": tr.partition,
        "epsg": tr.epsg,
        "n_points": n,
        "n_skipped_points": int(len(lat.skipped)),
        "skipped_points": lat.skipped.tolist(),
        "n_candidate_rows": int(len(feats)),
        "pred_arc_hmm": pred_hmm.tolist(),
        "pred_arc_classifier": pred_clf.tolist(),
        "segment_id": seg_full.tolist(),
        "route_arcs": route,
        "route_arcs_classifier": clf_route,
        "route_latlon": route_polyline(tr, route),
        "decoder": {"beta": config["beta"], "gap_s": config["gap_s"], **vinfo, **rinfo},
        "model_path": config["model_path"],
        "timings_s": {"load": t_load - t0, "features": t_feat - t_load, "model": t_model - t_feat,
                      "hmm": t_hmm - t_model, "total": t_hmm - t0,
                      "shortest_paths": lat.timings["shortest_paths_s"]},
    }
    if include_points:
        out["gps_latlon"] = tr.track[["lat", "lon"]].to_numpy().round(7).tolist()
        out["timestamps"] = tr.track["t"].tolist()
    return out


def route_polyline(tr, route: list[int]) -> list:
    """Route as a list of polylines ([[lat, lon], ...]); a new polyline starts at each break (-1)."""
    lines, cur = [], []
    for a in route:
        if a < 0:
            if cur:
                lines.append(cur)
            cur = []
            continue
        u, v = tr.arcs[a]
        pu = [float(tr.nodes_ll[u][1]), float(tr.nodes_ll[u][0])]
        pv = [float(tr.nodes_ll[v][1]), float(tr.nodes_ll[v][0])]
        if not cur or cur[-1] != pu:
            cur.append(pu)
        cur.append(pv)
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    trip_id = sys.argv[1] if len(sys.argv) > 1 else "00000000"
    res = predict_trip(trip_id, include_points=False)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("pred_arc_hmm", "pred_arc_classifier", "segment_id", "route_arcs",
                                   "route_latlon", "skipped_points")}, indent=2))
