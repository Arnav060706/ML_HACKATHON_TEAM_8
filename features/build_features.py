"""
Feature engineering + export of the candidate tables.

For every trip in a partition:
  1. load + clean + project to the trip's UTM zone        (data_processing/load_data.py)
  2. candidates: arcs within R m of each point, top-k     (features/candidates.py)
  3. labels: monotone alignment of points to the route    (features/labels.py)
  4. features for each (point, candidate arc) pair        (this file)

Outputs (in data/processed/):
  candidates_<partition>.parquet   one row per (point, candidate arc) -> model input
  points_<partition>.parquet       one row per GPS point (denominator for Top-1 /
                                   miss rate, true arc, alignment distance)

Feature list (all computed from GPS + road network only; the reference route is
used ONLY to create the `label` column, never as a feature):

  dist_m            perpendicular distance point -> arc (m)
  dist_rank         rank of this arc among the point's candidates (0 = nearest)
  dist_minus_min    dist_m - nearest candidate's distance (relative closeness)
  n_cands           size of the point's candidate set (ambiguity of the area)
  proj_t            where the projection falls along the arc (0 = start, 1 = end)
  at_endpoint       projection clipped to an arc end (point is "beyond" the arc)
  heading_diff      |GPS heading - arc bearing| in [0, 180] deg; separates the two
                    directions of a two-way street and crossing streets
  heading_valid     1 if the vehicle moved >= MIN_MOVE m (heading reliable), else
                    heading_diff is set to a neutral 90
  speed_mps         centred speed estimate (m/s); low speed => noisy heading
  seg_len_m         arc length (m)
  dt_prev, dt_next  time gap to the previous / next fix (s) -- GPS gaps
  dist_prev_pt,     distance of the PREVIOUS / NEXT GPS fix to the same arc:
  dist_next_pt      temporal context; the true arc tends to stay close to neighbours
  from_in_deg,      in-degree of the arc's start node / out-degree of its end node
  to_out_deg        (junction complexity)
  twoway            1 if the reverse arc v->u exists
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import iter_trips  # noqa: E402
from features.candidates import generate_candidates  # noqa: E402
from features.geometry import angle_diff_deg, bearing_deg, point_segment  # noqa: E402
from features.labels import assign_route_arcs  # noqa: E402

RADIUS = 50.0
TOP_K = 12
MIN_MOVE = 2.0  # metres of displacement needed for a reliable GPS heading

FEATURES = ["dist_m", "dist_rank", "dist_minus_min", "n_cands", "proj_t",
            "at_endpoint", "heading_diff", "heading_valid", "speed_mps",
            "seg_len_m", "dt_prev", "dt_next", "dist_prev_pt", "dist_next_pt",
            "from_in_deg", "to_out_deg", "twoway"]


def point_kinematics(track: pd.DataFrame):
    """GPS heading (deg), heading validity, speed (m/s) and time gaps per point."""
    x, y, t = (track[c].to_numpy() for c in ("x", "y", "t"))
    n = len(x)
    ip = np.r_[0, np.arange(n - 1)]          # previous index (self at start)
    inx = np.r_[np.arange(1, n), n - 1]      # next index (self at end)
    dx, dy = x[inx] - x[ip], y[inx] - y[ip]
    disp = np.hypot(dx, dy)
    dt_span = np.maximum(t[inx] - t[ip], 1e-9)
    heading = bearing_deg(dx, dy)
    valid = disp >= MIN_MOVE
    speed = np.where(t[inx] > t[ip], disp / dt_span, 0.0)
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    return heading, valid, speed, dt_prev, dt_next


def arc_attributes(trip):
    """Per-arc static attributes over the whole network."""
    arcs = trip.arcs
    nn = len(trip.nodes_xy)
    a, b = trip.arc_xy
    seg_len = np.linalg.norm(b - a, axis=1)
    bearing = bearing_deg(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    in_deg = np.bincount(arcs[:, 1], minlength=nn)
    out_deg = np.bincount(arcs[:, 0], minlength=nn)
    key = arcs[:, 0].astype(np.int64) * nn + arcs[:, 1]
    rkey = arcs[:, 1].astype(np.int64) * nn + arcs[:, 0]
    twoway = np.isin(rkey, key)
    return {"seg_len": seg_len, "bearing": bearing,
            "from_in_deg": in_deg[arcs[:, 0]], "to_out_deg": out_deg[arcs[:, 1]],
            "twoway": twoway}


def trip_tables(trip, radius=RADIUS, k=TOP_K):
    tr = trip.track
    n = len(tr)
    cand = generate_candidates(trip, radius=radius, k=k)
    has_route = len(trip.route) > 0
    if has_route:
        true_arc, route_pos, ldist = assign_route_arcs(trip)
    else:
        true_arc, route_pos, ldist = np.full(n, -1), np.full(n, -1), np.full(n, np.nan)

    heading, hvalid, speed, dt_prev, dt_next = point_kinematics(tr)
    att = arc_attributes(trip)

    pi = cand.point_idx.to_numpy()
    ai = cand.arc_id.to_numpy()
    d = cand.dist_m.to_numpy()
    g = cand.groupby("point_idx", sort=False)

    f = pd.DataFrame({"trip_id": trip.trip_id, "point_idx": pi, "arc_id": ai})
    f["t"] = tr.t.to_numpy()[pi]
    f["dist_m"] = d
    f["dist_rank"] = g.cumcount().to_numpy()
    f["dist_minus_min"] = d - g.dist_m.transform("min").to_numpy()
    f["n_cands"] = g.point_idx.transform("size").to_numpy()
    f["proj_t"] = cand.proj_t.to_numpy()
    f["at_endpoint"] = ((f.proj_t <= 0.0) | (f.proj_t >= 1.0)).astype(np.int8)
    hd = angle_diff_deg(heading[pi], att["bearing"][ai])
    f["heading_diff"] = np.where(hvalid[pi], hd, 90.0)
    f["heading_valid"] = hvalid[pi].astype(np.int8)
    f["speed_mps"] = speed[pi]
    f["seg_len_m"] = att["seg_len"][ai]
    f["dt_prev"] = dt_prev[pi]
    f["dt_next"] = dt_next[pi]

    # distance of neighbouring fixes to the same arc (temporal context)
    pxy = tr[["x", "y"]].to_numpy()
    a_all, b_all = trip.arc_xy
    prev_i = np.maximum(pi - 1, 0)
    next_i = np.minimum(pi + 1, n - 1)
    f["dist_prev_pt"], _ = point_segment(pxy[prev_i], a_all[ai], b_all[ai])
    f["dist_next_pt"], _ = point_segment(pxy[next_i], a_all[ai], b_all[ai])

    f["from_in_deg"] = att["from_in_deg"][ai]
    f["to_out_deg"] = att["to_out_deg"][ai]
    f["twoway"] = att["twoway"][ai].astype(np.int8)
    f["label"] = (ai == true_arc[pi]).astype(np.int8) if has_route else -1

    # per-point table
    n_c = np.bincount(pi, minlength=n)
    in_c = np.zeros(n, dtype=bool)
    if has_route:
        in_c[pi[f.label.to_numpy() == 1]] = True
    pts = pd.DataFrame({"trip_id": trip.trip_id, "point_idx": np.arange(n),
                        "t": tr.t.to_numpy(), "lon": tr.lon.to_numpy(), "lat": tr.lat.to_numpy(),
                        "x": pxy[:, 0], "y": pxy[:, 1], "epsg": trip.epsg,
                        "true_arc": true_arc, "route_pos": route_pos,
                        "label_dist_m": ldist, "n_cands": n_c,
                        "true_in_cands": in_c.astype(np.int8)})
    return f, pts


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in ("t", "x", "y", "lon", "lat"):
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == np.int64 and c not in ("arc_id", "true_arc"):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def build_partition(partition: str, radius=RADIUS, k=TOP_K, out_dir: Path | None = None):
    out_dir = Path(out_dir or ROOT / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats, pts = [], []
    for trip in iter_trips(partition):
        f, p = trip_tables(trip, radius, k)
        f.insert(1, "partition", partition)
        p.insert(1, "partition", partition)
        feats.append(f)
        pts.append(p)
        print(f"[{partition}] {trip.trip_id}: {len(p)} pts, {len(f)} cand rows "
              f"({time.time() - t0:.0f}s)", flush=True)
    F = _compact(pd.concat(feats, ignore_index=True))
    P = _compact(pd.concat(pts, ignore_index=True))
    F.to_parquet(out_dir / f"candidates_{partition}.parquet", index=False)
    P.to_parquet(out_dir / f"points_{partition}.parquet", index=False)
    summary = {"partition": partition, "radius_m": radius, "top_k": k,
               "n_trips": int(P.trip_id.nunique()), "n_points": len(P),
               "n_candidate_rows": len(F), "positive_rate": float((F.label == 1).mean()),
               "candidate_miss_rate": float(1 - P.true_in_cands.mean()),
               "points_without_candidates": int((P.n_cands == 0).sum()),
               "mean_candidates_per_point": float(P.n_cands.mean()),
               "features": FEATURES, "build_seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=2))
    return F, P, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()
    summaries = [build_partition(p, args.radius, args.k)[2] for p in args.partitions]
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "feature_build_summary.json", "w") as fh:
        json.dump(summaries, fh, indent=2)
