"""
Per-trip preprocessing report (train + student_test): what the loader cleaned,
which UTM zone each trip was projected into, time-gap statistics and a check
that the projection is sane (haversine vs UTM distance between consecutive fixes).

Output: results/preprocessing_report.csv  (+ summary printed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import PARTITIONS, iter_trips  # noqa: E402


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def main():
    rows = []
    for part in PARTITIONS:
        for trip in iter_trips(part):
            tr = trip.track
            dt = np.diff(tr.t.to_numpy())
            step_utm = np.hypot(np.diff(tr.x), np.diff(tr.y))
            step_hav = haversine_m(tr.lon[:-1].to_numpy(), tr.lat[:-1].to_numpy(),
                                   tr.lon[1:].to_numpy(), tr.lat[1:].to_numpy())
            rel_err = np.abs(step_utm - step_hav) / np.maximum(step_hav, 1)
            rows.append({
                "partition": part, "trip_id": trip.trip_id, "epsg": trip.epsg,
                "utm_zone": trip.epsg % 100,
                "hemisphere": "N" if trip.epsg < 32700 else "S",
                "n_raw_points": trip.meta["n_raw_points"],
                "n_clean_points": trip.meta["n_clean_points"],
                "n_dropped": trip.meta["n_raw_points"] - trip.meta["n_clean_points"],
                "n_out_of_order": trip.meta["n_out_of_order"],
                "duration_s": float(tr.t.iloc[-1] - tr.t.iloc[0]),
                "frac_dt_1s": float((dt == 1).mean()),
                "max_gap_s": float(dt.max()),
                "n_gaps_gt_10s": int((dt > 10).sum()),
                "track_length_km": float(step_utm.sum() / 1000),
                "n_nodes": len(trip.nodes_xy), "n_arcs": len(trip.arcs),
                "n_route_arcs": len(trip.route),
                "route_contiguous_frac": float(
                    (trip.arcs[trip.route[:-1], 1] == trip.arcs[trip.route[1:], 0]).mean())
                if len(trip.route) > 1 else np.nan,
                "utm_vs_haversine_max_rel_err": float(rel_err[step_hav > 5].max())
                if (step_hav > 5).any() else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "preprocessing_report.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.groupby("partition")[["n_raw_points", "n_dropped", "n_out_of_order",
                                       "frac_dt_1s", "max_gap_s", "n_gaps_gt_10s",
                                       "track_length_km", "route_contiguous_frac",
                                       "utm_vs_haversine_max_rel_err"]]
              .agg(["mean", "max"]).T.round(4))
        print("UTM zones used:", sorted(df.epsg.unique()))


if __name__ == "__main__":
    main()
