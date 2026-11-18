"""
Pre-flight check of a trip folder before the demo (run on Laptop 1).

For every trip in <PATHSNAP_DATA>/<partition>/ it checks that the .track/.nodes/.arcs files exist and
load, that the trip projects to a UTM zone and has candidate arcs, whether a .route file
(reference, used only server-side by evaluation/evaluate.py) is present, and that the full
predict_trip() pipeline runs. Nothing is tuned or changed.

Usage (from the repo root):
    python scripts/check_trips.py                   # partition from PATHSNAP_TRIPS_PARTITION or evaluator_test
    python scripts/check_trips.py student_test
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import DATA_DIR, load_trip_dir  # noqa: E402
from inference.predict import load_config, load_model, predict_trip  # noqa: E402


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PATHSNAP_TRIPS_PARTITION", "evaluator_test")
    folder = Path(DATA_DIR) / part
    print(f"data dir : {DATA_DIR}\npartition: {part}\nfolder   : {folder}")
    if not folder.is_dir():
        print(f"\nERROR: {folder} does not exist. Create it and copy the trip folders "
              f"(<id>/<id>.track, .nodes, .arcs, .route) into it.")
        sys.exit(1)
    trips = sorted(p.name for p in folder.iterdir() if p.is_dir())
    print(f"trips    : {len(trips)} {trips}\n")
    cfg = load_config()
    model = load_model(cfg)
    ok = 0
    for tid in trips:
        d = folder / tid
        missing = [e for e in ("track", "nodes", "arcs") if not (d / f"{tid}.{e}").exists()]
        has_route = (d / f"{tid}.route").exists() and (d / f"{tid}.route").stat().st_size > 0
        if missing:
            print(f"[FAIL] {tid}: missing {missing}")
            continue
        try:
            t0 = time.perf_counter()
            trip = load_trip_dir(d, load_route=False)
            pred = predict_trip(d, model=model, config=cfg, include_points=False)
            print(f"[ OK ] {tid}: {pred['n_points']:5d} fixes, EPSG {trip.epsg}, "
                  f"{pred['n_candidate_rows']:6d} candidate rows, {pred['n_skipped_points']} skipped, "
                  f"{pred['decoder']['n_segments']} segment(s), reference route: "
                  f"{'yes' if has_route else 'NO (accuracy unavailable)'}, {time.perf_counter() - t0:.1f}s")
            ok += 1
        except Exception as e:  # report and continue
            print(f"[FAIL] {tid}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(trips)} trips ready.")
    sys.exit(0 if ok == len(trips) and trips else 1)


if __name__ == "__main__":
    main()
