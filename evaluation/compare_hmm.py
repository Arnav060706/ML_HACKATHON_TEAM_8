"""
Classifier-only vs classifier + HMM on the 10 STUDENT-TEST trips (default) or another partition
given as the first argument (e.g. `python evaluation/compare_hmm.py evaluator_test`, for final
reporting only -- never used to choose anything), through the exact
pipeline the API uses: predict_trip() (no .route) followed by evaluate() (.route).

Outputs
  results/hmm_vs_classifier.csv          per trip and overall (pooled over fixes), both decoders
  results/hmm_failure_cases.csv          windows where classifier-only was right and the HMM wrong
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import DATA_DIR, list_trips  # noqa: E402
from evaluation.evaluate import evaluate, reference_labels  # noqa: E402
from inference.predict import load_config, load_model, predict_trip  # noqa: E402

PART = "student_test"


def failure_windows(trip_id, pred, true_arc, min_len=5):
    """Runs of >= min_len consecutive fixes where the classifier was right and the HMM wrong."""
    hmm = np.array(pred["pred_arc_hmm"])
    clf = np.array(pred["pred_arc_classifier"])
    bad = (clf == true_arc) & (hmm != true_arc)
    rows, k = [], 0
    while k < len(bad):
        if bad[k]:
            j = k
            while j < len(bad) and bad[j]:
                j += 1
            if j - k >= min_len:
                rows.append({"trip_id": trip_id, "first_point": k, "last_point": j - 1, "n_points": j - k,
                             "t_start": pred["timestamps"][k], "t_end": pred["timestamps"][j - 1],
                             "segment_id": pred["segment_id"][k],
                             "hmm_arcs": sorted(set(hmm[k:j].tolist())),
                             "true_arcs": sorted(set(true_arc[k:j].tolist()))})
            k = j
        else:
            k += 1
    return rows


def main():
    global PART
    if len(sys.argv) > 1:
        PART = sys.argv[1]          # e.g. "evaluator_test" -- final reporting only, never for tuning
    sfx = "" if PART == "student_test" else f"_{PART}"
    cfg = load_config()
    model = load_model(cfg)
    rows, fails = [], []
    for tid in list_trips(PART):
        t0 = time.perf_counter()
        pred = predict_trip(tid, model=model, config=cfg)
        res = evaluate(pred, trip_dir=DATA_DIR / PART / tid)
        if not res.get("reference_available", True):
            print(tid, "no reference route: skipped in metrics", flush=True)
            continue
        _, true_arc, _ = reference_labels(DATA_DIR / PART / tid)
        fails += failure_windows(tid, pred, true_arc)
        for dec in ("classifier_only", "hmm"):
            rows.append({"trip_id": tid, "decoder": dec, **res[dec],
                         "predict_s": pred["timings_s"]["total"], "n_skipped": pred["n_skipped_points"],
                         "n_segments": pred["decoder"]["n_segments"]})
        print(tid, {d: round(res[d]["top1_all"], 4) for d in ("classifier_only", "hmm")},
              f"{time.perf_counter() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    w = df.n_points
    overall = []
    for dec, g in df.groupby("decoder", sort=False):
        gw = g.n_points
        overall.append({"trip_id": "ALL (pooled over fixes)", "decoder": dec, "n_points": int(gw.sum()),
                        "top1_all": float((g.top1_all * gw).sum() / gw.sum()),
                        "top1_reliable": float((g.top1_reliable * gw).sum() / gw.sum()),
                        "route_continuity": float((g.route_continuity * gw).sum() / gw.sum()),
                        "route_precision_len": g.route_precision_len.mean(),
                        "route_recall_len": g.route_recall_len.mean(),
                        "route_f1_len": g.route_f1_len.mean(),
                        "route_mismatch_frac": g.route_mismatch_frac.mean(),
                        "predict_s": g.predict_s.mean(), "n_skipped": int(g.n_skipped.sum()),
                        "n_segments": int(g.n_segments.sum())})
    out = pd.concat([df, pd.DataFrame(overall)], ignore_index=True)
    out.insert(2, "beta", cfg["beta"])
    out.insert(3, "gap_s", cfg["gap_s"])
    out.insert(4, "model", cfg["model_path"])
    out.insert(1, "partition", PART)
    out.to_csv(ROOT / "results" / f"hmm_vs_classifier{sfx}.csv", index=False)
    pd.DataFrame(fails).to_csv(ROOT / "results" / f"hmm_failure_cases{sfx}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out[out.trip_id.str.startswith("ALL")].T)
    print(f"{len(fails)} failure windows written")
    _ = w, json


if __name__ == "__main__":
    main()
