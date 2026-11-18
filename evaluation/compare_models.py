"""
Compact RF vs B's Optuna-tuned RF as the emission model, on the 10 STUDENT-TEST trips.

Inputs (written earlier):
  data/processed/probs_student_test_<model>.parquet      classifier probabilities
  results/hmm_vs_classifier_<model>.csv                  compare_hmm.py output per model
Output:
  results/hmm_model_comparison.csv  one row per (model, decoder): Top-1, continuity, route F1,
                                    mismatch, plus the classifier's row-level macro-F1 and log loss
Both models use the same candidates, features, beta = 5 m and gap = 60 s.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
MODELS = {"rf_compact": "RF compact (leaf 50, 100 trees, depth 18, sqrt)",
          "rf_tuned": "RF Optuna-tuned (leaf 18, 138 trees, depth 30, max_features 0.5)"}


def main():
    cand = pd.read_parquet(PROC / "candidates_student_test.parquet",
                           columns=["trip_id", "point_idx", "arc_id", "label", "label_reliable"])
    rows = []
    for name, desc in MODELS.items():
        p = pd.read_parquet(PROC / f"probs_student_test_{name}.parquet")
        c = cand.merge(p, on=["trip_id", "point_idx", "arc_id"])
        r = c[c.label_reliable == 1]
        clf_f1 = f1_score(r.label, r.prob >= 0.5, average="macro")
        clf_ll = log_loss(r.label, np.clip(r.prob, 1e-15, 1 - 1e-15))
        res = pd.read_csv(RES / f"hmm_vs_classifier_{name}.csv", dtype={"trip_id": str})
        res = res[res.trip_id.str.startswith("ALL")]
        for _, x in res.iterrows():
            rows.append({"model": name, "description": desc, "decoder": x.decoder,
                         "top1_all": x.top1_all, "top1_reliable": x.top1_reliable,
                         "route_continuity": x.route_continuity, "route_f1_len": x.route_f1_len,
                         "route_mismatch_frac": x.route_mismatch_frac,
                         "classifier_macro_f1": clf_f1, "classifier_log_loss": clf_ll,
                         "predict_s_per_trip": x.predict_s})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "hmm_model_comparison.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out.drop(columns="description").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
