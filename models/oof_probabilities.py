"""
Candidate probabilities for the HMM experiments.

* train trips: OUT-OF-FOLD probabilities from GroupKFold(5) by trip_id. A model that
  scores the very rows it was fitted on is over-confident, and tuning the HMM's beta
  on such probabilities would favour trusting the emissions too much. OOF
  probabilities behave like probabilities on unseen trips.
* student_test trips: probabilities from the model refitted on all 80 train trips
  (the same model that is saved and used by the API).

The model is B's Random Forest configuration (models/run_experiments.py), optionally
overridden by a JSON of tuned parameters (e.g. B's Optuna best parameters).

Outputs
  data/processed/probs_train_oof.parquet      trip_id, point_idx, arc_id, prob
  data/processed/probs_student_test.parquet   trip_id, point_idx, arc_id, prob
  models/saved/<name>.joblib                  final model refitted on all train trips

Usage
  python models/oof_probabilities.py                        # B's untuned RF
  python models/oof_probabilities.py --params results/rf_best_params.json --name rf_tuned
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.build_features import FEATURES  # noqa: E402
from models.run_experiments import SEED, build_models  # noqa: E402

PROC = ROOT / "data" / "processed"
SAVE = ROOT / "models" / "saved"
KEY = ["trip_id", "point_idx", "arc_id"]


def make_model(params: dict | None):
    pipe = clone(build_models()["rf"]["pipe"])
    if params:
        pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    pipe.set_params(clf__random_state=SEED)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="JSON file with RandomForest parameters")
    ap.add_argument("--name", default="rf")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    params = json.load(open(args.params)) if args.params else None

    cols = KEY + FEATURES + ["label", "label_reliable"]
    tr = pd.read_parquet(PROC / "candidates_train.parquet", columns=cols)
    te = pd.read_parquet(PROC / "candidates_student_test.parquet", columns=cols)
    X = tr[FEATURES].to_numpy(np.float32)
    y = tr.label.to_numpy()
    rel = tr.label_reliable.to_numpy() == 1
    oof = np.full(len(tr), np.nan)

    t0 = time.time()
    for k, (itr, iva) in enumerate(GroupKFold(args.folds).split(X, y, groups=tr.trip_id)):
        fit = itr[rel[itr]]                       # fit on reliable rows only (as in B's harness)
        m = make_model(params).fit(X[fit], y[fit])
        oof[iva] = m.predict_proba(X[iva])[:, 1]
        del m
        print(f"fold {k}: fitted on {len(fit):,} rows ({time.time() - t0:.0f}s)", flush=True)

    final = make_model(params).fit(X[rel], y[rel])
    SAVE.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, SAVE / f"{args.name}.joblib", compress=3)
    p_te = final.predict_proba(te[FEATURES].to_numpy(np.float32))[:, 1]
    print(f"final model saved ({time.time() - t0:.0f}s)", flush=True)

    tr.assign(prob=oof)[KEY + ["prob"]].to_parquet(PROC / f"probs_train_oof_{args.name}.parquet", index=False)
    te.assign(prob=p_te)[KEY + ["prob"]].to_parquet(PROC / f"probs_student_test_{args.name}.parquet", index=False)
    meta = {"model": args.name, "params": params or "B's untuned RF (models/run_experiments.py)",
            "oof_folds": args.folds, "fit_rows": "label_reliable == 1", "seed": SEED,
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(PROC / f"probs_{args.name}_meta.json", "w"), indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
