"""
Model training and evaluation for PathSnap (part B).

Candidate-level binary classification: each row is one (trip_id, point_idx, arc_id)
candidate and label = 1 if the arc is the reference arc. For every point the
top-scoring candidate is the predicted arc.

Protocol
  * GroupKFold(5) by trip_id on the 80 training trips, then refit on all 80 and
    evaluate on the 10 student-test trips. SEED = 42 everywhere.
  * Scalers live inside sklearn pipelines (fit on training folds only).
  * Fitting uses only label_reliable == 1 rows (see features/build_features.py);
    evaluation uses all rows and also reports Top-1 on reliable points.

Usage
  python models/run_experiments.py --fast        # 15 train trips, 2 folds, results/fast_run/
  python models/run_experiments.py --time-fold   # time fold 0 of each model, then exit
  python models/run_experiments.py               # full run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Same default raw-data location / env override as data_processing/load_data.py. That module
# imports pyproj at load time, which is not needed here, so only the arcs reader is mirrored.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent / "Student folder" / "Student folder"))


def _read(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=dtype).to_numpy()

# ----------------------------------------------------------------- schema constants
SEED = 42
THRESHOLD = 0.5
N_FOLDS = 5
PROC_DIR = ROOT / "data" / "processed"
TRAIN_FILE = "candidates_train.parquet"
TEST_FILE = "candidates_student_test.parquet"
TRAIN_POINTS_FILE = "points_train.parquet"
TEST_POINTS_FILE = "points_student_test.parquet"
RAW_ROOT = DATA_DIR  # <...>/Student folder/Student folder ; arcs at <root>/<partition>/<trip>/<trip>.arcs
KEY = ["trip_id", "point_idx"]
EXCLUDE = ["trip_id", "partition", "point_idx", "arc_id", "t", "label", "label_reliable"]
LABEL = "label"
RELIABLE = "label_reliable"
FIT_ON_RELIABLE_ONLY = True
FAST_TRIPS, FAST_FOLDS = 15, 2
TRAIN_EVAL_ROWS = 200_000  # rows used for the training-set macro F1 / log loss
LOGLOSS_EPS = 1e-15
OUT_DIR = ROOT / "results"
SAVE_DIR = ROOT / "models" / "saved"

# ----------------------------------------------------------------------------- models
KNN_MAX_FIT_ROWS = 100_000
MLP_MAX_FIT_ROWS = 400_000  # keeps one MLP fit under ~3 min (uncapped fold 0 took 288 s)


def build_models(rf_overrides=None, knn_overrides=None):
    rf_kw = dict(n_estimators=100, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    rf_kw.update(rf_overrides or {})
    knn_kw = dict(n_neighbors=25, n_jobs=-1)
    knn_kw.update(knn_overrides or {})
    knn_rows = (knn_overrides or {}).get("max_fit_rows", KNN_MAX_FIT_ROWS)
    knn_kw.pop("max_fit_rows", None)
    try:
        import lightgbm as lgb
        gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 random_state=SEED, n_jobs=-1, verbose=-1)
        gbm_name, gbm_cfg = "lightgbm", "LGBMClassifier"
    except ImportError:
        gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                             early_stopping=False, random_state=SEED)
        gbm_name, gbm_cfg = "hist_gradient_boosting", "HistGradientBoostingClassifier (lightgbm not installed)"
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200,
                        random_state=SEED)
    return {
        "logreg": dict(family="linear", key_config="StandardScaler + LogisticRegression",
                       pipe=Pipeline([("scale", StandardScaler()),
                                      ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
                       max_fit_rows=None),
        "knn": dict(family="instance-based",
                    key_config=f"StandardScaler + KNN k=25, fit on random subsample <= {knn_rows} rows",
                    pipe=Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(**knn_kw))]),
                    max_fit_rows=knn_rows),
        "rf": dict(family="bagged trees", key_config="RandomForest (no scaling needed)",
                   pipe=Pipeline([("clf", RandomForestClassifier(**rf_kw))]), max_fit_rows=None),
        gbm_name: dict(family="boosted trees", key_config=gbm_cfg + " (no scaling needed)",
                       pipe=Pipeline([("clf", gbm)]), max_fit_rows=None),
        "mlp": dict(family="neural network",
                    key_config=f"StandardScaler + MLP (64,32), early_stopping, max_iter=200, "
                                f"fit on random subsample <= {MLP_MAX_FIT_ROWS} rows (time cap)",
                    pipe=Pipeline([("scale", StandardScaler()), ("clf", mlp)]),
                    max_fit_rows=MLP_MAX_FIT_ROWS),
    }


# ------------------------------------------------------------------------------ data
def load_tables(fast: bool):
    tr = pd.read_parquet(PROC_DIR / TRAIN_FILE)
    te = pd.read_parquet(PROC_DIR / TEST_FILE)
    ptr = pd.read_parquet(PROC_DIR / TRAIN_POINTS_FILE, columns=KEY + ["partition", RELIABLE, "true_in_cands"])
    pte = pd.read_parquet(PROC_DIR / TEST_POINTS_FILE, columns=KEY + ["partition", RELIABLE, "true_in_cands"])
    if fast:
        trips = np.array(sorted(tr.trip_id.unique()))
        keep = np.random.default_rng(SEED).choice(trips, FAST_TRIPS, replace=False)
        tr, ptr = tr[tr.trip_id.isin(keep)], ptr[ptr.trip_id.isin(keep)]
    tr = tr.sort_values(KEY + ["dist_rank"], kind="stable").reset_index(drop=True)
    te = te.sort_values(KEY + ["dist_rank"], kind="stable").reset_index(drop=True)
    return tr, te, ptr, pte


@lru_cache(maxsize=None)
def load_arcs(trip_id: str, partition: str) -> np.ndarray:
    """(A, 2) from_node, to_node of the trip's own .arcs file (cached per trip)."""
    return _read(Path(RAW_ROOT) / partition / trip_id / f"{trip_id}.arcs", 2, np.int64)


# --------------------------------------------------------------------------- metrics
def top1_positions(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Row positions of the highest-scoring candidate per point; ties keep the lower dist_rank."""
    codes = pd.factorize(df.trip_id)[0]
    pidx = df.point_idx.to_numpy()
    order = np.lexsort((-score, pidx, codes))
    c, q = codes[order], pidx[order]
    first = np.r_[True, (c[1:] != c[:-1]) | (q[1:] != q[:-1])]
    return order[first]


def top1_arcs(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    return df.iloc[top1_positions(df, score)][KEY + ["partition", "arc_id", LABEL, RELIABLE]]


def route_continuity(top: pd.DataFrame) -> float:
    cont = tot = 0
    for (tid, part), g in top.groupby(["trip_id", "partition"], sort=False):
        g = g.sort_values("point_idx")
        a = g.arc_id.to_numpy()
        if len(a) < 2:
            continue
        arcs = load_arcs(tid, part)
        same = a[:-1] == a[1:]
        chain = arcs[a[:-1], 1] == arcs[a[1:], 0]
        cont += int((same | chain).sum())
        tot += len(a) - 1
    return cont / tot if tot else float("nan")


def binary_metrics(y, p, sfx=""):
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {f"accuracy{sfx}": accuracy_score(y, pred),
            f"precision{sfx}": precision_score(y, pred, zero_division=0),
            f"recall{sfx}": recall_score(y, pred, zero_division=0),
            f"f1{sfx}": f1_score(y, pred, zero_division=0),
            f"macro_f1{sfx}": f1_score(y, pred, average="macro", zero_division=0),
            f"log_loss{sfx}": log_loss(y, np.clip(p, LOGLOSS_EPS, 1 - LOGLOSS_EPS), labels=[0, 1]),
            f"tn{sfx}": tn, f"fp{sfx}": fp, f"fn{sfx}": fn, f"tp{sfx}": tp}


def row_level_metrics(df, p):
    """Row-level metrics on label_reliable==1 rows (unsuffixed) and on all rows (_all)."""
    y = df[LABEL].to_numpy()
    rel = df[RELIABLE].to_numpy() == 1
    m = binary_metrics(y[rel], p[rel])
    m.update(binary_metrics(y, p, "_all"))
    return m


def point_metrics(df, score, pts):
    """Top-1 / miss-rate / continuity. Denominators come from the points table."""
    n_pts = len(pts)
    n_rel = int(pts[RELIABLE].sum())
    n_true_in = int(pts.true_in_cands.sum())
    top = top1_arcs(df, score)
    ok = top[LABEL].to_numpy() == 1
    return {"top1_all_points": ok.sum() / n_pts,
            "top1_given_true_in_cands": ok.sum() / n_true_in if n_true_in else float("nan"),
            "top1_reliable_points": (ok & (top[RELIABLE].to_numpy() == 1)).sum() / n_rel if n_rel else float("nan"),
            "candidate_miss_rate": 1 - n_true_in / n_pts,
            "route_continuity": route_continuity(top)}


def evaluate(scorer, X, df, pts):
    t0 = time.perf_counter()
    p = scorer(X)
    pred_t = time.perf_counter() - t0
    m = row_level_metrics(df, p)
    m.update(point_metrics(df, p, pts))
    m["predict_time_s"] = pred_t
    return m


def train_metrics(scorer, X, y):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(X), min(TRAIN_EVAL_ROWS, len(X)), replace=False))
    m = binary_metrics(y[idx], scorer(X[idx]))
    return {"train_macro_f1": m["macro_f1"], "train_log_loss": m["log_loss"]}


def fit_model(spec, X, y):
    X_fit, y_fit = X, y
    cap = spec["max_fit_rows"]
    if cap and len(X) > cap:
        idx = np.sort(np.random.default_rng(SEED).choice(len(X), cap, replace=False))
        X_fit, y_fit = X[idx], y[idx]
    from sklearn.base import clone
    pipe = clone(spec["pipe"])
    t0 = time.perf_counter()
    pipe.fit(X_fit, y_fit)
    return pipe, time.perf_counter() - t0, len(X_fit)


def baseline_scores(df):
    """Rule baselines, no training. Higher score = better candidate.
    nearest_arc:         nearest candidate by distance (dist_rank == 0).
    nearest_arc_heading: among candidates whose heading differs from the GPS heading by <= 90 deg
                         (or any candidate when the heading is invalid) take the nearest one;
                         wrong-way candidates are only used if nothing else is left."""
    wrong_way = (df.heading_valid.to_numpy() == 1) & (df.heading_diff.to_numpy() > 90.0)
    return {"baseline_nearest_arc": -df.dist_rank.to_numpy().astype(float),
            "baseline_nearest_arc_heading": -(df.dist_m.to_numpy().astype(float) + 1e6 * wrong_way),
            "baseline_A_rule": -(df.dist_m.to_numpy().astype(float) + 0.5 * df.heading_diff.to_numpy().astype(float))}


BASELINES = {"baseline_nearest_arc": "predict candidate with dist_rank==0",
             "baseline_nearest_arc_heading":
                 "nearest candidate (dist_m) among those with heading_diff<=90 or heading invalid; "
                 "wrong-way candidates only if nothing else is left",
             "baseline_A_rule":
                 "part A's rule: lowest dist_m + 0.5*heading_diff per point (heading_diff=90 when heading invalid)"}


def baseline_row(df, pts, score):
    """Predict 1 for the rule's top candidate of each point. No model, no fit."""
    t0 = time.perf_counter()
    p = np.zeros(len(df))
    p[top1_positions(df, score)] = 1.0
    pred_t = time.perf_counter() - t0
    m = row_level_metrics(df, p)
    m.update(point_metrics(df, score, pts))
    m.update({"predict_time_s": pred_t, "fit_time_s": 0.0})
    rel = df[RELIABLE].to_numpy() == 1
    tm = binary_metrics(df[LABEL].to_numpy()[rel], p[rel])  # rule is untrained: same rule, same rows
    m.update({"train_macro_f1": tm["macro_f1"], "train_log_loss": tm["log_loss"]})
    return m


def merge_baselines(args, out_dir, tr, te, ptr, pte, folds):
    """Recompute baseline rows (no training) and merge them into the existing result files."""
    meta_cols = ["model", "family", "key_config"]
    first = ["model", "family", "key_config", "split"]
    cv_split = "cv_mean_train_fast" if args.fast else "cv_mean_train80"
    old_folds = pd.read_csv(out_dir / "cv_per_fold.csv")
    old_cmp = pd.read_csv(out_dir / "model_comparison.csv")
    old_folds = old_folds[~old_folds.model.isin(BASELINES)]
    old_cmp = old_cmp[~old_cmp.model.isin(BASELINES)]
    fold_rows, test_rows = [], []
    for name, cfg in BASELINES.items():
        for k, (_, iva) in enumerate(folds):
            dva = tr.iloc[iva]
            pva = ptr[ptr.trip_id.isin(dva.trip_id.unique())]
            m = baseline_row(dva, pva, baseline_scores(dva)[name])
            m.update(model=name, family="baseline (no model)", key_config=cfg, n_fit_rows=0,
                     split="cv_fold", fold=k)
            fold_rows.append(m)
        m = baseline_row(te, pte, baseline_scores(te)[name])
        m.update(model=name, family="baseline (no model)", key_config=cfg, n_fit_rows=0,
                 split="student_test10")
        test_rows.append(m)
    nf = pd.DataFrame(fold_rows)
    metric_cols = [c for c in nf.columns if c not in meta_cols + ["split", "fold"]]
    cvm = nf.groupby(meta_cols, sort=False)[metric_cols].mean().reset_index()
    cvm.insert(3, "split", cv_split)
    pd.concat([old_folds, nf], ignore_index=True).to_csv(out_dir / "cv_per_fold.csv", index=False)
    cmp_df = pd.concat([old_cmp, cvm, pd.DataFrame(test_rows)], ignore_index=True)
    cmp_df = cmp_df[first + [c for c in cmp_df.columns if c not in first]]
    cmp_df.to_csv(out_dir / "model_comparison.csv", index=False)
    cfg_path = out_dir / "configs.json"
    if cfg_path.exists():
        cj = json.load(open(cfg_path))
        cj["baselines"] = BASELINES
        cj["baselines_note"] = "untrained rules (no fit), not one of the five models"
        json.dump(cj, open(cfg_path, "w"), indent=2, default=str)
    print("Merged baselines:", list(BASELINES), "into", out_dir)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help=f"{FAST_TRIPS} train trips, {FAST_FOLDS} folds")
    ap.add_argument("--time-fold", action="store_true", help="time fold 0 per model and exit")
    ap.add_argument("--baselines-only", action="store_true",
                    help="recompute only the untrained baselines and merge them into existing results files")
    ap.add_argument("--rf-trees", type=int, default=None)
    ap.add_argument("--rf-depth", type=int, default=None)
    ap.add_argument("--knn-rows", type=int, default=None)
    args = ap.parse_args()

    out_dir = OUT_DIR / "fast_run" if args.fast else OUT_DIR
    save_dir = ROOT / "models" / ("saved_fast" if args.fast else "saved")
    n_folds = FAST_FOLDS if args.fast else N_FOLDS
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    tr, te, ptr, pte = load_tables(args.fast)
    features = [c for c in tr.columns if c not in EXCLUDE]
    print(f"Features ({len(features)}): {features}")
    print(f"train: {tr.trip_id.nunique()} trips, {len(tr):,} rows | "
          f"student_test: {te.trip_id.nunique()} trips, {len(te):,} rows")

    rf_ov = {k: v for k, v in (("n_estimators", args.rf_trees), ("max_depth", args.rf_depth)) if v}
    knn_ov = {"max_fit_rows": args.knn_rows} if args.knn_rows else None
    models = build_models(rf_ov, knn_ov)
    print("GBM family uses:", [v["key_config"] for k, v in models.items() if v["family"] == "boosted trees"][0])

    X_tr = tr[features].to_numpy(np.float32)
    y_tr = tr[LABEL].to_numpy()
    rel_tr = tr[RELIABLE].to_numpy() == 1
    X_te = te[features].to_numpy(np.float32)

    removed = {"train_rows_total": int(len(tr)), "train_rows_removed": int((~rel_tr).sum()),
               "train_rows_removed_frac": float((~rel_tr).mean()),
               "train_points_total": int(len(ptr)), "train_points_removed": int((ptr[RELIABLE] == 0).sum()),
               "train_points_removed_frac": float((ptr[RELIABLE] == 0).mean()),
               "test_rows_unreliable": int((te[RELIABLE] == 0).sum()),
               "test_points_unreliable": int((pte[RELIABLE] == 0).sum()),
               "test_points_total": int(len(pte))}
    print("Removed from training (label_reliable==0):", json.dumps(removed))
    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(X_tr, y_tr, groups=tr.trip_id))
    if args.time_fold:
        folds = folds[:1]

    def fit_idx(idx):
        return idx[rel_tr[idx]] if FIT_ON_RELIABLE_ONLY else idx

    if args.baselines_only:
        merge_baselines(args, out_dir, tr, te, ptr, pte, folds)
        return

    fold_rows, test_rows, saved = [], [], {}
    all_idx = fit_idx(np.arange(len(tr)))
    meta_cols = ["model", "family", "key_config"]
    first = ["model", "family", "key_config", "split"]

    def write_partial():
        """Write cv_per_fold.csv and model_comparison.csv from everything finished so far."""
        folds_df = pd.DataFrame(fold_rows)
        folds_df.to_csv(out_dir / "cv_per_fold.csv", index=False)
        metric_cols = [c for c in folds_df.columns if c not in meta_cols + ["split", "fold"]]
        cv_mean = folds_df.groupby(meta_cols, sort=False)[metric_cols].mean().reset_index()
        cv_mean.insert(3, "split", "cv_mean_train_fast" if args.fast else "cv_mean_train80")
        cmp_df = pd.concat([cv_mean, pd.DataFrame(test_rows)], ignore_index=True)
        cmp_df = cmp_df[first + [c for c in cmp_df.columns if c not in first]]
        cmp_df.to_csv(out_dir / "model_comparison.csv", index=False)

    for name, spec in list(models.items()) + [(b, None) for b in BASELINES]:
        # ---- cross-validation folds
        for k, (itr, iva) in enumerate(folds):
            t_start = time.perf_counter()
            dva = tr.iloc[iva]
            pva = ptr[ptr.trip_id.isin(dva.trip_id.unique())]
            if spec is None:
                m = baseline_row(dva, pva, baseline_scores(dva)[name])
                m.update(model=name, family="baseline (no model)",
                         key_config=BASELINES[name], n_fit_rows=0)
            else:
                ifit = fit_idx(itr)
                pipe, fit_t, n_fit = fit_model(spec, X_tr[ifit], y_tr[ifit])
                sc = lambda A, p=pipe: p.predict_proba(A)[:, 1]
                m = evaluate(sc, X_tr[iva], dva, pva)
                m.update(train_metrics(sc, X_tr[ifit], y_tr[ifit]))
                m.update(model=name, family=spec["family"], key_config=spec["key_config"],
                         fit_time_s=fit_t, n_fit_rows=n_fit)
            m.update(split="cv_fold", fold=k)
            fold_rows.append(m)
            print(f"[cv] {name:28s} fold {k}: macroF1={m['macro_f1']:.4f} top1={m['top1_all_points']:.4f} "
                  f"fit={m['fit_time_s']:.1f}s pred={m['predict_time_s']:.1f}s "
                  f"(fold wall {time.perf_counter() - t_start:.1f}s)", flush=True)
        if args.time_fold:
            continue
        # ---- refit on all training trips, evaluate on the student-test trips
        if spec is None:
            m = baseline_row(te, pte, baseline_scores(te)[name])
            m.update(model=name, family="baseline (no model)", key_config=BASELINES[name], n_fit_rows=0)
        else:
            pipe, fit_t, n_fit = fit_model(spec, X_tr[all_idx], y_tr[all_idx])
            sc = lambda A, p=pipe: p.predict_proba(A)[:, 1]
            m = evaluate(sc, X_te, te, pte)
            m.update(train_metrics(sc, X_tr[all_idx], y_tr[all_idx]))
            m.update(model=name, family=spec["family"], key_config=spec["key_config"],
                     fit_time_s=fit_t, n_fit_rows=n_fit)
            joblib.dump(pipe, save_dir / f"{name}.joblib")
            saved[name] = pipe
        m["split"] = "student_test10"
        test_rows.append(m)
        print(f"[test] {name:28s}: macroF1={m['macro_f1']:.4f} top1_all={m['top1_all_points']:.4f} "
              f"top1_rel={m['top1_reliable_points']:.4f} fit={m['fit_time_s']:.1f}s", flush=True)
        write_partial()
    if args.time_fold:
        return

    # ---- MLP loss curve (real loss_curve_ only)
    mlp_clf = saved["mlp"].named_steps["clf"]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(mlp_clf.loss_curve_) + 1), mlp_clf.loss_curve_)
    plt.xlabel("epoch")
    plt.ylabel("training loss (loss_curve_)")
    plt.title(f"MLP (64,32) training loss, refit on train trips ({mlp_clf.n_iter_} epochs)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mlp_loss_curve.png", dpi=150)
    plt.close()

    # ---- configs
    configs = {"seed": SEED, "threshold": THRESHOLD, "fast_mode": args.fast,
               "features": features, "excluded_columns": EXCLUDE,
               "cv_scheme": f"GroupKFold(n_splits={n_folds}) by trip_id, no shuffle; metrics averaged over folds "
                            "(confusion counts are fold means)",
               "refit": "all training trips -> evaluated on 10 student-test trips",
               "fit_on_label_reliable_only": FIT_ON_RELIABLE_ONLY,
               "eval_uses_all_rows": True,
               "train_metrics_rows": f"random subsample of <= {TRAIN_EVAL_ROWS} fitted rows (seed {SEED})",
               "data_files": [TRAIN_FILE, TEST_FILE, TRAIN_POINTS_FILE, TEST_POINTS_FILE],
               "n_train_trips": int(tr.trip_id.nunique()), "n_test_trips": int(te.trip_id.nunique()),
               "models": {n: {"family": s["family"], "key_config": s["key_config"],
                              "max_fit_rows": s["max_fit_rows"],
                              "params": {k: (v if isinstance(v, (int, float, str, bool, type(None), tuple)) else str(v))
                                         for k, v in s["pipe"].named_steps["clf"].get_params().items()}}
                         for n, s in models.items()},
               "baselines": BASELINES, "baselines_note": "untrained rules, not one of the five models",
               "row_level_metrics": "unsuffixed = label_reliable==1 rows; *_all = all rows",
               "top1_denominators": "points_train / points_student_test parquet (points without a true "
                                    "candidate count as misses); top1_reliable_points uses label_reliable==1 points",
               "removed_from_training": removed}
    with open(out_dir / "configs.json", "w") as fh:
        json.dump(configs, fh, indent=2, default=str)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
