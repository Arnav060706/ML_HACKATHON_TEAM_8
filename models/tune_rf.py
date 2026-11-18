"""
Bayesian (Optuna TPE) tuning of the random forest for PathSnap (part B).

Search : 3-fold GroupKFold by trip_id on the 80 training trips, objective = macro F1 on
         label_reliable==1 validation rows (same definition as run_experiments.py).
         Each trial fits on a random FIT_FRAC sample of the label_reliable==1 training-fold
         rows and is scored on the full validation fold. Student-test data is never used here.
Final  : refit the best params on all 80 trips (reliable rows) -> models/saved/rf_tuned.joblib
         (compress=3, git-ignored), then compare untuned vs tuned RF with the same 5-fold
         GroupKFold as run_experiments.py and on the 10 student-test trips.

Usage
  python models/tune_rf.py                     # full run, timeout 1800 s
  python models/tune_rf.py --timeout 60 --fast # tiny smoke test, writes to results/tune_test/
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
import optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as rx  # noqa: E402

SEED = rx.SEED
SEARCH_FOLDS = 3
FIT_FRAC = 0.30
TIMEOUT_S = 1800


def est_size_mb(rf) -> float:
    """Rough in-memory size: ~64 B per node struct + 16 B per node value, summed over trees."""
    return sum(t.tree_.node_count for t in rf.estimators_) * 80 / 1e6


def fit_rf(params, X, y):
    spec = rx.build_models(rf_overrides=params)["rf"]
    return rx.fit_model(spec, X, y)  # (pipe, fit_time, n_rows)


def top1_all_points(df, score, pts):
    top = rx.top1_arcs(df, score)
    return float((top[rx.LABEL].to_numpy() == 1).sum() / len(pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    ap.add_argument("--fast", action="store_true", help="15-trip subset, 2 comparison folds, writes to results/tune_test/")
    args = ap.parse_args()

    out_dir = rx.OUT_DIR / "tune_test" if args.fast else rx.OUT_DIR
    save_dir = rx.ROOT / "models" / ("saved_fast" if args.fast else "saved")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    tr, te, ptr, pte = rx.load_tables(args.fast)
    features = [c for c in tr.columns if c not in rx.EXCLUDE]
    X_tr = tr[features].to_numpy(np.float32)
    y_tr = tr[rx.LABEL].to_numpy()
    rel = tr[rx.RELIABLE].to_numpy() == 1
    X_te = te[features].to_numpy(np.float32)
    print(f"train {tr.trip_id.nunique()} trips {len(tr):,} rows | test {te.trip_id.nunique()} trips", flush=True)

    search_folds = list(GroupKFold(n_splits=SEARCH_FOLDS).split(X_tr, y_tr, groups=tr.trip_id))
    rng = np.random.default_rng(SEED)

    def objective(trial: optuna.Trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 150),
            max_depth=trial.suggest_int("max_depth", 8, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        )
        f1s, lls, t1s, fit_ts, sizes = [], [], [], [], []
        for k, (itr, iva) in enumerate(search_folds):
            ifit = itr[rel[itr]]
            sub = np.sort(rng.choice(ifit, int(len(ifit) * FIT_FRAC), replace=False))
            pipe, fit_t, _ = fit_rf(params, X_tr[sub], y_tr[sub])
            dva = tr.iloc[iva]
            pva = ptr[ptr.trip_id.isin(dva.trip_id.unique())]
            p = pipe.predict_proba(X_tr[iva])[:, 1]
            m = rx.row_level_metrics(dva, p)
            f1s.append(m["macro_f1"]); lls.append(m["log_loss"])
            t1s.append(top1_all_points(dva, p, pva))
            fit_ts.append(fit_t); sizes.append(est_size_mb(pipe.named_steps["clf"]))
            trial.report(float(np.mean(f1s)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_all_points", float(np.mean(t1s)))
        trial.set_user_attr("fit_time_s", float(np.mean(fit_ts)))
        trial.set_user_attr("est_model_size_mb", float(np.mean(sizes)))
        print(f"[trial {trial.number}] f1={np.mean(f1s):.4f} ll={np.mean(lls):.4f} top1={np.mean(t1s):.4f} "
              f"fit={np.mean(fit_ts):.1f}s ~{np.mean(sizes):.0f}MB {params}", flush=True)
        return float(np.mean(f1s))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    print(f"Search: TPE seed {SEED}, {SEARCH_FOLDS}-fold, fit on {FIT_FRAC:.0%} of reliable rows, "
          f"timeout {args.timeout}s", flush=True)
    t0 = time.perf_counter()
    study.optimize(objective, timeout=args.timeout)
    search_t = time.perf_counter() - t0

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Search finished in {search_t:.0f}s: {len(done)} completed, {len(pruned)} pruned trials", flush=True)

    # ---- trial log
    rows = []
    for t in study.trials:
        r = {"trial": t.number, "state": t.state.name, **t.params, "cv_macro_f1": t.value,
             **{k: t.user_attrs.get(k) for k in ("cv_log_loss", "cv_top1_all_points", "fit_time_s", "est_model_size_mb")}}
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "optuna_trials.csv", index=False)

    best = study.best_trial
    best_params = dict(best.params)
    json.dump({"best_params": best_params, "best_cv_macro_f1_3fold_30pct": best.value,
               "best_cv_log_loss_3fold_30pct": best.user_attrs["cv_log_loss"],
               "best_cv_top1_all_points_3fold_30pct": best.user_attrs["cv_top1_all_points"],
               "completed_trials": len(done), "pruned_trials": len(pruned), "total_trials": len(study.trials),
               "search_seconds": search_t, "seed": SEED, "search_folds": SEARCH_FOLDS, "fit_fraction": FIT_FRAC},
              open(out_dir / "best_params.json", "w"), indent=2)

    # ---- plots
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    plot_optimization_history(study); plt.tight_layout()
    plt.savefig(out_dir / "optuna_optimization_history.png", dpi=150); plt.close("all")
    try:
        plot_param_importances(study); plt.tight_layout()
        plt.savefig(out_dir / "optuna_param_importance.png", dpi=150); plt.close("all")
    except Exception as e:  # too few completed trials
        print("param importance plot skipped:", e, flush=True)

    # ---- refit best on all training trips, save
    all_idx = np.where(rel)[0]
    tuned_pipe, tuned_fit_t, _ = fit_rf(best_params, X_tr[all_idx], y_tr[all_idx])
    model_path = save_dir / "rf_tuned.joblib"
    joblib.dump(tuned_pipe, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6
    print(f"Saved {model_path} ({size_mb:.1f} MB on disk, compress=3)", flush=True)

    # ---- untuned vs tuned, original 5-fold setup (same folds/seed as run_experiments.py)
    n_folds = rx.FAST_FOLDS if args.fast else rx.N_FOLDS
    folds = list(GroupKFold(n_splits=n_folds).split(X_tr, y_tr, groups=tr.trip_id))
    setups = {"untuned_rf": {}, "tuned_rf": best_params}
    cv_setup = f"{n_folds}-fold GroupKFold (run_experiments setup), fit on all reliable training-fold rows"
    out = []
    tuned_pipe_test = tuned_pipe
    for name, params in setups.items():
        fm = []
        for k, (itr, iva) in enumerate(folds):
            ifit = itr[rel[itr]]
            pipe, _, _ = fit_rf(params, X_tr[ifit], y_tr[ifit])
            dva = tr.iloc[iva]
            pva = ptr[ptr.trip_id.isin(dva.trip_id.unique())]
            m = rx.evaluate(lambda A, p=pipe: p.predict_proba(A)[:, 1], X_tr[iva], dva, pva)
            fm.append(m)
            print(f"[compare cv] {name} fold {k}: f1={m['macro_f1']:.4f} ll={m['log_loss']:.4f}", flush=True)
        cv = pd.DataFrame(fm).mean()
        pipe = tuned_pipe_test if name == "tuned_rf" else fit_rf(params, X_tr[all_idx], y_tr[all_idx])[0]
        mt = rx.evaluate(lambda A, p=pipe: p.predict_proba(A)[:, 1], X_te, te, pte)
        out.append({"model": name, "params": json.dumps(params) if params else "default (100, depth 18, leaf 5, sqrt)",
                    "cv_setup": cv_setup,
                    "cv_macro_f1": cv["macro_f1"], "cv_log_loss": cv["log_loss"], "cv_top1_all_points": cv["top1_all_points"],
                    "test_setup": "refit on all 80 train trips (reliable rows) -> 10 student-test trips (not used for tuning)",
                    "student_test_top1_all_points": mt["top1_all_points"], "student_test_macro_f1": mt["macro_f1"],
                    "student_test_log_loss": mt["log_loss"], "student_test_predict_time_s": mt["predict_time_s"]})
    cmp_df = pd.DataFrame(out)
    cmp_df.to_csv(out_dir / "tuning_comparison.csv", index=False)
    print(cmp_df.drop(columns=["params", "cv_setup", "test_setup"]).to_string(), flush=True)

    base_ll = cmp_df.loc[cmp_df.model == "untuned_rf", "cv_log_loss"].iloc[0]
    tuned_ll = cmp_df.loc[cmp_df.model == "tuned_rf", "cv_log_loss"].iloc[0]
    print(f"Best trial 3-fold/30% CV log loss: {best.user_attrs['cv_log_loss']:.4f}", flush=True)
    print(f"5-fold CV log loss: untuned {base_ll:.4f}, tuned {tuned_ll:.4f}", flush=True)
    if tuned_ll > base_ll or best.user_attrs["cv_log_loss"] > 0.1081:
        print("WARNING: tuned CV log loss is worse than the untuned RF "
              f"(best trial {best.user_attrs['cv_log_loss']:.4f}, tuned 5-fold {tuned_ll:.4f}, untuned 0.1081 / {base_ll:.4f})",
              flush=True)
    print(f"rf_tuned.joblib size: {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
