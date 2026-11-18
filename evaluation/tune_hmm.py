"""
Tune the HMM decoder parameters beta (transition scale, metres) and gap_s (chain-split
threshold, seconds) on the TRAIN and STUDENT_TEST trips only.

* train trips use OUT-OF-FOLD classifier probabilities (models/oof_probabilities.py), so the
  emissions are as (un)reliable as they will be on unseen trips;
* student_test trips use the final model's probabilities.
* The candidate sets are the exported round-1 tables (same generator as inference).

For every trip the lattice (candidates, emissions, shortest-path transition costs) is built
ONCE; every (beta, gap_s) pair then only re-runs Viterbi, which is cheap.

Selection rule: highest pooled Top-1 accuracy on reliable TRAIN fixes; student_test is the
confirmation set. Evaluator data is never used.

Outputs: results/hmm_tuning.csv (every config x partition), results/hmm_tuning_per_trip.csv,
         inference/config.json (beta, gap_s updated), results/hmm_tuning.png
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

from data_processing.load_data import DATA_DIR  # noqa: E402
from evaluation.evaluate import continuity, reference_labels, route_overlap  # noqa: E402
from inference.hmm import (build_lattice, classifier_only, reconstruct_route,  # noqa: E402
                           state_arcs, viterbi)
from inference.road_graph import RoadGraph  # noqa: E402

BETAS = [1, 2, 3, 5, 8, 12, 20, 35, 60]
GAPS = [10, 30, 60, 120, 300, float("inf")]
PROC = ROOT / "data" / "processed"


def trip_rows(partition, trip_id, cand, prob, configs):
    tdir = DATA_DIR / partition / trip_id
    trip, true_arc, reliable = reference_labels(tdir)      # evaluation side: reads .route
    graph = RoadGraph(trip)                                  # graph never uses the route
    lat = build_lattice(graph, trip.track, cand, prob)
    a, b = trip.arc_xy
    arc_len = np.linalg.norm(b - a, axis=1)
    n = lat.n_points
    rows = []

    def record(name, beta, gap, arcs, seg, info):
        pred = np.full(n, -1, dtype=np.int64)
        pred[lat.point_idx] = arcs
        hit = pred == true_arc
        route, rinfo = reconstruct_route(graph, lat, arcs, seg)
        r = {"partition": partition, "trip_id": trip_id, "decoder": name, "beta": beta, "gap_s": gap,
             "n_points": n, "n_reliable": int(reliable.sum()), "hits_all": int(hit.sum()),
             "hits_reliable": int((hit & reliable).sum()),
             "continuity_pairs": max(len(arcs) - 1, 0),
             "continuity_ok": continuity(pred, trip.arcs[:, 0], trip.arcs[:, 1]) * max(len(arcs) - 1, 0)
             if len(arcs) > 1 else 0.0,
             "n_segments": info.get("n_segments", 1), "splits_gap": info.get("splits_gap", 0),
             "splits_no_route": info.get("splits_no_route", 0), **rinfo}
        ov = route_overlap(route, trip.route, arc_len)
        r.update({k: v for k, v in ov.items()})
        r["ref_len_m"] = float(arc_len[np.unique(trip.route)].sum())
        rows.append(r)

    clf = state_arcs(lat, classifier_only(lat))
    record("classifier_only", np.nan, np.nan, clf, np.zeros(len(clf), dtype=np.int64), {})
    for beta, gap in configs:
        st, seg, info = viterbi(lat, beta, gap)
        record("hmm", beta, gap, state_arcs(lat, st), seg, info)
    return rows


def pooled(df):
    g = df.groupby(["partition", "decoder", "beta", "gap_s"], dropna=False)
    out = g.agg(n_points=("n_points", "sum"), n_reliable=("n_reliable", "sum"),
                hits_all=("hits_all", "sum"), hits_reliable=("hits_reliable", "sum"),
                cont_ok=("continuity_ok", "sum"), cont_pairs=("continuity_pairs", "sum"),
                route_f1_mean=("route_f1_len", "mean"), route_mismatch_mean=("route_mismatch_frac", "mean"),
                route_breaks=("route_breaks", "sum"), segments=("n_segments", "sum"),
                splits_gap=("splits_gap", "sum"), splits_no_route=("splits_no_route", "sum")).reset_index()
    out["top1_all"] = out.hits_all / out.n_points
    out["top1_reliable"] = out.hits_reliable / out.n_reliable
    out["route_continuity"] = out.cont_ok / out.cont_pairs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="rf_compact", help="suffix of data/processed/probs_*_<name>.parquet")
    ap.add_argument("--max-train-trips", type=int, default=None)
    args = ap.parse_args()
    configs = [(b, g) for b in BETAS for g in GAPS]
    rows, t0 = [], time.time()
    for part, pfile in [("train", f"probs_train_oof_{args.probs}.parquet"),
                        ("student_test", f"probs_student_test_{args.probs}.parquet")]:
        cand = pd.read_parquet(PROC / f"candidates_{part}.parquet",
                               columns=["trip_id", "point_idx", "arc_id", "proj_t"])
        pr = pd.read_parquet(PROC / pfile)
        cand = cand.merge(pr, on=["trip_id", "point_idx", "arc_id"], how="left", validate="one_to_one")
        assert cand.prob.notna().all()
        trips = sorted(cand.trip_id.unique())
        if part == "train" and args.max_train_trips:
            trips = trips[:args.max_train_trips]
        for tid in trips:
            c = cand[cand.trip_id == tid].sort_values(["point_idx", "arc_id"])
            rows += trip_rows(part, tid, c[["point_idx", "arc_id", "proj_t"]], c.prob.to_numpy(), configs)
            print(f"[{part}] {tid} done ({time.time() - t0:.0f}s)", flush=True)

    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(ROOT / "results" / "hmm_tuning_per_trip.csv", index=False)
    pool = pooled(per_trip)
    pool.to_csv(ROOT / "results" / "hmm_tuning.csv", index=False)

    tr = pool[(pool.partition == "train") & (pool.decoder == "hmm")]
    best = tr.sort_values(["top1_reliable", "route_f1_mean"], ascending=False).iloc[0]
    cfg_path = ROOT / "inference" / "config.json"
    cfg = json.load(open(cfg_path))
    cfg.update({"beta": float(best.beta), "gap_s": float(best.gap_s) if np.isfinite(best.gap_s) else 1e12,
                "note": f"beta/gap_s chosen by evaluation/tune_hmm.py on train OOF probabilities "
                        f"({args.probs}); max pooled Top-1 on reliable train fixes"})
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print("chosen:", best[["beta", "gap_s", "top1_reliable", "top1_all", "route_continuity",
                           "route_f1_mean"]].to_dict())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for part, a in zip(["train", "student_test"], ax):
        sub = pool[(pool.partition == part) & (pool.decoder == "hmm")]
        for gap, g in sub.groupby("gap_s"):
            a.plot(g.beta, 100 * g.top1_reliable, marker="o", label=f"gap {gap:g} s")
        clf = pool[(pool.partition == part) & (pool.decoder == "classifier_only")].top1_reliable.iloc[0]
        a.axhline(100 * clf, color="k", ls="--", label="classifier only")
        a.set_xscale("log")
        a.set_xlabel("beta (m, log scale)")
        a.set_ylabel("Top-1 accuracy, reliable fixes (%)")
        a.set_title(f"{part} ({'out-of-fold probabilities' if part == 'train' else 'final model'})")
        a.grid(alpha=0.3)
    ax[0].axvline(best.beta, color="red", lw=0.8)
    ax[1].legend(fontsize=7)
    fig.suptitle("HMM decoder tuning: beta x gap threshold")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "hmm_tuning.png", dpi=140)


if __name__ == "__main__":
    main()
