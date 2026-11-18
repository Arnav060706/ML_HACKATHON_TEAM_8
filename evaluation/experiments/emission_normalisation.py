# Experiment: raw vs per-fix normalised emissions, finer beta grid (gap 60 s). Run from repo root.
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
from data_processing.load_data import DATA_DIR
from evaluation.evaluate import reference_labels
from inference.hmm import build_lattice, viterbi, state_arcs, EMISSION_FLOOR
from inference.road_graph import RoadGraph
BETAS = [2, 3, 4, 5, 6, 7]
rows = []
t0 = time.time()
for part, pf in [("train", "probs_train_oof_rf_compact"), ("student_test", "probs_student_test_rf_compact")]:
    cand = pd.read_parquet(f"data/processed/candidates_{part}.parquet", columns=["trip_id", "point_idx", "arc_id", "proj_t"])
    cand = cand.merge(pd.read_parquet(f"data/processed/{pf}.parquet"), on=["trip_id", "point_idx", "arc_id"])
    for tid in sorted(cand.trip_id.unique()):
        c = cand[cand.trip_id == tid].sort_values(["point_idx", "arc_id"])
        trip, true_arc, rel = reference_labels(DATA_DIR / part / tid)
        lat = build_lattice(RoadGraph(trip), trip.track, c[["point_idx", "arc_id", "proj_t"]], c.prob.to_numpy())
        raw = lat.log_e
        norm = [np.log(np.maximum(p / p.sum(), EMISSION_FLOOR)) if p.sum() > 0 else l for p, l in zip(lat.prob, raw)]
        for emis, le in [("raw", raw), ("normalised", norm)]:
            lat.log_e = le
            for b in BETAS:
                st, seg, _ = viterbi(lat, b, 60.0)
                pred = np.full(lat.n_points, -1); pred[lat.point_idx] = state_arcs(lat, st)
                hit = pred == true_arc
                rows.append((part, tid, emis, b, lat.n_points, int(rel.sum()), int(hit.sum()), int((hit & rel).sum())))
    print(part, "done", round(time.time() - t0), flush=True)
d = pd.DataFrame(rows, columns=["partition", "trip_id", "emission", "beta", "n", "n_rel", "hit", "hit_rel"])
d.to_csv("results/hmm_emission_experiment_per_trip.csv", index=False)
g = d.groupby(["partition", "emission", "beta"])[["n", "n_rel", "hit", "hit_rel"]].sum()
g["top1_all"] = g.hit / g.n; g["top1_reliable"] = g.hit_rel / g.n_rel
g.reset_index().to_csv("results/hmm_emission_experiment.csv", index=False)
print(g[["top1_all", "top1_reliable"]].round(4).unstack(0))
