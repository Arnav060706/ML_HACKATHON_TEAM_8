"""
Candidate miss rate on the 80 TRAINING trips for a grid of (R, k).

miss rate = fraction of GPS points whose reference arc (from labels.py) is NOT
in the point's candidate set. Points with no candidates at all count as misses.

Two versions are reported:
  miss_rate          over ALL points
  miss_rate_aligned  over points whose distance to their aligned reference arc
                     is <= UNALIGNED_M. Points further than that cannot be
                     explained by the reference route at all (route has gaps or
                     the track runs where the network has no road, e.g. trip
                     00000054) -- no radius choice can fix them, so they would
                     only blur the comparison between R values.
We also record the mean candidate-set size, because a larger R / k lowers the
miss rate but makes the classification (and the HMM state space) harder/slower.

Only the training partition is used to choose R and k (no test leakage).
Output: results/candidate_miss_rate.csv, results/label_alignment_stats.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import iter_trips  # noqa: E402
from features.candidates import generate_candidates, top_k_with_ties  # noqa: E402
from features.labels import assign_route_arcs  # noqa: E402

R_GRID = [15, 25, 35, 50, 75, 100]
K_GRID = [None, 4, 6, 8, 12, 16]
UNALIGNED_M = 100.0


def main():
    t0 = time.time()
    tallies = {}            # (R, k) -> [n_points, n_miss, n_cand_rows, n_aligned, n_miss_aligned]
    stats = []
    for trip in iter_trips("train"):
        cand = generate_candidates(trip, radius=max(R_GRID))
        true_arc, pos, ldist = assign_route_arcs(trip)
        n = len(trip.track)
        stats.append({"trip_id": trip.trip_id, "epsg": trip.epsg, "n_points": n,
                      "n_route_arcs": len(trip.route),
                      "label_dist_median": np.median(ldist),
                      "label_dist_p95": np.percentile(ldist, 95),
                      "label_dist_max": ldist.max(),
                      "frac_unaligned": float((ldist > UNALIGNED_M).mean()),
                      "route_arcs_with_points": len(np.unique(pos)) / max(len(trip.route), 1)})
        for R in R_GRID:
            cr = cand[cand.dist_m <= R]
            for k in K_GRID:
                ck = cr if k is None else top_k_with_ties(cr, k)
                hit = np.zeros(n, dtype=bool)
                m = ck.arc_id.to_numpy() == true_arc[ck.point_idx.to_numpy()]
                hit[ck.point_idx.to_numpy()[m]] = True
                ok = ldist <= UNALIGNED_M
                tl = tallies.setdefault((R, k), [0, 0, 0, 0, 0])
                tl[0] += n
                tl[1] += int((~hit).sum())
                tl[2] += len(ck)
                tl[3] += int(ok.sum())
                tl[4] += int((~hit & ok).sum())
        print(f"{trip.trip_id} done ({time.time() - t0:.0f}s)", flush=True)

    rows = [{"R_m": R, "k": "all" if k is None else k, "n_points": v[0],
             "miss_rate": v[1] / v[0], "n_aligned_points": v[3],
             "miss_rate_aligned": v[4] / v[3], "mean_candidates": v[2] / v[0]}
            for (R, k), v in tallies.items()]
    out = pd.DataFrame(rows).sort_values(["R_m", "k"], key=lambda s: s.astype(str).str.zfill(4))
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "candidate_miss_rate.csv", index=False)
    pd.DataFrame(stats).to_csv(ROOT / "results" / "label_alignment_stats.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
