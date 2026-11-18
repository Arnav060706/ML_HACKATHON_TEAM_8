"""
Sanity checks on the exported tables + simple non-learned baselines.

  * data-cleaning counts per trip (invalid / out-of-order / duplicate fixes)
  * per-feature mean for positives vs negatives (train)   -> results/feature_label_means.csv
  * rule baselines, Top-1 per GPS point on train and student_test:
        nearest arc           argmin dist_m
        nearest + heading     argmin dist_m + 0.5 * heading_diff   (no fitting; a fixed hand rule)
    Top-1 denominators are ALL points (points with no candidate / missed true
    arc count as wrong) and, separately, reliable points only. These are the
    floor every learned model (person B) must beat.     -> results/baseline_top1.csv
  * plot of the miss-rate sweep                          -> results/candidate_miss_rate.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.build_features import FEATURES  # noqa: E402

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def top1(F: pd.DataFrame, P: pd.DataFrame, score: np.ndarray) -> dict:
    F = F.assign(score=score)
    best = F.loc[F.groupby(["trip_id", "point_idx"]).score.idxmin()]
    hit = best.set_index(["trip_id", "point_idx"]).label.reindex(
        pd.MultiIndex.from_frame(P[["trip_id", "point_idx"]])).fillna(0).to_numpy()
    rel = P.label_reliable.to_numpy() == 1
    return {"top1_all_points": hit.mean(), "top1_reliable_points": hit[rel].mean()}


def main():
    rows, means = [], None
    for part in ["train", "student_test"]:
        F = pd.read_parquet(PROC / f"candidates_{part}.parquet")
        P = pd.read_parquet(PROC / f"points_{part}.parquet")
        if part == "train":
            means = F[F.label_reliable == 1].groupby("label")[FEATURES].mean().T
            means.columns = ["mean_negative", "mean_positive"]
            means.to_csv(RES / "feature_label_means.csv")
            print(means.round(2))
        for name, s in [("nearest_arc", F.dist_m.to_numpy()),
                        ("nearest_plus_heading",
                         (F.dist_m + 0.5 * F.heading_diff).to_numpy())]:
            r = {"partition": part, "baseline": name, **top1(F, P, s)}
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "baseline_top1.csv", index=False)
    print(out.to_string(index=False))

    # --- miss-rate plot -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = pd.read_csv(RES / "candidate_miss_rate.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, g in m.groupby("k", sort=False):
        ax[0].plot(g.R_m, 100 * g.miss_rate_aligned, marker="o", label=f"k={k}")
        ax[1].plot(g.mean_candidates, 100 * g.miss_rate_aligned, marker="o", label=f"k={k}")
    for a in ax:
        a.axhline(1.5, ls="--", c="grey", lw=0.8)
        a.set_ylabel("candidate miss rate (%)  [alignable train points]")
        a.grid(alpha=0.3)
    ax[0].set_xlabel("search radius R (m)")
    ax[1].set_xlabel("mean candidates per GPS point")
    sel = m[(m.R_m == 100) & (m.k.astype(str) == "8")].iloc[0]
    ax[1].scatter([sel.mean_candidates], [100 * sel.miss_rate_aligned], s=160,
                  facecolors="none", edgecolors="red", label="chosen R=100, k=8")
    ax[1].legend(fontsize=8)
    fig.suptitle("Candidate miss rate vs R and k (80 training trips)")
    fig.tight_layout()
    fig.savefig(RES / "candidate_miss_rate.png", dpi=130)


if __name__ == "__main__":
    main()
