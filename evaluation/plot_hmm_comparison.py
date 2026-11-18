"""
Per-trip plot of classifier-only vs classifier + HMM on the 10 student-test trips, from
results/hmm_vs_classifier.csv (written by evaluation/compare_hmm.py).

Output: results/hmm_vs_classifier.png
  panel 1: observation-level Top-1 accuracy
  panel 2: route continuity (consecutive predicted arcs equal or directly connected)
  panel 3: whole-sequence route F1 (length-weighted overlap with the reference route)
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(ROOT / "results" / "hmm_vs_classifier.csv", dtype={"trip_id": str})
    trips = d[~d.trip_id.str.startswith("ALL")]
    allr = d[d.trip_id.str.startswith("ALL")].set_index("decoder")
    metrics = [("top1_all", "Top-1 accuracy (per GPS point)"),
               ("route_continuity", "Route continuity"),
               ("route_f1_len", "Route F1, length-weighted (whole sequence)")]
    ids = sorted(trips.trip_id.unique())
    x = np.arange(len(ids))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (m, title) in zip(axes, metrics):
        p = trips.pivot(index="trip_id", columns="decoder", values=m).loc[ids]
        ax.bar(x - 0.2, p["classifier_only"], 0.4, label="classifier only (RF argmax)", color="#9ecae1")
        ax.bar(x + 0.2, p["hmm"], 0.4, label="classifier + HMM/Viterbi", color="#08519c")
        lo = min(p.min().min(), 1) - 0.05
        ax.set_ylim(max(0, lo), 1.005)
        ax.set_title(f"{title}: pooled {allr.loc['classifier_only', m]:.3f} -> {allr.loc['hmm', m]:.3f}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left")
    axes[-1].set_xticks(x, [i[-2:] for i in ids])
    axes[-1].set_xlabel("student-test trip (last two digits of id)")
    fig.suptitle("Classifier-only vs classifier + HMM on the 10 student-test trips "
                 "(beta = 5 m, gap = 60 s; compact RF)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "hmm_vs_classifier.png", dpi=140)
    print("saved results/hmm_vs_classifier.png")


if __name__ == "__main__":
    main()
