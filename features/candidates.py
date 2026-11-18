"""
Candidate generation: for every GPS point, the road arcs within R metres.

Spatial index
-------------
Arcs are line segments, but a KD-tree indexes points. We therefore densify every
arc into sample points spaced at most STEP metres apart and build a
scipy cKDTree over those samples. Any location on an arc is at most STEP/2 from
its nearest sample, so a ball query of radius R + STEP/2 around the GPS point
returns a guaranteed SUPERSET of the arcs whose true (perpendicular) distance is
<= R. The exact point-to-segment distance is then computed for those arcs only,
filtered to <= R, and the k nearest are kept.

Only arcs whose bounding box touches the trip's GPS bounding box (+ margin) are
indexed: the supplied network covers a much larger area than the trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from features.geometry import point_segment

STEP = 10.0  # densification step in metres


def _densify(a: np.ndarray, b: np.ndarray, arc_ids: np.ndarray, step: float = STEP):
    seg_len = np.linalg.norm(b - a, axis=1)
    n = np.maximum(np.ceil(seg_len / step).astype(int), 1) + 1      # samples per arc
    owner = np.repeat(np.arange(len(a)), n)
    # fraction along the arc for each sample: 0, 1/(n-1), ..., 1
    start = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(owner.size) - start
    frac = k / np.repeat(n - 1, n)
    pts = a[owner] + frac[:, None] * (b[owner] - a[owner])
    return pts, arc_ids[owner]


def generate_candidates(trip, radius: float = 50.0, k: int | None = None,
                        step: float = STEP) -> pd.DataFrame:
    """Return a long table of (point_idx, arc_id, dist_m, proj_t) pairs.

    radius : search radius R in metres.
    k      : keep at most the k nearest arcs per point (None = keep all within R).
    """
    pxy = trip.track[["x", "y"]].to_numpy()
    a_all, b_all = trip.arc_xy

    # --- restrict to arcs near the trip --------------------------------------
    margin = radius + step
    lo = pxy.min(0) - margin
    hi = pxy.max(0) + margin
    amin = np.minimum(a_all, b_all)
    amax = np.maximum(a_all, b_all)
    near = np.all(amax >= lo, axis=1) & np.all(amin <= hi, axis=1)
    arc_ids = np.flatnonzero(near)
    a, b = a_all[arc_ids], b_all[arc_ids]

    # --- KD-tree over densified arc samples ------------------------------------
    samples, owner = _densify(a, b, np.arange(len(arc_ids)), step)
    tree = cKDTree(samples)
    hits = tree.query_ball_point(pxy, r=radius + step / 2.0)

    pi, li = [], []
    for i, h in enumerate(hits):
        if h:
            u = np.unique(owner[h])
            pi.append(np.full(u.size, i))
            li.append(u)
    if not pi:
        return pd.DataFrame(columns=["point_idx", "arc_id", "dist_m", "proj_t"])
    pi = np.concatenate(pi)
    li = np.concatenate(li)

    # --- exact distance + filter ------------------------------------------------
    d, t = point_segment(pxy[pi], a[li], b[li])
    keep = d <= radius
    df = pd.DataFrame({"point_idx": pi[keep], "arc_id": arc_ids[li[keep]],
                       "dist_m": d[keep], "proj_t": t[keep]})
    df = df.sort_values(["point_idx", "dist_m", "arc_id"], kind="stable")
    if k is not None:
        df = top_k_with_ties(df, k)
    return df.reset_index(drop=True)


def top_k_with_ties(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the k nearest arcs per point, but never split a distance tie.

    A two-way street is two arcs (u->v and v->u) with IDENTICAL geometry and
    therefore identical distance; cutting between them would arbitrarily drop
    the true direction. Expects df sorted by (point_idx, dist_m).
    """
    rank = df.groupby("point_idx", sort=False).cumcount().to_numpy()
    size = df.groupby("point_idx", sort=False).point_idx.transform("size").to_numpy()
    at_kth = rank == np.minimum(k, size) - 1
    kth = pd.Series(df.dist_m.to_numpy()[at_kth], index=df.point_idx.to_numpy()[at_kth])
    kth_per_row = kth.reindex(df.point_idx.to_numpy()).to_numpy()
    return df[df.dist_m.to_numpy() <= kth_per_row + 1e-6]
