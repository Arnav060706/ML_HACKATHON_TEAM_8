"""
Label construction: map every GPS point to one arc of the reference route.

The .route file is only an ordered list of arc ids -- it does NOT say which GPS
point was on which arc. We infer that with a monotone alignment:

    find j_0 <= j_1 <= ... <= j_{N-1}  (indices into the route list)
    minimising  sum_i  min(dist(point_i, route_arc[j_i]), CAP)

i.e. each point goes to a nearby route arc, but the position in the route can
never move backwards in time. This is solved exactly by dynamic programming:

    C[0, j] = D[0, j]
    C[i, j] = D[i, j] + min_{j' <= j} C[i-1, j']          (prefix minimum)

O(N * M) time with numpy (N points, M route arcs). Compared with a greedy
"nearest route arc" rule it (a) cannot jump backwards when the route passes the
same street twice or doubles back, and (b) cannot pick the reverse twin of a
two-way street out of order. CAP limits the influence of outlier GPS fixes.
"""
from __future__ import annotations

import numpy as np

from features.geometry import point_segment

CAP = 200.0  # metres; robust cap on per-point cost


def assign_route_arcs(trip, cap: float = CAP):
    """Return (true_arc_id[N], route_pos[N], label_dist_m[N]) for the trip's points."""
    route = trip.route
    pxy = trip.track[["x", "y"]].to_numpy()
    n, m = len(pxy), len(route)
    if n == 0 or m == 0:
        return (np.full(n, -1), np.full(n, -1), np.full(n, np.nan))

    a_all, b_all = trip.arc_xy
    a, b = a_all[route], b_all[route]                     # (M, 2)
    D, _ = point_segment(pxy[:, None, :], a[None], b[None])  # (N, M)
    Dc = np.minimum(D, cap)

    cols = np.arange(m)
    back = np.empty((n, m), dtype=np.int32)
    C = Dc[0].copy()
    for i in range(1, n):
        pm = np.minimum.accumulate(C)
        # index of the prefix minimum (latest j' <= j attaining it)
        back[i] = np.maximum.accumulate(np.where(C == pm, cols, 0))
        C = Dc[i] + pm

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(C))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    true_arc = route[pos]
    label_dist = D[np.arange(n), pos]
    return true_arc, pos, label_dist
