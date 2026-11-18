"""
Directed road graph of one trip, built from its .nodes/.arcs (no .route involved).

* Nodes = road-network nodes, edges = arcs (directed, from_node -> to_node), edge
  weight = arc length in metres (in the trip's UTM projection).
* Only the corridor of nodes within CORRIDOR_M of some GPS fix is kept: the supplied
  networks cover far more than the trip (up to 166k arcs), and a route between two
  consecutive fixes never needs to leave a 1.5 km corridor around the track.
* Parallel arcs between the same (u, v) keep the shortest one; self-loops are dropped
  (they can never shorten a path).
* Shortest paths: scipy.sparse.csgraph.dijkstra with a distance `limit` (cutoff), run
  for many sources at once (C implementation; pure-Python Dijkstra was too slow for
  ~10^5 transition pairs per trip).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

CORRIDOR_M = 1500.0
BATCH = 48          # sources per dijkstra call (memory: BATCH x n_corridor x 8 bytes)
PATH_CACHE = 256    # predecessor rows kept for route reconstruction


class RoadGraph:
    def __init__(self, trip, corridor_m: float = CORRIDOR_M):
        self.arcs = trip.arcs
        a, b = trip.arc_xy
        self.arc_len = np.linalg.norm(b - a, axis=1)
        self.arc_from = trip.arcs[:, 0]
        self.arc_to = trip.arcs[:, 1]
        n_all = len(trip.nodes_xy)

        # corridor: nodes within corridor_m of any GPS fix
        gps = cKDTree(trip.track[["x", "y"]].to_numpy())
        d, _ = gps.query(trip.nodes_xy, k=1, distance_upper_bound=corridor_m)
        in_c = np.isfinite(d)
        self.local = np.full(n_all, -1, dtype=np.int64)
        self.local[in_c] = np.arange(in_c.sum())
        self.glob = np.flatnonzero(in_c)
        self.n = int(in_c.sum())

        u, v = self.arc_from, self.arc_to
        keep = in_c[u] & in_c[v] & (u != v)
        lu, lv, w = self.local[u[keep]], self.local[v[keep]], self.arc_len[keep]
        order = np.lexsort((w, lv, lu))                      # shortest first per (u, v)
        lu, lv, w, ids = lu[order], lv[order], w[order], np.flatnonzero(keep)[order]
        first = np.r_[True, (lu[1:] != lu[:-1]) | (lv[1:] != lv[:-1])]
        lu, lv, w, ids = lu[first], lv[first], w[first], ids[first]
        # tiny epsilon so zero-length arcs are still stored as edges in the sparse matrix
        self.G = csr_matrix((np.maximum(w, 1e-3), (lu, lv)), shape=(self.n, self.n))
        self.edge_arc = {(int(x), int(y)): int(i) for x, y, i in zip(lu, lv, ids)}
        self._pred = OrderedDict()

    # ------------------------------------------------------------------ distances
    def sp_lengths(self, src: np.ndarray, tgt: np.ndarray, limit: np.ndarray) -> np.ndarray:
        """Shortest-path length (m) from global node src[i] to global node tgt[i];
        inf if unreachable within limit[i] or outside the corridor."""
        out = np.full(len(src), np.inf)
        same = src == tgt
        out[same] = 0.0
        ls, lt = self.local[src], self.local[tgt]
        ok = ~same & (ls >= 0) & (lt >= 0)
        if not ok.any():
            return out
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(ls[idx], kind="stable")]
        srcs, starts = np.unique(ls[order], return_index=True)
        bounds = np.r_[starts, len(order)]
        for b0 in range(0, len(srcs), BATCH):
            b1 = min(b0 + BATCH, len(srcs))
            rows = order[bounds[b0]:bounds[b1]]
            lim = float(limit[rows].max())
            D = dijkstra(self.G, directed=True, indices=srcs[b0:b1], limit=lim)
            r = np.searchsorted(srcs[b0:b1], ls[rows])
            dist = D[r, lt[rows]]
            dist[dist > limit[rows]] = np.inf
            out[rows] = dist
        return out

    # ----------------------------------------------------------------- paths
    def path_arcs(self, u: int, v: int, limit: float) -> list[int] | None:
        """Arc ids of a shortest path from global node u to global node v (None if none)."""
        if u == v:
            return []
        lu, lv = self.local[u], self.local[v]
        if lu < 0 or lv < 0:
            return None
        key = (int(lu), float(limit))
        pred = self._pred.get(key)
        if pred is None:
            _, pred = dijkstra(self.G, directed=True, indices=int(lu), limit=limit,
                               return_predecessors=True)
            self._pred[key] = pred
            if len(self._pred) > PATH_CACHE:
                self._pred.popitem(last=False)
        if pred[lv] < 0:
            return None
        nodes = [int(lv)]
        while nodes[-1] != lu:
            nodes.append(int(pred[nodes[-1]]))
            if len(nodes) > 100000:
                return None
        nodes.reverse()
        return [self.edge_arc[(x, y)] for x, y in zip(nodes[:-1], nodes[1:])]
