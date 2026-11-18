"""
HMM / Viterbi map-matching on top of the candidate classifier.

Hidden states
    For GPS fix i, the states are its candidate arcs from the SAME generator as round 1
    (features/candidates.py: R = 100 m, top-k = 8 with ties). Each state keeps
    (arc_id, proj_t), where proj_t is the position of the fix's projection along the arc.
    The candidate -> arc correspondence is never re-ordered.

Emission
    log P(fix_i | arc) := log(max(p_clf(arc | features_i), EMISSION_FLOOR))
    p_clf is the selected Random Forest's candidate probability.

Transition (Newson & Krumm style)
    d_gps = straight-line distance between the two fixes (m)
    d_net = network distance between the two projected positions:
              same arc  : |t_b - t_a| * len(arc)
              otherwise : (1 - t_a) * len(a) + shortest_path(to(a) -> from(b)) + t_b * len(b)
    log P(b | a) := max( -|d_net - d_gps| / beta , LOG_FLOOR )
    Shortest paths use a cutoff of d_gps + DETOUR_MAX. Pairs that are unreachable within
    the cutoff get the floor probability (very small, never zero).

Segmentation
    The chain is split, and every segment decoded independently, when
      * the time gap between consecutive fixes (with candidates) exceeds gap_s, or
      * no candidate of fix i can reach any candidate of fix i+1 (all transitions at
        the floor: no route is possible).
    Fixes with no candidate inside R are skipped (no state) and reported.

Decoding
    Viterbi in log space per segment:
      delta_0(j) = log e_0(j)
      delta_k(j) = log e_k(j) + max_i [ delta_{k-1}(i) + log A_k(i, j) ],  back_k(j) = argmax_i
    then backtracking from argmax_j delta_end(j).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EMISSION_FLOOR = 1e-6
LOG_FLOOR = float(np.log(1e-12))  # "very small probability, not zero"
DETOUR_MAX = 1000.0               # shortest-path cutoff = d_gps + DETOUR_MAX (m)
PATH_LIMIT = 3000.0               # cutoff when filling the route between matched arcs


@dataclass
class Lattice:
    point_idx: np.ndarray            # fixes that have >= 1 candidate, in time order
    t: np.ndarray
    xy: np.ndarray
    arcs: list                       # per fix: candidate arc ids
    pos: list                        # per fix: proj_t of each candidate
    log_e: list                      # per fix: log emission of each candidate
    prob: list                       # per fix: classifier probability of each candidate
    cost: list                       # cost[k] (k >= 1): |d_net - d_gps|, shape (K_{k-1}, K_k), inf = unreachable
    dt: np.ndarray                   # dt[k-1] between fix k-1 and fix k
    d_gps: np.ndarray
    skipped: np.ndarray              # point_idx of fixes without candidates
    n_points: int
    timings: dict = field(default_factory=dict)


def build_lattice(graph, track: pd.DataFrame, cand: pd.DataFrame, prob: np.ndarray) -> Lattice:
    """track: cleaned track with x, y, t (row i = point_idx i).
    cand : rows (point_idx, arc_id, proj_t) sorted by point_idx; prob aligned with cand."""
    import time
    t0 = time.perf_counter()
    cand = cand.assign(prob=np.asarray(prob, dtype=float))
    pidx = cand.point_idx.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(pidx)) + 1]
    ends = np.r_[starts[1:], len(cand)]
    valid = pidx[starts]
    arc_col, pos_col, p_col = cand.arc_id.to_numpy(), cand.proj_t.to_numpy(), cand.prob.to_numpy()
    arcs = [arc_col[s:e] for s, e in zip(starts, ends)]
    pos = [pos_col[s:e].astype(float) for s, e in zip(starts, ends)]
    prb = [p_col[s:e] for s, e in zip(starts, ends)]
    log_e = [np.log(np.maximum(p, EMISSION_FLOOR)) for p in prb]

    xy = track[["x", "y"]].to_numpy()[valid]
    t = track["t"].to_numpy()[valid]
    d_gps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(t)

    # ---- collect every shortest-path query of every step, solve them in one batch
    L, A_to, A_from = graph.arc_len, graph.arc_to, graph.arc_from
    q_src, q_tgt, q_lim, q_slices = [], [], [], []
    n_q = 0
    for k in range(1, len(valid)):
        a, b = arcs[k - 1], arcs[k]
        src = np.repeat(A_to[a], len(b))
        tgt = np.tile(A_from[b], len(a))
        q_src.append(src)
        q_tgt.append(tgt)
        q_lim.append(np.full(len(src), d_gps[k - 1] + DETOUR_MAX))
        q_slices.append((n_q, n_q + len(src)))
        n_q += len(src)
    if n_q:
        sp = graph.sp_lengths(np.concatenate(q_src), np.concatenate(q_tgt), np.concatenate(q_lim))
    t1 = time.perf_counter()

    cost = [None]
    for k in range(1, len(valid)):
        a, b = arcs[k - 1], arcs[k]
        ta, tb = pos[k - 1], pos[k]
        s0, s1 = q_slices[k - 1]
        spk = sp[s0:s1].reshape(len(a), len(b))
        d_net = (1 - ta)[:, None] * L[a][:, None] + spk + (tb * L[b])[None, :]
        same = a[:, None] == b[None, :]
        if same.any():
            d_same = np.abs(tb[None, :] - ta[:, None]) * L[a][:, None]
            d_net = np.where(same, d_same, d_net)
        cost.append(np.abs(d_net - d_gps[k - 1]))
    skipped = np.setdiff1d(np.arange(len(track)), valid)
    return Lattice(valid, t, xy, arcs, pos, log_e, prb, cost, dt, d_gps, skipped, len(track),
                   {"shortest_paths_s": t1 - t0, "lattice_s": time.perf_counter() - t0,
                    "n_sp_queries": int(n_q)})


def viterbi(lat: Lattice, beta: float, gap_s: float):
    """Returns (state index per valid fix, segment id per valid fix, info dict)."""
    n = len(lat.point_idx)
    states = np.zeros(n, dtype=np.int64)
    seg_id = np.zeros(n, dtype=np.int64)
    if n == 0:
        return states, seg_id, {"n_segments": 0, "splits_gap": 0, "splits_no_route": 0}
    back = [None] * n
    segments = []
    start, delta = 0, lat.log_e[0].copy()
    splits_gap = splits_noroute = 0
    for k in range(1, n):
        C = lat.cost[k]
        gap = lat.dt[k - 1] > gap_s
        noroute = not np.isfinite(C).any()
        if gap or noroute:
            splits_gap += int(gap)
            splits_noroute += int(noroute and not gap)
            segments.append((start, k - 1, delta))
            start, delta = k, lat.log_e[k].copy()
            continue
        log_a = np.maximum(-C / beta, LOG_FLOOR)
        score = delta[:, None] + log_a
        back[k] = score.argmax(axis=0)
        delta = score.max(axis=0) + lat.log_e[k]
    segments.append((start, n - 1, delta))
    for sid, (s, e, d) in enumerate(segments):
        j = int(np.argmax(d))
        states[e] = j
        for k in range(e, s, -1):
            j = int(back[k][j])
            states[k - 1] = j
        seg_id[s:e + 1] = sid
    return states, seg_id, {"n_segments": len(segments), "splits_gap": splits_gap,
                            "splits_no_route": splits_noroute}


def classifier_only(lat: Lattice):
    """Independent per-fix argmax of the classifier probability (no sequence model)."""
    states = np.array([int(np.argmax(p)) for p in lat.prob], dtype=np.int64)
    return states


def state_arcs(lat: Lattice, states: np.ndarray) -> np.ndarray:
    return np.array([lat.arcs[k][s] for k, s in enumerate(states)], dtype=np.int64)


def reconstruct_route(graph, lat: Lattice, arcs_seq: np.ndarray, seg_id: np.ndarray | None = None):
    """Turn per-fix matched arcs into a connected arc sequence by inserting shortest paths
    between consecutive different arcs. Returns (route_arcs, info). A break (-1) is
    inserted where the segment changes or no connecting path exists within PATH_LIMIT."""
    route, fills, breaks = [], 0, 0
    if seg_id is None:
        seg_id = np.zeros(len(arcs_seq), dtype=np.int64)
    for k, a in enumerate(arcs_seq):
        if k == 0:
            route.append(int(a))
            continue
        prev = route[-1] if route[-1] >= 0 else None
        if seg_id[k] != seg_id[k - 1]:
            route += [-1, int(a)]
            breaks += 1
            continue
        if a == prev:
            continue
        if graph.arc_to[prev] == graph.arc_from[a]:
            route.append(int(a))
            continue
        p = graph.path_arcs(int(graph.arc_to[prev]), int(graph.arc_from[a]), PATH_LIMIT)
        if p is None:
            route += [-1, int(a)]
            breaks += 1
        else:
            route += p + [int(a)]
            fills += 1
    return route, {"route_fills": fills, "route_breaks": breaks}
