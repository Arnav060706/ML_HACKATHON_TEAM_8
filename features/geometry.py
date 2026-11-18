"""Planar geometry helpers (inputs are projected UTM metres)."""
from __future__ import annotations

import numpy as np


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised point-to-segment projection.

    p, a, b : (..., 2) arrays broadcastable against each other.
    Returns (distance_m, t) where t in [0, 1] is the position of the
    orthogonal projection along a->b (clipped to the segment).
    """
    v = b - a
    w = p - a
    vv = np.einsum("...i,...i->...", v, v)
    t = np.einsum("...i,...i->...", w, v) / np.where(vv > 0, vv, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * v
    d = np.linalg.norm(p - proj, axis=-1)
    return d, t


def bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass bearing in degrees (0 = north/+y, 90 = east/+x)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
