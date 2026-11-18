"""
Raw-data loading and coordinate projection for PathSnap.

Each trip folder <trip_id>/ holds four header-less, whitespace-separated files:
    <trip_id>.track  lon  lat  t[s]          GPS observations (~1 Hz)
    <trip_id>.nodes  lon  lat                road-network nodes (row index = node_id)
    <trip_id>.arcs   from_node  to_node      directed road segments (row index = arc_id)
    <trip_id>.route  arc_id                  reference route, chronological arc sequence

Projection choice
-----------------
Longitude/latitude are angles, so Euclidean distance in degrees is meaningless
(1 deg of longitude is ~111 km at the equator but ~72 km at 49 N). The trips are
spread over several countries (e.g. ~8.6E/49.9N, ~11E/47.7N, ~54E/56N), so ONE
projected CRS cannot serve every trip. For each trip we therefore pick the WGS84
UTM zone that contains the trip's GPS centroid:

    zone = floor((lon + 180) / 6) + 1,  EPSG = 32600 + zone (north) / 32700 + zone (south)

UTM is conformal, in metres, and its scale error inside a 6-degree zone is
< 0.1 %, i.e. < 1 m over a 1 km span -- far below GPS noise (5-20 m). A whole trip
(a few km) always sits well inside one zone, so distances, headings and segment
lengths can all be computed with plain planar geometry afterwards.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

# Default location of the raw data (can be overridden with env var PATHSNAP_DATA).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "Student folder" / "Student folder"
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", DEFAULT_DATA_DIR))
PARTITIONS = ("train", "student_test")


def _read(path: Path, ncols: int, dtype) -> np.ndarray:
    """Fast reader for header-less whitespace-separated numeric files."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=dtype)
    return df.to_numpy()


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 / UTM zone containing (lon, lat)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # lon, lat, t, x, y
    nodes_ll: np.ndarray         # (N, 2) lon, lat
    nodes_xy: np.ndarray         # (N, 2) metres
    arcs: np.ndarray             # (A, 2) from_node, to_node
    route: np.ndarray            # (R,) arc ids (empty if not supplied)
    epsg: int
    meta: dict = field(default_factory=dict)

    # --- convenience geometry -------------------------------------------------
    @property
    def arc_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Start and end coordinates (metres) of every arc."""
        return self.nodes_xy[self.arcs[:, 0]], self.nodes_xy[self.arcs[:, 1]]


def list_trips(partition: str, data_dir: Path = DATA_DIR) -> list[str]:
    root = Path(data_dir) / partition
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def find_trip_dir(trip_id: str, data_dir: Path = DATA_DIR) -> Path:
    """Locate <data_dir>/<any partition folder>/<trip_id> (train, student_test, evaluator set, ...)."""
    data_dir = Path(data_dir)
    for part in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if (part / trip_id).is_dir():
            return part / trip_id
    if (data_dir / trip_id).is_dir():
        return data_dir / trip_id
    raise FileNotFoundError(f"trip {trip_id} not found under {data_dir}")


def load_trip(trip_id: str, partition: str, data_dir: Path = DATA_DIR,
              load_route: bool = True) -> Trip:
    return load_trip_dir(Path(data_dir) / partition / trip_id, load_route=load_route,
                         partition=partition)


def load_trip_dir(d: Path, load_route: bool = True, partition: str | None = None) -> Trip:
    """Load one trip folder. With load_route=False the .route file is never opened
    (used by the inference pipeline so reference labels cannot leak into predictions)."""
    d = Path(d)
    trip_id = d.name
    partition = partition or d.parent.name
    track = _read(d / f"{trip_id}.track", 3, np.float64)
    nodes = _read(d / f"{trip_id}.nodes", 2, np.float64)
    arcs = _read(d / f"{trip_id}.arcs", 2, np.int64)
    route = (_read(d / f"{trip_id}.route", 1, np.int64).ravel()
             if load_route else np.empty(0, dtype=np.int64))

    tr = pd.DataFrame(track, columns=["lon", "lat", "t"])
    meta = {"n_raw_points": len(tr)}

    # --- cleaning: drop invalid / non-finite / out-of-range / duplicate points ---
    ok = np.isfinite(tr[["lon", "lat", "t"]]).all(axis=1)
    ok &= tr.lon.between(-180, 180) & tr.lat.between(-90, 90)
    ok &= ~((tr.lon == 0) & (tr.lat == 0))
    tr = tr[ok]
    # sort by time (stable) and remove duplicated timestamps (keep first)
    meta["n_out_of_order"] = int((np.diff(tr.t.to_numpy()) < 0).sum())
    tr = tr.sort_values("t", kind="stable").drop_duplicates("t", keep="first")
    tr = tr.reset_index(drop=True)
    meta["n_clean_points"] = len(tr)

    # --- projection: per-trip UTM zone from the GPS centroid ---------------------
    epsg = utm_epsg(float(tr.lon.mean()), float(tr.lat.mean()))
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)  # lon,lat order
    tr["x"], tr["y"] = tf.transform(tr.lon.to_numpy(), tr.lat.to_numpy())
    nx_, ny_ = tf.transform(nodes[:, 0], nodes[:, 1])
    nodes_xy = np.column_stack([nx_, ny_])

    return Trip(trip_id, partition, tr, nodes, nodes_xy, arcs, route, epsg, meta)


def iter_trips(partition: str, data_dir: Path = DATA_DIR, **kw):
    for tid in list_trips(partition, data_dir):
        yield load_trip(tid, partition, data_dir, **kw)


if __name__ == "__main__":
    import time
    t0 = time.time()
    trip = load_trip(list_trips("train")[0], "train")
    print(trip.trip_id, trip.epsg, trip.meta, len(trip.nodes_xy), len(trip.arcs),
          len(trip.route), f"{time.time() - t0:.2f}s")
    print(trip.track.head())
