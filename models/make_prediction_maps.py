"""
Folium maps of real model predictions for the report (part B).

Layers: raw GPS fixes (blue), nearby road network (grey, arcs within 100 m of a fix),
reference route (green), predicted route from predict_trip() (red) and, hidden by default,
each fix snapped to its predicted arc (orange).

Usage (from the repo root)
  python models/make_prediction_maps.py                      # first 2 trips of student_test
  python models/make_prediction_maps.py 00000003 00000007    # chosen trips
  python models/make_prediction_maps.py --partition student_test
  PATHSNAP_MAP_PARTITION=student_test python models/make_prediction_maps.py

The model path comes from inference/config.json. Output: results/maps/prediction_<trip_id>.html
Only student_test (or train) trips are accepted; evaluator partitions are refused.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import DATA_DIR, list_trips, load_trip_dir  # noqa: E402
from evaluation.evaluate import read_route  # noqa: E402
from features.candidates import generate_candidates  # noqa: E402
from inference.predict import load_config, load_model, predict_trip  # noqa: E402

OUT = ROOT / "results" / "maps"
N_DEFAULT_TRIPS = 2


def make_map(trip_id: str, partition: str, model, cfg) -> Path:
    tdir = DATA_DIR / partition / trip_id
    trip = load_trip_dir(tdir, load_route=False)
    ref = read_route(tdir)  # reference route: drawn only, never given to the model
    pred = predict_trip(tdir, model=model, config=cfg)

    seg = lambda a: [trip.nodes_ll[trip.arcs[a][0]][::-1].tolist(),
                     trip.nodes_ll[trip.arcs[a][1]][::-1].tolist()]
    near = sorted(set(generate_candidates(trip, radius=100.0).arc_id.tolist()))
    gps = pred["gps_latlon"]

    m = folium.Map(location=[sum(p[0] for p in gps) / len(gps), sum(p[1] for p in gps) / len(gps)],
                   zoom_start=15, tiles="OpenStreetMap")

    g_net = folium.FeatureGroup("road network (arcs within 100 m of a fix)", show=True)
    for a in near:
        folium.PolyLine(seg(a), color="#9a9a9a", weight=1, opacity=0.6).add_to(g_net)
    g_net.add_to(m)

    g_ref = folium.FeatureGroup("reference route", show=True)
    for a in dict.fromkeys(ref.tolist()):  # unique arcs, keeps order
        folium.PolyLine(seg(a), color="green", weight=7, opacity=0.6).add_to(g_ref)
    g_ref.add_to(m)

    g_pred = folium.FeatureGroup("predicted route (classifier + HMM)", show=True)
    for piece in pred["route_latlon"]:
        folium.PolyLine(piece, color="red", weight=3, opacity=0.9).add_to(g_pred)
    g_pred.add_to(m)

    g_gps = folium.FeatureGroup("raw GPS fixes", show=True)
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, weight=0).add_to(g_gps)
    g_gps.add_to(m)

    g_match = folium.FeatureGroup("fixes snapped to predicted arc", show=False)
    for p in pred["matched_latlon"]:
        if p is not None:
            folium.CircleMarker(p, radius=2, color="orange", fill=True, weight=0).add_to(g_match)
    g_match.add_to(m)

    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
      padding:8px 12px;border:1px solid #888;font-size:13px">
      <b>Prediction: trip {trip_id} ({partition})</b><br>
      <span style="color:#9a9a9a">&#9644;</span> road network (within 100 m)<br>
      <span style="color:green">&#9644;</span> reference route<br>
      <span style="color:red">&#9644;</span> predicted route<br>
      <span style="color:blue">&#9679;</span> raw GPS fix<br>
      <span style="color:orange">&#9679;</span> fix snapped to predicted arc (off by default)</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)],
                  [max(p[0] for p in gps), max(p[1] for p in gps)]])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"prediction_{trip_id}.html"
    m.save(path)
    print(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB) | {pred['n_points']} fixes, "
          f"{len(near)} network arcs, {len(set(ref.tolist()))} reference arcs, "
          f"{len(pred['route_arcs'])} predicted arcs", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trips", nargs="*", help="trip ids (default: first 2 of the partition)")
    ap.add_argument("--partition", default=os.environ.get("PATHSNAP_MAP_PARTITION", "student_test"))
    args = ap.parse_args()
    if "evaluator" in args.partition.lower():
        sys.exit("refusing to draw evaluator trips")
    trips = args.trips or list_trips(args.partition, DATA_DIR)[:N_DEFAULT_TRIPS]
    cfg = load_config()
    model = load_model(cfg)
    print(f"model: {cfg['model_path']} | partition: {args.partition} | trips: {trips}", flush=True)
    for t in trips:
        make_map(t, args.partition, model, cfg)


if __name__ == "__main__":
    main()
