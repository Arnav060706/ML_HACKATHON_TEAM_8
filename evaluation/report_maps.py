"""
Folium maps for the report (Section 6): GPS fixes, nearby road network, reference route,
predicted route (classifier + HMM) and classifier-only route, for student-test trips.

The prediction comes from inference/predict.py (never reads .route); the reference route is read
here only to draw it (evaluation side).

Usage:  python evaluation/report_maps.py 00000040 00000077
Output: results/maps/report_<trip_id>.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import DATA_DIR, load_trip_dir  # noqa: E402
from evaluation.evaluate import evaluate, read_route  # noqa: E402
from features.candidates import generate_candidates  # noqa: E402
from inference.predict import load_config, load_model, predict_trip  # noqa: E402

PART = "student_test"


def seg(trip, a):
    u, v = trip.arcs[a]
    return [trip.nodes_ll[u][::-1].tolist(), trip.nodes_ll[v][::-1].tolist()]


def make_map(trip_id, model, cfg):
    tdir = DATA_DIR / PART / trip_id
    trip = load_trip_dir(tdir, load_route=False)
    pred = predict_trip(tdir, model=model, config=cfg)
    m_eval = evaluate(pred, trip_dir=tdir)
    ref = read_route(tdir)
    gps = pred["gps_latlon"]
    m = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="OpenStreetMap")

    g_net = folium.FeatureGroup("Road network (arcs within 100 m)", show=True)
    for a in sorted(set(generate_candidates(trip, radius=100.0).arc_id)):
        folium.PolyLine(seg(trip, a), color="#8c8c8c", weight=1.5, opacity=0.6).add_to(g_net)
    g_net.add_to(m)

    g_ref = folium.FeatureGroup("Reference route", show=True)
    for a in ref:
        folium.PolyLine(seg(trip, a), color="#1a9641", weight=9, opacity=0.45).add_to(g_ref)
    g_ref.add_to(m)

    g_clf = folium.FeatureGroup("Classifier-only route (RF argmax)", show=False)
    from inference.predict import route_polyline
    for line in route_polyline(trip, pred["route_arcs_classifier"]):
        folium.PolyLine(line, color="#ff8c00", weight=3, opacity=0.9, dash_array="6,6").add_to(g_clf)
    g_clf.add_to(m)

    g_pred = folium.FeatureGroup("Predicted route (RF + HMM/Viterbi)", show=True)
    for line in pred["route_latlon"]:
        folium.PolyLine(line, color="#d7191c", weight=3.5, opacity=0.95).add_to(g_pred)
    g_pred.add_to(m)

    g_gps = folium.FeatureGroup("Raw GPS fixes", show=True)
    for i, (la, lo) in enumerate(gps):
        if i % 2 == 0:  # every 2nd fix keeps the HTML light
            folium.CircleMarker([la, lo], radius=2, color="#2b83ba", fill=True, weight=0,
                                fill_opacity=0.8).add_to(g_gps)
    g_gps.add_to(m)

    h, c = m_eval["hmm"], m_eval["classifier_only"]
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;
      padding:9px 12px;border:1px solid #888;border-radius:4px;font-size:13px;line-height:1.5">
      <b>Trip {trip_id} (student_test)</b><br>
      <span style="color:#8c8c8c">&#9644;</span> road network (arcs within 100 m)<br>
      <span style="color:#1a9641">&#9644;&#9644;</span> reference route<br>
      <span style="color:#d7191c">&#9644;</span> predicted route (RF + HMM)<br>
      <span style="color:#ff8c00">- - -</span> classifier-only route (toggle in layers)<br>
      <span style="color:#2b83ba">&#9679;</span> raw GPS fix (every 2nd)<br>
      <b>Top-1:</b> HMM {h['top1_all']:.3f} | classifier {c['top1_all']:.3f}<br>
      <b>Route F1:</b> HMM {h['route_f1_len']:.3f} | classifier {c['route_f1_len']:.3f}</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    out = ROOT / "results" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / f"report_{trip_id}.html")
    print(f"saved results/maps/report_{trip_id}.html  HMM Top-1 {h['top1_all']:.4f}")


if __name__ == "__main__":
    cfg = load_config()
    model = load_model(cfg)
    for tid in (sys.argv[1:] or ["00000040", "00000077"]):
        make_map(tid, model, cfg)
