"""
Folium map to visually verify the label construction on one trip.

Layers: road arcs near the trip (grey), reference route (green), raw GPS fixes
(blue), and for every Nth fix a red line from the GPS fix to its projection on
the arc it was LABELLED with (output of the monotone alignment). If the labels
are right, the red ticks are short and land on the green route.

Usage: python features/label_check_map.py [trip_id] [partition]
Output: results/maps/label_check_<trip_id>.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.load_data import load_trip  # noqa: E402
from features.candidates import generate_candidates  # noqa: E402
from features.geometry import point_segment  # noqa: E402
from features.labels import assign_route_arcs  # noqa: E402


def main(trip_id="00000001", partition="train", every=5):
    trip = load_trip(trip_id, partition)
    true_arc, _, ldist = assign_route_arcs(trip)
    inv = Transformer.from_crs(f"EPSG:{trip.epsg}", "EPSG:4326", always_xy=True)
    tr = trip.track
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15,
                   tiles="OpenStreetMap")

    # nearby network: only arcs that are candidates (within 100 m) of some fix,
    # otherwise dense urban networks make the HTML huge
    a, b = trip.arc_xy
    pxy = tr[["x", "y"]].to_numpy()
    near = np.unique(generate_candidates(trip, radius=100.0).arc_id.to_numpy())
    g_net = folium.FeatureGroup("road network (arcs within 100 m)", show=True)
    for i in near:
        u, v = trip.arcs[i]
        folium.PolyLine([trip.nodes_ll[u][::-1], trip.nodes_ll[v][::-1]],
                        color="#9a9a9a", weight=1, opacity=0.6).add_to(g_net)
    g_net.add_to(m)

    g_route = folium.FeatureGroup("reference route", show=True)
    for i in trip.route:
        u, v = trip.arcs[i]
        folium.PolyLine([trip.nodes_ll[u][::-1], trip.nodes_ll[v][::-1]],
                        color="green", weight=5, opacity=0.7).add_to(g_route)
    g_route.add_to(m)

    g_gps = folium.FeatureGroup("raw GPS fixes", show=True)
    for lat, lon in zip(tr.lat, tr.lon):
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True,
                            weight=0).add_to(g_gps)
    g_gps.add_to(m)

    g_lab = folium.FeatureGroup(f"label: fix -> assigned arc (every {every}th)", show=True)
    for i in range(0, len(tr), every):
        _, t = point_segment(pxy[i], a[true_arc[i]], b[true_arc[i]])
        q = a[true_arc[i]] + t * (b[true_arc[i]] - a[true_arc[i]])
        qlon, qlat = inv.transform(q[0], q[1])
        col = "red" if ldist[i] <= 100 else "black"
        folium.PolyLine([[tr.lat[i], tr.lon[i]], [qlat, qlon]], color=col, weight=2,
                        tooltip=f"pt {i}: arc {true_arc[i]}, {ldist[i]:.1f} m").add_to(g_lab)
    g_lab.add_to(m)

    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
      padding:8px 12px;border:1px solid #888;font-size:13px">
      <b>Label check: trip %s (%s)</b><br>
      <span style="color:#9a9a9a">&#9644;</span> road network<br>
      <span style="color:green">&#9644;</span> reference route<br>
      <span style="color:blue">&#9679;</span> raw GPS fix<br>
      <span style="color:red">&#9644;</span> fix &rarr; labelled arc (&le;100 m)<br>
      <span style="color:black">&#9644;</span> unreliable label (&gt;100 m)</div>""" % (trip_id, partition)
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    out = ROOT / "results" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / f"label_check_{trip_id}.html")
    print("saved", out / f"label_check_{trip_id}.html")


if __name__ == "__main__":
    main(*sys.argv[1:3])
