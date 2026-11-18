"""
PathSnap Task C: Exploratory Data Analysis & Road Network Topology
Analyzes GPS trajectory quality, time gaps, speed profiles, road network
graph connectivity with NetworkX, and generates interactive Folium maps.
"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import folium

def haversine_m(lon1, lat1, lon2, lat2):
    """Calculate haversine distance in meters."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def run_eda(data_dir, output_dir):
    tables_dir = os.path.join(output_dir, "tables")
    plots_dir = os.path.join(output_dir, "plots")
    maps_dir = os.path.join(output_dir, "maps")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    partitions = ["train", "student_test"]
    trip_inventory = []
    all_time_gaps = []
    all_speeds_kmh = []
    large_gaps_records = []

    print(f"Starting PathSnap Task C Exploratory Data Analysis on {data_dir}...")

    for part in partitions:
        part_dir = os.path.join(data_dir, part)
        if not os.path.exists(part_dir):
            print(f"Warning: {part_dir} not found. Skipping {part}.")
            continue
            
        trips = sorted([d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))])
        print(f"Processing partition: {part} ({len(trips)} trips)...")

        for trip_id in trips:
            trip_path = os.path.join(part_dir, trip_id)
            track_fp = os.path.join(trip_path, f"{trip_id}.track")
            nodes_fp = os.path.join(trip_path, f"{trip_id}.nodes")
            arcs_fp = os.path.join(trip_path, f"{trip_id}.arcs")
            route_fp = os.path.join(trip_path, f"{trip_id}.route")

            track_exists = os.path.exists(track_fp)
            nodes_exists = os.path.exists(nodes_fp)
            arcs_exists = os.path.exists(arcs_fp)
            route_exists = os.path.exists(route_fp)

            track_size = os.path.getsize(track_fp) if track_exists else 0
            nodes_size = os.path.getsize(nodes_fp) if nodes_exists else 0
            arcs_size = os.path.getsize(arcs_fp) if arcs_exists else 0
            route_size = os.path.getsize(route_fp) if route_exists else 0

            track_rows, null_count, time_disorders, dup_timestamps = 0, 0, 0, 0
            min_lon, max_lon = np.nan, np.nan
            min_lat, max_lat = np.nan, np.nan
            duration_s, total_dist_m, max_time_gap, pct_1s_gaps = 0.0, 0.0, 0.0, 0.0

            if track_exists:
                df_track = pd.read_csv(track_fp, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
                track_rows = len(df_track)
                null_count = int(df_track.isnull().sum().sum())
                ts = df_track["timestamp"].values
                dt = np.diff(ts)
                time_disorders = int(np.sum(dt < 0))
                dup_timestamps = int(np.sum(dt == 0))

                if len(dt) > 0:
                    all_time_gaps.extend(dt.tolist())
                    max_time_gap = float(np.max(dt))
                    pct_1s_gaps = float(np.sum(np.isclose(dt, 1.0, atol=1e-3)) / len(dt) * 100.0)

                    for g_idx in np.where(dt > 3.0)[0]:
                        large_gaps_records.append({
                            "partition": part,
                            "trip_id": trip_id,
                            "gap_index": int(g_idx),
                            "gap_seconds": float(dt[g_idx]),
                            "prev_timestamp": float(ts[g_idx]),
                            "next_timestamp": float(ts[g_idx + 1]),
                            "from_lon": float(df_track["lon"].values[g_idx]),
                            "from_lat": float(df_track["lat"].values[g_idx]),
                            "to_lon": float(df_track["lon"].values[g_idx + 1]),
                            "to_lat": float(df_track["lat"].values[g_idx + 1]),
                        })

                    dists = haversine_m(df_track["lon"].values[:-1], df_track["lat"].values[:-1],
                                        df_track["lon"].values[1:], df_track["lat"].values[1:])
                    total_dist_m = float(np.sum(dists))
                    valid_dt = np.where(dt > 0, dt, 1e-6)
                    speeds_km_h = (dists / valid_dt) * 3.6
                    all_speeds_kmh.extend(speeds_km_h.tolist())

                min_lon, max_lon = float(df_track["lon"].min()), float(df_track["lon"].max())
                min_lat, max_lat = float(df_track["lat"].min()), float(df_track["lat"].max())
                duration_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

            nodes_rows = len(pd.read_csv(nodes_fp, sep=r"\s+", header=None)) if nodes_exists else 0
            df_arcs = pd.read_csv(arcs_fp, sep=r"\s+", header=None, names=["from_node", "to_node"]) if arcs_exists else pd.DataFrame()
            arcs_rows = len(df_arcs)

            route_rows = 0
            route_is_continuous = True
            route_invalid_arcs = 0
            if route_exists:
                df_route = pd.read_csv(route_fp, sep=r"\s+", header=None, names=["arc_id"])
                route_rows = len(df_route)
                r_arcs = df_route["arc_id"].values
                route_invalid_arcs = int(np.sum((r_arcs < 0) | (r_arcs >= arcs_rows)))
                r_arcs = r_arcs[(r_arcs >= 0) & (r_arcs < arcs_rows)]
                if arcs_exists and len(r_arcs) > 1:
                    from_nodes = df_arcs.iloc[r_arcs]["from_node"].values
                    to_nodes = df_arcs.iloc[r_arcs]["to_node"].values
                    route_is_continuous = bool(np.sum(to_nodes[:-1] != from_nodes[1:]) == 0)

            trip_inventory.append({
                "partition": part,
                "trip_id": trip_id,
                "track_points": track_rows,
                "track_size_bytes": track_size,
                "nodes_count": nodes_rows,
                "nodes_size_bytes": nodes_size,
                "arcs_count": arcs_rows,
                "arcs_size_bytes": arcs_size,
                "route_arcs": route_rows,
                "route_size_bytes": route_size,
                "duration_s": duration_s,
                "dist_m": total_dist_m,
                "null_count": null_count,
                "time_disorders": time_disorders,
                "dup_timestamps": dup_timestamps,
                "pct_1s_gaps": pct_1s_gaps,
                "max_time_gap_s": max_time_gap,
                "route_is_continuous": route_is_continuous,
                "route_invalid_arcs": route_invalid_arcs,
                "min_lon": min_lon, "max_lon": max_lon,
                "min_lat": min_lat, "max_lat": max_lat
            })

    df_inventory = pd.DataFrame(trip_inventory)
    df_inventory.to_csv(os.path.join(tables_dir, "trip_inventory_detailed.csv"), index=False)
    pd.DataFrame(large_gaps_records).to_csv(os.path.join(tables_dir, "large_gps_gaps.csv"), index=False)

    # Road network analysis on ALL trips (train + student_test)
    print("Analysing road-network topology with NetworkX (all trips)...")
    graph_stats = []
    for part in partitions:
        part_dir = os.path.join(data_dir, part)
        if not os.path.exists(part_dir):
            continue
        for tid in sorted(os.listdir(part_dir)):
            if os.path.isdir(os.path.join(part_dir, tid)):
                n_fp = os.path.join(part_dir, tid, f"{tid}.nodes")
                a_fp = os.path.join(part_dir, tid, f"{tid}.arcs")
                if os.path.exists(n_fp) and os.path.exists(a_fp):
                    dn = pd.read_csv(n_fp, sep=r"\s+", header=None)
                    da = pd.read_csv(a_fp, sep=r"\s+", header=None, names=["u", "v"])
                    G = nx.DiGraph()
                    G.add_nodes_from(range(len(dn)))
                    G.add_edges_from(zip(da["u"], da["v"]))
                    isolates = len(list(nx.isolates(G)))
                    wccs = [len(c) for c in nx.weakly_connected_components(G)]
                    largest_wcc = max(wccs) if wccs else 0
                    sccs = [len(c) for c in nx.strongly_connected_components(G)]
                    largest_scc = max(sccs) if sccs else 0
                    indeg = np.bincount(da["v"].to_numpy(), minlength=len(dn))
                    outdeg = np.bincount(da["u"].to_numpy(), minlength=len(dn))

                    graph_stats.append({
                        "partition": part,
                        "trip_id": tid,
                        "total_nodes": len(dn),
                        "total_arcs": len(da),
                        "isolated_nodes": isolates,
                        "pct_isolated_nodes": (isolates / len(dn)) * 100.0,
                        "source_nodes (in=0, out>0)": int(((indeg == 0) & (outdeg > 0)).sum()),
                        "sink_nodes (out=0, in>0)": int(((outdeg == 0) & (indeg > 0)).sum()),
                        "weakly_connected_components": len(wccs),
                        "largest_wcc_nodes": largest_wcc,
                        "pct_nodes_in_largest_wcc": (largest_wcc / len(dn)) * 100.0,
                        "strongly_connected_components": len(sccs),
                        "largest_scc_nodes": largest_scc,
                        "self_loops": len(list(nx.nodes_with_selfloops(G)))
                    })

    df_graph = pd.DataFrame(graph_stats)
    df_graph.to_csv(os.path.join(tables_dir, "road_network_graph_analysis.csv"), index=False)

    # Partition Inventory Summary
    summary_rows = []
    for part in partitions:
        sub = df_inventory[df_inventory["partition"] == part]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "Partition": part.capitalize(),
            "Trips Count": len(sub),
            "Total GPS Observations": sub["track_points"].sum(),
            "Mean GPS Points/Trip": round(sub["track_points"].mean(), 1),
            "Median GPS Points/Trip": round(sub["track_points"].median(), 1),
            "Min GPS Points": int(sub["track_points"].min()),
            "Max GPS Points": int(sub["track_points"].max()),
            "Total Trajectory Dist (km)": round(sub["dist_m"].sum() / 1000.0, 2),
            "Total Duration (hours)": round(sub["duration_s"].sum() / 3600.0, 2),
            "Total Nodes (sum across trips)": sub["nodes_count"].sum(),
            "Total Arcs (sum across trips)": sub["arcs_count"].sum(),
            "Total Route Arcs": sub["route_arcs"].sum(),
            "Total Missing Values": sub["null_count"].sum(),
            "Total Timestamp Disorders": sub["time_disorders"].sum(),
            "Continuous Routes (%)": round((sub["route_is_continuous"].sum() / len(sub)) * 100.0, 1),
            "Total Disk Size (MB)": round((sub["track_size_bytes"].sum() + sub["nodes_size_bytes"].sum() +
                                           sub["arcs_size_bytes"].sum() + sub["route_size_bytes"].sum()) / (1024*1024), 2)
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(tables_dir, "partition_inventory_summary.csv"), index=False)

    # Plots
    print("Generating EDA plots...")
    gaps_arr = np.array(all_time_gaps)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(gaps_arr[gaps_arr <= 5.0], bins=50, kde=False, color="#1f77b4", edgecolor="black")
    plt.title(r"GPS Sampling Interval Distribution ($\Delta t \leq 5$s)")
    plt.xlabel(r"Sampling Interval $\Delta t$ (seconds)")
    plt.ylabel("Observation Count")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    sns.boxplot(y=gaps_arr, color="#aec7e8")
    plt.title("GPS Time Gaps Boxplot (Including Outliers)")
    plt.ylabel(r"Interval $\Delta t$ (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gps_sampling_interval_distribution.png"), dpi=300)
    plt.close()

    # Trajectory length plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=df_inventory, x="track_points", hue="partition", kde=True, bins=25, ax=axes[0], palette="tab10")
    axes[0].set_title("Distribution of GPS Points per Trip")
    axes[0].set_xlabel("Number of GPS Observations")
    axes[0].set_ylabel("Number of Trips")

    sns.histplot(data=df_inventory, x="dist_m", hue="partition", kde=True, bins=25, ax=axes[1], palette="tab10")
    axes[1].set_title("Distribution of Trajectory Length (Meters)")
    axes[1].set_xlabel("Cumulative GPS Path Length (m)")
    axes[1].set_ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "trajectory_length_and_points.png"), dpi=300)
    plt.close()

    # Speed distribution
    speeds_arr = np.array(all_speeds_kmh)
    speeds_valid = speeds_arr[(speeds_arr >= 0) & (speeds_arr <= 160)]
    plt.figure(figsize=(9, 5))
    sns.histplot(speeds_valid, bins=60, kde=True, color="#2ca02c")
    plt.axvline(x=50, color="orange", linestyle="--", label="Urban limit (50 km/h)")
    plt.axvline(x=100, color="red", linestyle="--", label="Highway limit (100 km/h)")
    plt.title(r"Implied GPS Speed Distribution ($v = \Delta d / \Delta t$)")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Point Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gps_speed_distribution.png"), dpi=300)
    plt.close()

    # Road-network topology summary (all 90 trips)
    g = df_graph.sort_values("pct_nodes_in_largest_wcc").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = g["partition"].map({"train": "#1f77b4", "student_test": "#ff7f0e"})
    axes[0].bar(range(len(g)), g["pct_nodes_in_largest_wcc"], color=colors)
    axes[0].set_ylim(max(0, g["pct_nodes_in_largest_wcc"].min() - 5), 100.5)
    axes[0].set_title("Share of nodes in the largest weakly connected component")
    axes[0].set_xlabel("Trip (sorted; blue = train, orange = student_test)")
    axes[0].set_ylabel("% of nodes in largest WCC")
    axes[0].set_xticks([])
    axes[1].scatter(g["total_nodes"], g["weakly_connected_components"], c=colors, s=25)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Network size vs number of weakly connected components")
    axes[1].set_xlabel("Nodes |V| (log)")
    axes[1].set_ylabel("Weakly connected components (log)")
    fig.suptitle(f"Road-network topology, {len(g)} trips. Isolated nodes: "
                 f"{int(df_graph['isolated_nodes'].sum())} in total; self-loops: "
                 f"{int(df_graph['self_loops'].sum())}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "road_network_topology_summary.png"), dpi=200)
    plt.close()

    # Static GPS trajectory + road network plot (for the PDF report)
    plot_trajectory_static(data_dir, "train", "00000001",
                           os.path.join(plots_dir, "trajectory_and_network_00000001.png"))

    # Folium maps for two training trips
    for tid in ["00000001", "00000002"]:
        generate_folium_map(data_dir, "train", tid, os.path.join(maps_dir, f"trip_{tid}_map.html"))
    print("EDA execution completed successfully.")


def plot_trajectory_static(data_dir, partition, trip_id, out_fp, margin=0.003):
    """Matplotlib view of raw GPS, nearby road network and reference route (lon/lat)."""
    from matplotlib.collections import LineCollection
    trip_path = os.path.join(data_dir, partition, trip_id)
    rd = lambda ext, names: pd.read_csv(os.path.join(trip_path, f"{trip_id}.{ext}"), sep=r"\s+",
                                        header=None, names=names)
    tr = rd("track", ["lon", "lat", "t"])
    nd = rd("nodes", ["lon", "lat"]).to_numpy()
    ar = rd("arcs", ["u", "v"]).to_numpy()
    ro = rd("route", ["arc"]).to_numpy().ravel()
    lo_x, hi_x = tr.lon.min() - margin, tr.lon.max() + margin
    lo_y, hi_y = tr.lat.min() - margin, tr.lat.max() + margin
    a, b = nd[ar[:, 0]], nd[ar[:, 1]]
    keep = ((np.maximum(a[:, 0], b[:, 0]) >= lo_x) & (np.minimum(a[:, 0], b[:, 0]) <= hi_x) &
            (np.maximum(a[:, 1], b[:, 1]) >= lo_y) & (np.minimum(a[:, 1], b[:, 1]) <= hi_y))
    lat0 = tr.lat.mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1.6]})
    # right panel: ~600 m window around the middle of the trip, to show GPS noise vs the route
    mid = tr.iloc[len(tr) // 2]
    half_lat = 300 / 111_000.0
    half_lon = half_lat / np.cos(np.radians(lat0))
    views = [("Whole trip", lo_x, hi_x, lo_y, hi_y, 2, 0.3),
             ("Zoom: ~600 m window at mid-trip", mid.lon - half_lon, mid.lon + half_lon,
              mid.lat - half_lat, mid.lat + half_lat, 18, 0.8)]
    for ax, (title, x0, x1, y0, y1, ms, lw) in zip(axes, views):
        ax.add_collection(LineCollection(np.stack([a[keep], b[keep]], 1), colors="#9a9a9a",
                                         linewidths=lw, label="road network (arcs)"))
        ax.add_collection(LineCollection(np.stack([nd[ar[ro, 0]], nd[ar[ro, 1]]], 1), colors="#2ca02c",
                                         linewidths=4, alpha=0.8, label="reference route"))
        ax.scatter(tr.lon, tr.lat, s=ms, c="#1f77b4", edgecolors="white", linewidths=0.3,
                   label="raw GPS fixes", zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1.0 / np.cos(np.radians(lat0)))  # equal metres in x and y
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(title)
    axes[0].add_patch(plt.Rectangle((mid.lon - half_lon, mid.lat - half_lat), 2 * half_lon, 2 * half_lat,
                                    fill=False, ec="red", lw=1.5))
    axes[1].legend(loc="lower right", framealpha=0.95)
    fig.suptitle(f"Trip {trip_id} ({partition}): raw GPS trajectory, road network and reference route")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=200)
    plt.close()

def generate_folium_map(data_dir, partition, trip_id, out_fp):
    trip_path = os.path.join(data_dir, partition, trip_id)
    track_fp = os.path.join(trip_path, f"{trip_id}.track")
    nodes_fp = os.path.join(trip_path, f"{trip_id}.nodes")
    arcs_fp = os.path.join(trip_path, f"{trip_id}.arcs")
    route_fp = os.path.join(trip_path, f"{trip_id}.route")

    if not (os.path.exists(track_fp) and os.path.exists(nodes_fp)):
        return

    df_track = pd.read_csv(track_fp, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    df_nodes = pd.read_csv(nodes_fp, sep=r"\s+", header=None, names=["lon", "lat"])
    df_arcs = pd.read_csv(arcs_fp, sep=r"\s+", header=None, names=["from_node", "to_node"]) if os.path.exists(arcs_fp) else pd.DataFrame()
    df_route = pd.read_csv(route_fp, sep=r"\s+", header=None, names=["arc_id"]) if os.path.exists(route_fp) else pd.DataFrame()

    m = folium.Map(location=[df_track["lat"].mean(), df_track["lon"].mean()], zoom_start=15, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Road Network
    node_coords = df_nodes[["lat", "lon"]].values
    margin = 0.005
    mask = (
        (df_nodes["lon"].values >= df_track["lon"].min() - margin) & (df_nodes["lon"].values <= df_track["lon"].max() + margin) &
        (df_nodes["lat"].values >= df_track["lat"].min() - margin) & (df_nodes["lat"].values <= df_track["lat"].max() + margin)
    )
    valid_nodes = set(np.where(mask)[0])

    arc_lines = []
    if len(df_arcs) > 0:
        for u, v in zip(df_arcs["from_node"].values, df_arcs["to_node"].values):
            if u in valid_nodes or v in valid_nodes:
                arc_lines.append([[node_coords[u][0], node_coords[u][1]], [node_coords[v][0], node_coords[v][1]]])
                if len(arc_lines) >= 3000:
                    break

    fg_network = folium.FeatureGroup(name="Road Network (Arcs)", show=True)
    folium.PolyLine(locations=arc_lines, color="#7f7f7f", weight=2, opacity=0.45).add_to(fg_network)
    fg_network.add_to(m)

    # Route
    if len(df_route) > 0 and len(df_arcs) > 0:
        route_lines = []
        for a in df_route["arc_id"].values:
            if 0 <= a < len(df_arcs):
                u = df_arcs.iloc[a]["from_node"]
                v = df_arcs.iloc[a]["to_node"]
                route_lines.append([[node_coords[u][0], node_coords[u][1]], [node_coords[v][0], node_coords[v][1]]])
        fg_route = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        folium.PolyLine(locations=route_lines, color="#2ca02c", weight=5, opacity=0.85).add_to(fg_route)
        fg_route.add_to(m)

    # GPS
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    folium.PolyLine(locations=df_track[["lat", "lon"]].values.tolist(), color="#1f77b4", weight=2, dash_array="5, 5").add_to(fg_gps)
    for idx, row in df_track.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3.5, color="#0b559f", fill=True, fill_color="#3182bd", fill_opacity=0.9, weight=1,
            tooltip=f"Point #{idx} | t={row['timestamp']:.1f}s"
        ).add_to(fg_gps)

    folium.Marker([df_track.iloc[0]["lat"], df_track.iloc[0]["lon"]], popup=f"Trip {trip_id} Start", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fg_gps)
    folium.Marker([df_track.iloc[-1]["lat"], df_track.iloc[-1]["lon"]], popup=f"Trip {trip_id} End", icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(fg_gps)
    fg_gps.add_to(m)

    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
      padding:8px 12px;border:1px solid #888;border-radius:4px;font-size:13px">
      <b>Trip {trip_id} ({partition})</b><br>
      <span style="color:#7f7f7f">&#9644;</span> road network (arcs)<br>
      <span style="color:#2ca02c">&#9644;</span> reference route<br>
      <span style="color:#3182bd">&#9679;</span> raw GPS fix (dashed line = GPS sequence)<br>
      <span style="color:green">&#9654;</span> start &nbsp; <span style="color:red">&#9632;</span> end</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    m.fit_bounds([[df_track["lat"].min(), df_track["lon"].min()],
                  [df_track["lat"].max(), df_track["lon"].max()]])
    m.save(out_fp)
    print(f"Saved interactive Folium map to {out_fp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap Task C EDA")
    default_data = os.environ.get("PATHSNAP_DATA", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Student folder", "Student folder"))
    parser.add_argument("--data_dir", type=str, default=default_data)
    parser.add_argument("--output_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    run_eda(args.data_dir, args.output_dir)
