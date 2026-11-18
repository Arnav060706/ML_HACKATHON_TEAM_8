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
                            "gap_seconds": float(dt[g_idx]),
                            "prev_timestamp": float(ts[g_idx]),
                            "next_timestamp": float(ts[g_idx + 1])
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
            if route_exists:
                df_route = pd.read_csv(route_fp, sep=r"\s+", header=None, names=["arc_id"])
                route_rows = len(df_route)
                r_arcs = df_route["arc_id"].values
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
                "min_lon": min_lon, "max_lon": max_lon,
                "min_lat": min_lat, "max_lat": max_lat
            })

    df_inventory = pd.DataFrame(trip_inventory)
    df_inventory.to_csv(os.path.join(tables_dir, "trip_inventory_detailed.csv"), index=False)
    pd.DataFrame(large_gaps_records).to_csv(os.path.join(tables_dir, "large_gps_gaps.csv"), index=False)

    # Road network analysis on sample trips
    sample_trips = ["00000001", "00000002", "00000005", "00000010", "00000025", "00000000", "00000007"]
    graph_stats = []
    for part in partitions:
        part_dir = os.path.join(data_dir, part)
        if not os.path.exists(part_dir):
            continue
        for tid in os.listdir(part_dir):
            if tid in sample_trips:
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

                    graph_stats.append({
                        "partition": part,
                        "trip_id": tid,
                        "total_nodes": len(dn),
                        "total_arcs": len(da),
                        "isolated_nodes": isolates,
                        "pct_isolated_nodes": (isolates / len(dn)) * 100.0,
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
            "Total Trajectory Dist (km)": round(sub["dist_m"].sum() / 1000.0, 2),
            "Total Duration (hours)": round(sub["duration_s"].sum() / 3600.0, 2),
            "Total Nodes": sub["nodes_count"].sum(),
            "Total Arcs": sub["arcs_count"].sum(),
            "Total Route Arcs": sub["route_arcs"].sum(),
            "Total Missing Values": sub["null_count"].sum(),
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

    # Folium Map generation for Trip 00000001
    generate_folium_map(data_dir, "train", "00000001", os.path.join(maps_dir, "trip_00000001_map.html"))
    print("EDA execution completed successfully.")

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
    folium.TileLayer("CartoDB Positron", name="CartoDB Positron (Light)").add_to(m)

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

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    m.save(out_fp)
    print(f"Saved interactive Folium map to {out_fp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap Task C EDA")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Arnav Sharma\.gemini\antigravity\scratch\pathsnap\data")
    parser.add_argument("--output_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    run_eda(args.data_dir, args.output_dir)
