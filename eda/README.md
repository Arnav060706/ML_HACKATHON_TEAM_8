# Exploratory Data Analysis & Road Network Topology (Task C)

This folder contains the complete data inventory, quality checks, network topology analysis using NetworkX, statistical summary tables, visualization plots, and interactive Folium maps for the **PathSnap** hackathon.

## 1. Directory Structure

```
eda/
├── README.md                          # Task C documentation and summary
├── run_eda.py                         # Standalone reproducible EDA pipeline
├── tables/
│   ├── partition_inventory_summary.csv# Partition-level inventory (80 train, 10 test)
│   ├── trip_inventory_detailed.csv    # Per-trip metrics across all 90 trips
│   ├── large_gps_gaps.csv             # Details on all 1,951 GPS intervals > 3s
│   └── road_network_graph_analysis.csv# NetworkX topology metrics for all 90 trip networks
├── plots/
│   ├── gps_sampling_interval_distribution.png  # Histogram & boxplot of sampling rate
│   ├── trajectory_length_and_points.png        # Points & trajectory distance distributions
│   ├── gps_speed_distribution.png              # Implied vehicle speed distribution
│   ├── road_network_topology_summary.png       # Largest-WCC share and component counts (90 trips)
│   └── trajectory_and_network_00000001.png     # Static GPS + road network + route plot
└── maps/
    ├── trip_00000001_map.html         # Interactive Folium map for Trip 00000001
    └── trip_00000002_map.html         # Interactive Folium map for Trip 00000002
```

## 2. Key Insights & Empirical Findings

### A. Coordinate System & Projection (Critical finding for Task A & Task B)
* **Initial Assumption:** Training trip `00000001` suggested Germany (UTM Zone 32N / `EPSG:32632`).
* **Full Audit Ground Truth:** The 90 trips in `train` and `student_test` are **globally distributed**, ranging across **25 UTM zones** from North America (e.g. Trip 8 in Alabama, UTM 16N), Europe, Russia (e.g. Trip 3 in Moscow, UTM 37N), to the Southern Hemisphere.
* **Impact:** Applying a static UTM 32N projection across all trips introduces severe metric distortion for non-European routes. Each trip **must** be dynamically projected into its local UTM zone (`EPSG:32600 + zone` for North, `EPSG:32700 + zone` for South).

### B. Trajectory Quality & GPS Sampling
* **Completeness:** **0 missing values (NaNs)**, **0 timestamp inversions**, and **0 duplicate timestamps** across all 227,577 observations.
* **Sampling Rate:** **94.32%** of observations have $\Delta t = 1.0\text{ s}$ (nominal 1 Hz).
* **Pause Gaps:** 1,951 intervals exceed 3 s; 617 exceed 10 s; 70 exceed 60 s.
  * Trip `00000090` contains a 42.5-hour overnight gap ($153,064\text{ s}$).
  * Trip `00000078` contains a 14.7-hour gap ($52,810\text{ s}$).
  * **HMM Strategy:** Long gaps ($\Delta t > 30\text{ s}$) must segment the trajectory to avoid corrupting transition probabilities.

### C. Road Network Topology (NetworkX, all 90 trip networks)
* **Isolated Vertices:** **0 isolated nodes** ($0.0\%$) across all 90 road networks.
* **Graph Connectivity:** The giant Weakly Connected Component (WCC) contains a median **98.0%** of nodes per trip (range 69.3% for trip `00000095` to 100%). Smaller disjoint components result from bounding-box spatial cropping and separate pedestrian/service paths.
* **Self-Loops:** **12 self-loops** ($u \to u$) in total across the 90 networks.

### D. Reference Route Limitations
* **Route Discontinuities:** In **12.5%** of training trips (e.g. Trips 2, 19, 33, 41, 42, 69, 75, 80, 86, 89), reference routes contain topological gaps where $\text{arc}_k.\text{to\_node} \neq \text{arc}_{k+1}.\text{from\_node}$.
* Ground truth routes should be treated as reference routes with measurement noise rather than absolute physical perfection.

## 3. How to Run

To regenerate all tables, plots, and Folium maps:

```bash
python eda/run_eda.py            # data from ../Student folder/Student folder or $PATHSNAP_DATA
python eda/run_eda.py --data_dir /path/to/raw_dataset --output_dir eda/
```
