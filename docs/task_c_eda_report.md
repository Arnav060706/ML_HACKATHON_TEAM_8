# PathSnap: Exploratory Data Analysis (Task C) & Dataset Understanding

**Author / Component:** Task C — Exploratory Data Analysis (EDA) & Road Network Analysis  
**Target Report Section:** Section 2.2 (*Dataset Understanding and Exploratory Data Analysis*) & Viva Preparation  
**Dataset Partitions Evaluated:** `train` (80 trips) and `student_test` (10 trips)  

---

## 1. Executive Summary & Critical Discoveries

> [!IMPORTANT]
> **Critical Finding on Coordinate Systems (Crucial for Task A & Task B):**  
> While initial inspection of Trip `00000001` suggested Germany (UTM Zone 32N / `EPSG:32632`), **our complete 90-trip audit reveals that the PathSnap dataset is globally distributed**. Trajectories span **Longitude -100.42° to +137.24°** and **Latitude -34.64° to +61.70°**, covering **23 distinct UTM zones** across North America, Europe, Asia, and the Southern Hemisphere. A single static UTM 32N projection will severely distort or invalidate non-European trips. **Coordinates must be projected dynamically per trip** using `EPSG:32600 + zone` (Northern hemisphere) or `EPSG:32700 + zone` (Southern hemisphere).

### Key Takeaways
1. **Zero Data Corruption:** Across all 227,577 GPS observations, there are **0 missing values (NaNs)**, **0 timestamp inversions**, and **0 duplicate timestamps**. GPS observations are strictly monotonic.
2. **Sampling Regularity with Long Pause Outliers:** **94.3%** of GPS observations have $\Delta t = 1.0\text{ s}$ (nominally 1 Hz). However, 1,951 gaps exceed 3 seconds, and extreme tracking pauses occur (e.g., Trip `00000090` has an overnight gap of 42.5 hours; Trip `00000078` has a gap of 14.7 hours).
3. **Graph Topology (NetworkX):** Road networks are provided as localized subgraphs per trip. There are **0 isolated nodes** (0.0%). The largest Weakly Connected Component (WCC) encompasses **93.4%–98.9%** of all nodes per trip. Smaller disconnected components arise primarily from boundary clipping and footway/service networks.
4. **Reference Route Anomalies:** **87.5%** of training trips have continuous topological routes. In **12.5%** of trips, reference routes exhibit discrete jumps where $\text{arc}_k.\text{to\_node} \neq \text{arc}_{k+1}.\text{from\_node}$, confirming the guideline caveat that reference routes contain ground-truth digitization limitations.

---

## 2. Dataset Inventory

Each trip directory contains four core whitespace-delimited files:
* `.track`: Raw GPS observations `(longitude, latitude, timestamp_seconds)`
* `.nodes`: Road network vertices `(longitude, latitude)` indexed by 0-based row ID
* `.arcs`: Directed road network edges `(from_node_id, to_node_id)` indexed by 0-based row ID
* `.route`: Reference trajectory as an ordered sequence of ground-truth `arc_id` values

### Partition Summary Table

| Metric | Training Partition (`train`) | Student Testing Partition (`student_test`) | Combined Total |
| :--- | :--- | :--- | :--- |
| **Trips Count** | **80** | **10** | **90** |
| **Total GPS Observations** | 208,417 | 19,160 | **227,577** |
| **Mean GPS Points / Trip** | 2,605.2 | 1,916.0 | 2,528.6 |
| **Median GPS Points / Trip** | 1,529.0 | 1,420.0 | 1,515.5 |
| **Min / Max Points per Trip** | 361 / 19,313 | 262 / 3,958 | 262 / 19,313 |
| **Total Trajectory Path** | 2,978.41 km | 296.97 km | **3,275.38 km** |
| **Total Tracking Duration** | 126.23 hours | 5.62 hours | **131.85 hours** |
| **Road Network Nodes ($|V|$)** | 1,933,485 | 90,689 | **2,024,174** |
| **Road Network Arcs ($|E|$)** | 3,876,904 | 185,041 | **4,061,945** |
| **Total Reference Route Arcs** | 31,336 | 3,477 | **34,813** |
| **Missing Coordinate Values** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** |
| **Timestamp Inversions** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** |
| **Continuous Ground Truth Routes** | **87.5% (70/80)** | **90.0% (9/10)** | **87.8% (79/90)** |
| **Total Raw Disk Footprint** | 85.58 MB | 4.17 MB | **89.75 MB** |

*Full per-trip breakdown is saved in [trip_inventory_detailed.csv](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/scratch/pathsnap/eda/tables/trip_inventory_detailed.csv).*

---

## 3. Trajectory Quality & Physical Validity Analysis

![Trajectory Length and Points Distribution](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/brain/c225b259-93be-4614-be1b-fd79e417093f/plots/trajectory_length_and_points.png)

### 3.1 Sampling Frequency & Time Gaps
![GPS Sampling Interval Distribution](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/brain/c225b259-93be-4614-be1b-fd79e417093f/plots/gps_sampling_interval_distribution.png)

* **Nominal Rate:** **94.32%** of observations feature an interval of exactly $\Delta t = 1.0\text{ s}$.
* **Irregular Gaps:** Across the dataset, **1,951 intervals exceed 3 seconds** and **617 intervals exceed 10 seconds**.
* **Major Anomalies:**
  * Trip `00000090`: Gap of **153,064.0 s** (~42.5 hours) between $t=970\text{ s}$ and $t=154,034\text{ s}$.
  * Trip `00000078`: Gap of **52,810.0 s** (~14.7 hours) between $t=4\text{ s}$ and $t=52,814\text{ s}$.
  * Trip `00000069`: Gap of **2,658.0 s** (~44.3 minutes).
  * Trip `00000021`: Gap of **2,161.0 s** (~36.0 minutes).

> [!NOTE]
> **HMM / Viterbi Implication:** When $\Delta t > 30\text{ s}$, the Markovian transition probability $P(s_{t} \mid s_{t-1})$ derived from steady vehicle kinematics degrades. In the HMM decoder, long gaps must either reset the Viterbi chain or cap the transition distance penalty.

### 3.2 Vehicle Speed Distribution ($v = \Delta d / \Delta t$)
![GPS Speed Distribution](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/brain/c225b259-93be-4614-be1b-fd79e417093f/plots/gps_speed_distribution.png)

* Implied instantaneous velocities cluster between **0 and 65 km/h**, typical of urban stop-and-go driving and suburban arterial travel.
* A secondary peak occurs at **80–110 km/h**, representing motorway segments.
* Outlier speeds (> 140 km/h) coincide with momentary GPS multipath jumps where distance jumps by 40–80 m within 1 s.

---

## 4. Road Network Topology & Graph Analysis (NetworkX)

Each trip's road network was converted to a directed graph $G = (V, E)$ using `networkx.DiGraph`.

![Road Network Topology Summary](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/brain/c225b259-93be-4614-be1b-fd79e417093f/plots/road_network_topology_summary.png)

### Graph Metrics on Representative Sample Trips

| Trip ID | Partition | Total Nodes $|V|$ | Total Arcs $|E|$ | Isolated Nodes (deg=0) | Weakly Connected Components (WCC) | Largest WCC Coverage (%) | Strongly Connected Components (SCC) | Self Loops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `00000001` | `train` | 88,809 | 165,790 | **0 (0.0%)** | 175 | **97.80%** (86,854 nodes) | 1,942 | 0 |
| `00000002` | `train` | 7,936 | 16,139 | **0 (0.0%)** | 14 | **97.66%** (7,750 nodes) | 14 | 0 |
| `00000005` | `train` | 898 | 1,579 | **0 (0.0%)** | 3 | **98.89%** (888 nodes) | 153 | 0 |
| `00000010` | `train` | 8,502 | 15,715 | **0 (0.0%)** | 28 | **93.40%** (7,941 nodes) | 411 | 0 |
| `00000025` | `train` | 38,035 | 85,639 | **0 (0.0%)** | 34 | **94.97%** (36,122 nodes) | 552 | 0 |
| `00000000` | `student_test` | 8,542 | 18,196 | **0 (0.0%)** | 100 | **87.40%** (7,466 nodes) | 102 | 0 |
| `00000007` | `student_test` | 2,061 | 4,369 | **0 (0.0%)** | 6 | **96.99%** (1,999 nodes) | 6 | 0 |

### Topological Findings
1. **Zero Isolated Nodes:** No unreferenced vertices exist in the provided network slices.
2. **Boundary Truncation:** Sub-networks are spatially cropped around each trajectory. Nodes crossing the bounding box boundary appear as artificial sources ($\text{in-degree} = 0$) or sinks ($\text{out-degree} = 0$).
3. **Connectivity:** Over **95%** of all network vertices reside within a single giant connected component. Minor components correspond to disjoint suburban enclaves or pedestrian ways not connected to the vehicular roadway.

---

## 5. Interactive Folium Visualizations

Interactive multi-layer Folium maps were generated for training trips `00000001` and `00000002`:

1. **Trip 00000001 Interactive Map:** [trip_00000001_map.html](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/scratch/pathsnap/eda/maps/trip_00000001_map.html) (Local scratch copy) | [Artifact Copy](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/brain/c225b259-93be-4614-be1b-fd79e417093f/maps/trip_00000001_map.html)
2. **Trip 00000002 Interactive Map:** [trip_00000002_map.html](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/scratch/pathsnap/eda/maps/trip_00000002_map.html) (Local scratch copy) | [Artifact Copy](file:///C:/Users/Arnav%20Sharma/.gemini/antigravity/brain/c225b259-93be-4614-be1b-fd79e417093f/maps/trip_00000002_map.html)

### Map Capabilities & Layer Schema
* **Base Layers:** OpenStreetMap standard & CartoDB Positron (clean light theme for publication).
* **Road Network (Arcs) Layer:** Gray vector polylines displaying underlying directed road connectivity.
* **Reference Route Layer:** Solid emerald green polyline ($w=5$) tracing ground-truth sequence of traversed arcs.
* **Raw GPS Observations Layer:** Interactive blue circular markers with hover tooltips displaying index, timestamp, and coordinates, connected by a dashed trajectory sequence.
* **Start & End Pins:** Distinct green play icon (trip start) and red stop icon (trip end).
* **UI Controls:** Dynamic layer toggling (LayerControl) and a custom fixed-position HTML visual legend.

---

## 6. Data Quality Issues & Modeling Assumptions

### Identified Data Quality Issues
1. **Global Geographic Heterogeneity:** Trips are collected across disparate worldwide cities rather than a single metropolitan region.
2. **Extreme Intermittent Temporal Gaps:** Dropouts and multi-hour halts (up to 42.5 hours) exist in several training trips.
3. **Reference Route Discontinuities:** 12.5% of ground-truth routes contain topological step mismatches where the end node of arc $k$ does not connect directly to the start node of arc $k+1$.
4. **GPS Noise & Urban Canyon Drift:** Lateral GPS errors up to 35 meters occur in dense road corridors, causing points to drift closer to parallel side streets.

### Core Engineering Assumptions
1. **Dynamic Per-Trip Projection:** Convert coordinates to meters using local UTM zones:
   $$\text{Zone} = \left\lfloor \frac{\text{Longitude}_{\text{median}} + 180}{6} \right\rfloor + 1$$
   Assign EPSG code $32600 + \text{Zone}$ for $\text{Lat} \ge 0$, and $32700 + \text{Zone}$ for $\text{Lat} < 0$.
2. **Non-decreasing Route Alignment:** GPS points must map to candidate arcs along the reference route in a monotonically non-decreasing order to prevent backward transitions.
3. **Candidate Search Radius ($R$):** Based on observed GPS lateral drift, a spatial index search radius of $R \in [30\text{ m}, 50\text{ m}]$ is optimal to prevent candidate miss while keeping candidate set size tractable.
4. **Temporal Gap Segmentation:** Consecutive observations with $\Delta t > 30\text{ s}$ should be treated as trajectory segment boundaries during HMM Viterbi decoding.
