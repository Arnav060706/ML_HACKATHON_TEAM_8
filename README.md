# PathSnap: Distributed Map-Matching & Route Reconstruction

**Team 8 — PES University | UE24CS352A: Machine Learning**

PathSnap is an end-to-end distributed machine learning system designed to reconstruct realistic vehicle trajectories by matching noisy, sparse GPS observations to real-world road networks. The system combines local UTM projection, KD-tree candidate arc generation, supervised machine learning classifiers, and Hidden Markov Model (HMM) Viterbi decoding served across a two-laptop distributed architecture.

> **Status: complete.** EDA, preprocessing, candidates, labels, features, five model families and model selection (Evaluation 1); Bayesian (Optuna TPE) tuning of the Random Forest; HMM/Viterbi route reconstruction; FastAPI server with server-side evaluation; Leaflet evaluator UI.
>
> **Deployed pipeline:** tuned RF (`models/saved/rf_tuned.joblib`) + HMM/Viterbi (β = 5 m, gap 60 s, `inference/config.json`). Student-test results: Top-1 **0.933** (classifier alone 0.864), route continuity **0.982**, route F1 **0.973** (`results/hmm_vs_classifier.csv`).
>
> **Run the demo:** see `api/README.md` (Laptop 1) and `ui/README.md` (Laptop 2). Quick start, from the repo root:
> `set PATHSNAP_TRIPS_PARTITION=evaluator_test` then `uvicorn api.app:app --host 0.0.0.0 --port 8000`; open `ui/index.html` on Laptop 2 and enter `http://<laptop-1-ip>:8000`.
>
> **Where things are:** HMM and inference in `inference/`; evaluation, β tuning and comparisons in `evaluation/`; tuning in `models/tune_rf.py`; notes in `docs/` (`hmm_note.md`, `hmm_checklist.md`, `partition_usage.md`).

---

## 1. System Architecture & Repository Structure

The repository is organized into modular components adhering strictly to the hackathon guidelines:

```
ML_HACKATHON_TEAM_8/
├── api/                               # Backend FastAPI application (Laptop 1)
│   ├── app.py                         # API endpoints (/health, /trips, /predict)
│   └── README.md                      # API deployment instructions
├── data/
│   └── processed/                     # Engineered candidate and point tables (Parquet)
├── data_processing/                   # Raw trip loader and spatial projection pipeline
│   ├── load_data.py                   # Per-trip UTM projection and file parsers
│   └── preprocessing_report.py        # Dataset validation and geometry verification
├── eda/                               # Exploratory Data Analysis & Graph Topology (Task C)
│   ├── run_eda.py                     # Standalone reproducible EDA pipeline
│   ├── README.md                      # EDA findings, audit numbers and assumptions
│   ├── maps/                          # Interactive Folium maps with layer controls
│   │   ├── trip_00000001_map.html     # Dense urban trip visualization
│   │   └── trip_00000002_map.html     # Suburban trip visualization
│   ├── plots/                         # Publication-quality distribution plots
│   │   ├── gps_sampling_interval_distribution.png
│   │   ├── gps_speed_distribution.png
│   │   ├── road_network_topology_summary.png
│   │   ├── trajectory_and_network_00000001.png  # GPS + road network + route (static)
│   │   └── trajectory_length_and_points.png
│   └── tables/                        # Detailed statistical CSV reports
│       ├── partition_inventory_summary.csv
│       ├── trip_inventory_detailed.csv
│       ├── large_gps_gaps.csv
│       └── road_network_graph_analysis.csv  # NetworkX topology of all 90 trip networks
├── features/                          # Spatial indexing and feature engineering (Task A)
│   ├── candidates.py                  # Densified KD-tree arc candidate generation
│   ├── geometry.py                    # UTM conversions, segment projections, headings
│   ├── labels.py                      # Monotone DP alignment of points to reference route
│   ├── build_features.py              # 17-feature extraction pipeline
│   ├── miss_rate_sweep.py             # Sweep over radius R and top-k
│   ├── label_check_map.py             # Visual label validation maps
│   └── validate_tables.py             # Table schema and integrity validation
├── models/                            # Model training and comparison (Task B)
│   ├── run_experiments.py             # GroupKFold harness: 5 model families + rule baselines
│   ├── select_model.py                # Evidence-based model selection + confusion matrices
│   └── README.md                      # Protocol, models, metrics and selection
├── results/                           # Experiment outputs, tables, and metric summaries
│   ├── baseline_top1.csv              # Distance and heading heuristic baselines
│   ├── candidate_miss_rate.csv        # Candidate recall sweep across (R, k)
│   ├── candidate_miss_rate.png        # Miss rate curve across search radii
│   ├── feature_build_summary.json     # Feature extraction metadata
│   ├── feature_label_means.csv        # Feature contrast between true and false arcs
│   ├── label_alignment_stats.csv      # Route coverage and alignment statistics
│   ├── preprocessing_report.csv       # Per-trip UTM zones, time gaps, and route breaks
│   ├── model_comparison.csv           # All metrics: CV mean + student-test, per model
│   ├── cv_per_fold.csv                # Per-fold CV metrics (stability)
│   ├── configs.json                   # Model configs, seed, features, protocol
│   ├── model_selection.md / .csv      # Ranked selection evidence and decision
│   ├── confusion_matrices.png         # Student-test confusion matrices
│   ├── mlp_loss_curve.png             # MLP training loss (only model with epochs)
│   ├── *_comparison.png               # CV macro-F1, test Top-1/macro-F1, log loss plots
│   ├── section_3_model_comparison_table.md  # Report Section 3.1 table
│   └── maps/                          # Ground truth alignment verification maps
├── ui/                                # Evaluator user interface (Laptop 2)
│   ├── index.html                     # Standalone browser interface for route testing
│   └── README.md                      # UI setup and evaluation instructions
├── requirements.txt                   # Project dependencies
└── README.md                          # Master documentation
```

---

## 2. Key Engineering & EDA Discoveries

### Dynamic Multi-UTM Projection
* **Empirical Audit:** Across the 90 trips (`train` + `student_test`), coordinates span **Longitude -100.42° to +137.24°** and **Latitude -34.64° to +61.70°**, covering **25 distinct UTM zones** across Europe, North America, Russia, and the Southern Hemisphere.
* **Solution:** Trajectories are projected dynamically into local UTM metric planes using `EPSG:32600 + zone` (North) and `EPSG:32700 + zone` (South), agreeing with Haversine distances to within $0.55\%$.

### Candidate Generation & Parameter Selection ($R, k$)
* **Densified KD-tree:** Road segments are densified every 10 meters before KD-tree indexing, followed by exact point-to-segment Euclidean distance calculation.
* **Ties Preservation:** Top-$k$ candidate selection preserves ties to avoid discarding parallel arcs of opposing traffic directions on two-way streets.
* **Optimal Radius:** A sweep on the 80 training trips demonstrated that **$R = 100\text{ m}, k = 8$** achieves a candidate miss rate of **$1.37\%$** (averaging 6.7 candidates per point), outperforming $R = 50\text{ m}$ ($2.00\%$ miss rate).

### Road Network Topology (NetworkX)
* Analysed on **all 90 trip networks** (train + student_test): **0 isolated nodes** and **12 self-loops** in total.
* The largest Weakly Connected Component (WCC) holds a median **98.0%** of each network's nodes (range 69.3% for trip 95 to 100%). Minor disjoint components correspond to bounding-box spatial cropping or isolated pedestrian ways.

### Trajectory Quality & Tracking Gaps
* **Data Integrity:** **0 missing values (NaNs)**, **0 timestamp inversions**, and **0 duplicate timestamps** across 227,577 observations.
* **Sampling Rate:** Nominally 1 Hz (**94.32%** of steps are exactly $1.0\text{ s}$).
* **Pause Handling:** Tracking pauses up to 42.5 hours exist in several trips. They are exposed to the classifier through the `dt_prev` / `dt_next` features. The planned HMM decoder will split the Viterbi chain at large gaps (threshold to be tuned on train/student_test).

### Model Training & Selection (Task B)
* Candidate-level binary classification, **GroupKFold(5) by `trip_id`** on the 80 training trips, refit on all 80, evaluated on the 10 student-test trips. Seed 42; scalers inside pipelines; fitted on `label_reliable == 1` rows.
* Five algorithm families: Logistic Regression, KNN, Random Forest, HistGradientBoosting, MLP (+ three untrained rule baselines).
* **Selected model: Random Forest** (`results/model_selection.md`). It is best on CV Top-1 (0.8365), CV log loss (0.1081, best in 5/5 folds) and CV macro-F1 (0.9073 ± 0.017). On student-test: Top-1 0.8636, macro-F1 0.9180, log loss 0.0966.

---

## 3. Installation & Quick Start

### Prerequisites
Python 3.10+ recommended.

```bash
git clone https://github.com/Arnav060706/ML_HACKATHON_TEAM_8.git
cd ML_HACKATHON_TEAM_8
pip install -r requirements.txt
```

### Reproducing Exploratory Data Analysis
```bash
python eda/run_eda.py --data_dir /path/to/raw_dataset
```

The raw data is not committed. Scripts look for it at `../Student folder/Student folder` (next to the repo) or at the path in the `PATHSNAP_DATA` environment variable.

### Preprocessing, Candidate Generation & Feature Engineering
```bash
python data_processing/preprocessing_report.py   # per-trip QA, UTM zones, gaps
python features/miss_rate_sweep.py               # choose R and k on train trips
python features/build_features.py                # -> data/processed/*.parquet
python features/validate_tables.py               # sanity checks, rule baselines, plot
```

### Model Training & Selection
```bash
python models/run_experiments.py                 # 5 models, GroupKFold CV + student-test (~20+ min)
python results/plot_model_comparison.py          # comparison plots + Section 3.1 table
python models/select_model.py                    # model_selection.md + confusion matrices
```

See `docs/partition_usage.md` for which dataset partition every analysis uses.

---

## 4. Two-Laptop Deployment Architecture

| Role | Machine | Responsibilities | Command |
| :--- | :--- | :--- | :--- |
| **Model & API** | Laptop 1 | Hosts trained classifier, HMM/Viterbi decoder, and road network cache | `uvicorn api.app:app --host 0.0.0.0 --port 8000` |
| **Evaluator UI** | Laptop 2 | Displays the 10 evaluation trips, triggers route reconstruction, renders telemetry | Open `ui/index.html` in browser |

---

## 5. Team & Academic Integrity

* **Institution:** PES University, RR Campus
* **Course:** UE24CS352A — Machine Learning
* **Team:** Team 8
* **Commit History:** Developed and authored independently by team members without external code borrowing, hardcoded evaluation answers, or data leakage across partitions.
