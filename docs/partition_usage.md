# Dataset partition used by every analysis

The report must state which partition each analysis used. Partitions: **train** (80 trips),
**student_test** (10 trips), **evaluator_test** (10 trips, never available to us, never used).

| Analysis / artifact | Script | Partition(s) | Notes |
| --- | --- | --- | --- |
| Dataset inventory, missing values, timestamp order, gaps | `eda/run_eda.py` → `eda/tables/*` | train + student_test (reported separately) | descriptive only |
| Sampling-interval, speed and trajectory-length plots | `eda/run_eda.py` → `eda/plots/*` | train + student_test | descriptive only |
| Road-network topology (isolated nodes, WCC/SCC, self-loops) | `eda/run_eda.py` → `road_network_graph_analysis.csv` | train + student_test (all 90 networks) | descriptive only |
| Folium maps (raw GPS, network, reference route) | `eda/run_eda.py` → `eda/maps/*` | train (trips 00000001, 00000002) | |
| Static trajectory + network plot | `eda/run_eda.py` | train (trip 00000001) | |
| Per-trip preprocessing / UTM zones / projection check | `data_processing/preprocessing_report.py` | train + student_test | descriptive only |
| Candidate miss-rate sweep and **choice of R = 100 m, k = 8** | `features/miss_rate_sweep.py` | **train only** | design choice, no test data used |
| Label alignment statistics | `features/miss_rate_sweep.py` | train | |
| Label-check Folium maps | `features/label_check_map.py` | train (trips 00000001, 00000033) | |
| Feature tables | `features/build_features.py` | train and student_test, separate files | same code for both; no fitted transforms |
| Feature means by class | `features/validate_tables.py` | train (reliable rows) | |
| Rule baselines | `features/validate_tables.py`, `models/run_experiments.py` | train (CV folds) and student_test | untrained |
| Model fitting + cross-validation (GroupKFold(5) by trip) | `models/run_experiments.py` | **train only** | scalers fitted on training folds |
| Model comparison on held-out trips | `models/run_experiments.py` | student_test | refit on all 80 train trips first |
| Model selection | `models/select_model.py` | primary criteria from train CV; student_test as confirmation | |
| Confusion matrices | `models/select_model.py` | student_test | |
| MLP training-loss curve | `models/run_experiments.py` | train | |
| Bayesian tuning (planned) | — | train only (GroupKFold); tuned model reported on student_test | |
| HMM decoder parameters (planned) | — | train + student_test only | |
