# Part A: preprocessing, candidates, labels, features

All numbers below come from the CSV/JSON files in `results/`. Run the scripts to reproduce them.

## How to reproduce

```bash
pip install -r requirements.txt
# raw data is expected at ../Student folder/Student folder/{train,student_test}/<trip_id>/
# (or point to it with:  set PATHSNAP_DATA=<path>)
python data_processing/preprocessing_report.py   # results/preprocessing_report.csv
python features/miss_rate_sweep.py               # results/candidate_miss_rate.csv  (train only)
python features/build_features.py                # data/processed/*.parquet  (~1 min)
python features/validate_tables.py              # sanity checks, baselines, miss-rate plot
python features/label_check_map.py 00000001 train  # results/maps/label_check_*.html
```

## Pipeline

| Step | File | What it does |
|---|---|---|
| Load + clean | `data_processing/load_data.py` | Reads `.track/.nodes/.arcs/.route`. Drops non-finite, out-of-range and (0,0) fixes, sorts by time and removes duplicate timestamps (no fix was dropped in either partition). |
| Projection | `data_processing/load_data.py` | **Per-trip WGS84/UTM zone** from the GPS centroid (EPSG 326zz north / 327zz south). The trips use **25 different zones** (Europe, Russia, Americas, Asia, southern hemisphere), so a single CRS would be wrong. UTM is metric and conformal, with <0.1 % scale error inside a zone. Checked against haversine: max difference 0.55 %, which comes from haversine's spherical-Earth assumption. |
| Candidates | `features/candidates.py` | Arcs are densified every 10 m and the samples go into a `scipy.cKDTree`. A ball query of radius R+5 m gives a guaranteed superset of arcs, then the exact point-to-segment distance is applied, then a **top-k that keeps distance ties** (u->v and v->u of a two-way street are identical geometrically). |
| Labels | `features/labels.py` | `.route` is just an ordered arc list, so each GPS fix is matched to a route arc by **dynamic programming**. Route position must be non-decreasing in time and the summed (capped at 200 m) point-to-arc distance is minimised. The result is exact and O(N·M). |
| Features | `features/build_features.py` | 17 features per (point, candidate). See the docstring. Built from GPS + network only; the route is used only for `label`. |

## Choosing R and k (training trips only)

`results/candidate_miss_rate.csv`, `results/candidate_miss_rate.png`

| R (m) | k | miss rate, all pts | miss rate, alignable pts | mean cands/pt |
|---|---|---|---|---|
| 25 | 12 | 13.3 % | 3.87 % | 4.16 |
| 50 | 12 | 11.7 % | 2.00 % | 6.52 |
| 75 | 8 | 11.4 % | 1.74 % | 6.11 |
| **100** | **8** | **11.1 %** | **1.37 %** | **6.70** |
| 100 | all | 9.9 % | 0.00 % | 22.36 |

Rule: pick the smallest mean candidate set whose miss rate on alignable points is ≤ 1.5 %. That gives **R = 100 m, k = 8**. It beats R=50, k=12 on both miss rate and set size. A wide radius rescues outlier fixes while k stops dense junctions from flooding the set.

The roughly 10 % floor in the all-points miss rate is **not** caused by the radius. Those points sit more than 100 m from any arc of their reference route:
* trip `00000054` (19,313 fixes): only 14 % of its fixes have any road arc within 50 m, so the track runs where the supplied network has no road.
* trips `00000033`, `00000002`, … : the `.route` file has breaks (`route_contiguous_frac` < 1 in 11 trips). Part of the drive is missing from the reference, so those fixes align to a distant arc.

These are flagged `label_reliable = 0` (aligned distance > 100 m). Training uses 90.1 % of train fixes and 99.9 % of student_test fixes as reliable.

## Exported tables (`data/processed/`)

| File | Rows | Grain |
|---|---|---|
| `candidates_train.parquet` | 1,396,401 | one (GPS fix, candidate arc) pair, 80 trips |
| `candidates_student_test.parquet` | 135,725 | same, 10 trips |
| `points_train.parquet` | 208,417 | one GPS fix |
| `points_student_test.parquet` | 19,160 | one GPS fix |

`candidates_*` columns: `trip_id, partition, point_idx, arc_id, t`, the 17 features, `label` (1 = reference arc), `label_reliable`.
`points_*` columns: `trip_id, point_idx, t, lon, lat, x, y, epsg, true_arc, route_pos, label_dist_m, label_reliable, n_cands, true_in_cands`.

Positive rate is 13.3 % (train) / 14.1 % (student_test). That is about 1 positive per 7 candidates, so class weighting or macro-F1 is needed.

### Notes for training (person B)
* **Group by `trip_id`** (GroupKFold): fixes of one trip are highly correlated.
* **Fit on `label_reliable == 1` rows**, otherwise the model learns that the obviously-nearest arc is "wrong" on route gaps. Report metrics both ways.
* **Top-1 / miss rate denominator = `points_*` table**, not the candidate table. Fixes with no candidates (1,413 in train, 0 in student_test) or whose true arc was cut by k count as wrong.
* Exclude ID/meta columns from X: `trip_id, partition, point_idx, arc_id, t, label, label_reliable`.
* Scale features for LR / KNN / SVM / MLP (fit the scaler on the training fold only).

## Floor to beat (rule baselines, no learning): `results/baseline_top1.csv`

| partition | nearest arc | nearest + 0.5·heading_diff |
|---|---|---|
| train (reliable pts) | 55.6 % | 76.3 % |
| student_test (reliable pts) | 54.0 % | 78.4 % |

Nearest-arc alone is about 50 % because on two-way streets the two directions tie on distance. Heading breaks the tie, which is the main reason it is a feature.

## Known data-quality issues seen in this step
* Huge time gaps: trip 90 (42 h), trip 78 (14.7 h), trips 21/69 (~40 min). Captured by `dt_prev/dt_next`; the HMM should treat a big gap as a break.
* 6–7 % of steps are not exactly 1 s apart in train (98 % are 1 s in student_test).
* At low speed (< ~5 m/s) the GPS heading is unreliable. 3 % of true arcs have heading_diff > 150°, and those have a mean speed of 4 m/s. `speed_mps` and `heading_valid` let the model discount heading there.
* 10–13 % of route arcs get no GPS fix, because short arcs are crossed in < 1 s. Labels therefore never use them, but they are still part of the route.
