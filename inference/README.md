# Inference handoff (person A → person B): API and Folium maps

Run everything **from the repo root** (the modules import `data_processing`, `features`, `inference`,
`evaluation` as packages), e.g. `uvicorn api.app:app --host 0.0.0.0 --port 8000`.

## 1. `predict_trip()`: file, signature, model

File: **`inference/predict.py`**. It never opens a `.route` file.

```python
from inference.predict import load_config, load_model, predict_trip

cfg = load_config()                 # reads inference/config.json
model = load_model(cfg)             # joblib.load(cfg["model_path"]); do this ONCE at API startup (~1 s)

pred = predict_trip(trip, model=model, config=cfg, data_dir=DATA_DIR, include_points=True)
```

| Argument | Meaning |
| --- | --- |
| `trip` | a **trip id** such as `"00000007"` (searched in every sub-folder of `data_dir`), **or a path** to a trip folder containing `<id>.track/.nodes/.arcs` |
| `model` | an already-loaded model (recommended). If `None`, it is loaded from `cfg["model_path"]` on every call (slow) |
| `config` | dict from `load_config()`: `model_path`, `beta` (5.0), `gap_s` (60.0), `radius_m` (100), `top_k` (8) |
| `data_dir` | root folder holding partition folders; default = env var `PATHSNAP_DATA`, else `../Student folder/Student folder` next to the repo |

**Model requirements** (for `models/saved/rf_tuned.joblib`): any sklearn estimator/Pipeline with
`predict_proba(X)[:, 1]` = P(candidate is the true arc). `X` is a float32 numpy array of shape
`(n_candidates, 17)` with columns in exactly this order (`features.build_features.FEATURES`):

`dist_m, dist_rank, dist_minus_min, n_cands, proj_t, at_endpoint, heading_diff, heading_valid, speed_mps,
seg_len_m, dt_prev, dt_next, dist_prev_pt, dist_next_pt, from_in_deg, to_out_deg, twoway`

This is the same order as `candidates_*.parquet` minus the excluded columns, i.e. what
`run_experiments.py` trains on. **To switch to the tuned model**, set `"model_path":
"models/saved/rf_tuned.joblib"` in `inference/config.json` (or pass your own loaded model). Keep
`min_samples_leaf` large (≥ 20): the round-1 RF with leaf 5 needs several GB of RAM and froze an 8 GB laptop.
The compact RF (leaf 50) is 19.5 MB on disk.

## 2. What `predict_trip()` returns (JSON-serialisable dict)

All per-point lists are **aligned by index** with `gps_latlon` (point *i* = row *i* of the cleaned track).

| Key | Type | Meaning |
| --- | --- | --- |
| `trip_id`, `partition`, `epsg`, `n_points` | str/int | trip info (`epsg` = UTM zone used) |
| `gps_latlon` | `[[lat, lon], ...]` | raw GPS fixes |
| `timestamps` | `[float]` | seconds per fix |
| `pred_arc_hmm` | `[int]` | **predicted arc id per GPS point** (HMM/Viterbi); `-1` = point had no candidate (skipped) |
| `matched_latlon` | `[[lat, lon] or null]` | where each point was snapped on its predicted arc (`null` if skipped) |
| `pred_arc_classifier` | `[int]` | classifier-only arc per point (for comparison) |
| `segment_id` | `[int]` | Viterbi segment of each point (chain splits at gaps / dead ends) |
| `route_latlon` | `[[[lat, lon], ...], ...]` | **predicted route as polylines**, one polyline per continuous piece. Draw each with `folium.PolyLine` / Leaflet `L.polyline` |
| `route_arcs` | `[int]` | the same route as arc ids in driving order (`-1` marks a break) |
| `route_arcs_classifier` | `[int]` | classifier-only route (used by `evaluate`) |
| `n_skipped_points`, `skipped_points` | int, `[int]` | points with no candidate within 100 m (also logged) |
| `decoder` | dict | `beta`, `gap_s`, `n_segments`, `splits_gap`, `splits_no_route`, `route_fills`, `route_breaks` |
| `model_path` | str | model used |
| `timings_s` | dict | `load`, `features`, `model`, `hmm`, `total`, `shortest_paths` (seconds) |

Size: ~100 KB per student-test trip, ~0.1–1.5 s per trip after the model is loaded.

## 3. `evaluate()`: owner A, the only code that reads `.route`

File: **`evaluation/evaluate.py`**. Call it on the server **after** `predict_trip`, and return only the metrics.

```python
from evaluation.evaluate import evaluate
metrics = evaluate(pred, data_dir=DATA_DIR, trip_dir=None)   # trip_dir: optional explicit folder path
```

Returns (plain floats; `None` instead of NaN, so it is valid JSON):

```json
{"trip_id": "00000007",
 "hmm":             {"n_points": 1354, "top1_all": 0.9934, "top1_reliable": 0.9934, "route_continuity": 1.0,
                     "route_precision_len": 0.9921, "route_recall_len": 0.9930, "route_f1_len": 0.9925,
                     "route_mismatch_frac": 0.0149},
 "classifier_only": {"...": "same keys"}}
```

* `top1_all`: fraction of all GPS points whose predicted arc is the reference arc (**Top-1 accuracy**).
* `route_continuity`: fraction of consecutive predicted arcs that are equal or directly connected.
* `route_f1_len`: **whole-sequence measure**, the length-weighted F1 overlap between the predicted
  route and the reference route. `route_mismatch_frac` is Newson & Krumm's mismatch (lower = better).

## 4. Trip files on Laptop 1 and `GET /trips`

* Default layout (no setup needed if the repo sits next to the data):
  `Hackathon/repo/` and `Hackathon/Student folder/Student folder/{train,student_test}/<id>/`.
  Otherwise set `PATHSNAP_DATA` to the folder that **contains the partition folders**.
* **Evaluator trips:** copy the 10 folders into a new partition folder, e.g.
  `<PATHSNAP_DATA>/evaluator_test/<id>/`. `predict_trip("<id>")` finds them automatically.
* `GET /trips`: **scan a folder, don't hard-code ids**. Use
  `data_processing.load_data.list_trips(partition, data_dir)`, which returns the sorted sub-folder
  names, with the partition taken from an env var (e.g. `PATHSNAP_TRIPS_PARTITION`, default
  `"student_test"`, set to `"evaluator_test"` for the demo). This also guarantees no hard-coded
  answers in the repo.

## 5. For the Folium script (student-test trips)

```python
from data_processing.load_data import DATA_DIR, load_trip_dir
from evaluation.evaluate import read_route          # reference route (evaluation side only)
from features.candidates import generate_candidates

tdir = DATA_DIR / "student_test" / "00000007"
trip = load_trip_dir(tdir, load_route=False)
trip.nodes_ll        # (N, 2) array of [lon, lat]  <- note the order: lon first
trip.arcs            # (A, 2) array of [from_node, to_node]; row index = arc_id
ref = read_route(tdir)                               # array of reference arc ids, in order
pred = predict_trip(tdir, model=model, config=cfg)   # gps_latlon, route_latlon, matched_latlon, ...

# arc_id -> [[lat, lon], [lat, lon]]
seg = lambda a: [trip.nodes_ll[trip.arcs[a][0]][::-1].tolist(), trip.nodes_ll[trip.arcs[a][1]][::-1].tolist()]
# nearby road network only (the full network has up to 166k arcs and makes huge HTML):
near = set(generate_candidates(trip, radius=100.0).arc_id)
```

Layers: road network = `seg(a)` for `a in near` (grey); reference route = `seg(a)` for `a in ref`
(green); predicted route = `pred["route_latlon"]` (one `PolyLine` per piece, e.g. red); GPS points =
`pred["gps_latlon"]` (blue circles). `features/label_check_map.py` is a working template with a
legend and `LayerControl`. Use the `"OpenStreetMap"` tiles: CartoDB now needs an API key.
