# Inference: predict_trip() and evaluate() (person A → person B's API)

```python
from inference.predict import load_config, load_model, predict_trip   # never reads .route
from evaluation.evaluate import evaluate                               # the only code that reads .route

cfg = load_config()            # inference/config.json: model_path, beta, gap_s, radius_m, top_k
model = load_model(cfg)        # load ONCE at API startup

pred = predict_trip("00000000", model=model, config=cfg)   # trip id (searched in PATHSNAP_DATA) or a trip folder path
metrics = evaluate(pred)                                   # server-side only; never send .route to the UI
```

## `predict_trip()` output (JSON-serialisable)

| key | meaning |
| --- | --- |
| `trip_id`, `partition`, `epsg`, `n_points` | trip info |
| `gps_latlon` | `[[lat, lon], ...]` raw GPS fixes (for the UI map) |
| `timestamps` | seconds, per fix |
| `route_latlon` | predicted route as a list of polylines `[[[lat, lon], ...], ...]`; a new polyline starts where the chain was split |
| `route_arcs` | predicted route as arc ids (`-1` = break) |
| `pred_arc_hmm` | per-fix arc from HMM/Viterbi (`-1` = fix had no candidate, skipped) |
| `pred_arc_classifier` | per-fix arc from the classifier alone (for comparison) |
| `segment_id` | Viterbi segment of each fix |
| `n_skipped_points`, `skipped_points` | fixes with no candidate within R (logged) |
| `decoder` | beta, gap_s, number of segments, splits at gaps / dead ends, route fills / breaks |
| `timings_s` | load, features, model, hmm, total |

## `evaluate(pred)` output

`{"trip_id", "hmm": {...}, "classifier_only": {...}}`. Each block has `top1_all`, `top1_reliable`,
`route_continuity`, `route_precision_len`, `route_recall_len`, `route_f1_len` and `route_mismatch_frac`.

## Data location for the demo

Set `PATHSNAP_DATA` to the folder that contains the partition folders (e.g. `.../Student folder/Student folder`,
or the folder holding the evaluator trips). `predict_trip("<id>")` searches every sub-folder for the trip.

## Memory note

The original round-1 RF (`min_samples_leaf=5`) is several GB in memory and froze this laptop.
The decoder uses `models/saved/rf_compact.joblib` (`min_samples_leaf=50`, see
`models/rf_compact_params.json`). Keep the tuned model's `min_samples_leaf` similarly large,
or check its size, before loading it in the API.
