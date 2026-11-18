# PathSnap Backend API (Laptop 1)

FastAPI service for the two-laptop demo. It loads the model **once** at startup, warms up on every
trip in the served folder, and logs every request (trip, Top-1, continuity, route F1, timing) to the
console. Show this console to the evaluators.

## Run (from the repo root)

```bash
pip install -r requirements.txt

# Windows (cmd)
set PATHSNAP_DATA=C:\path\to\folder\containing\partition\folders
set PATHSNAP_TRIPS_PARTITION=evaluator_test          # folder with the 10 evaluation trips
uvicorn api.app:app --host 0.0.0.0 --port 8000

# PowerShell: $env:PATHSNAP_DATA="..."; $env:PATHSNAP_TRIPS_PARTITION="evaluator_test"
```

* Defaults: `PATHSNAP_DATA` = `../Student folder/Student folder` next to the repo;
  `PATHSNAP_TRIPS_PARTITION` = `student_test`.
* Put the 10 evaluator trips in `<PATHSNAP_DATA>/evaluator_test/<trip_id>/`
  (`<trip_id>.track/.nodes/.arcs/.route`).
* `--host 0.0.0.0` makes the API reachable from Laptop 2. Open TCP port 8000 in the Windows firewall
  and give Laptop 2 the address `http://<laptop-1-ip>:8000` (`ipconfig` shows the IP).
* Startup log: model loaded (~2–4 s), warm-up over all trips (~5 s), then `API ready`.

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | model loaded, model path, β / gap, data folder, partition, number of trips, warm-up stats |
| GET | `/trips` | `{"partition", "count", "trips": [...]}`: trip ids found by **scanning** the partition folder |
| POST | `/predict` | body `{"trip_id": "00000000", "use_viterbi": true}` |
| GET | `/predict/{trip_id}?use_viterbi=true` | same, for browser / curl testing |

`/predict` response: `gps_latlon`, `route_latlon` (polylines), `matched_latlon`, `route_arcs`,
`pred_arc_hmm`, `decoder`, `timings_s`, `decoder_used`, and **`metrics`**: `top1_accuracy`,
`route_continuity`, `route_f1`, `route_precision`, `route_recall`, `route_mismatch`, plus a
`comparison` of classifier-only vs classifier + HMM.

The metrics are computed **on the server** by `evaluation/evaluate.py`, the only code that reads
`.route`. Reference labels are never sent to the UI, and `inference/predict.py` never reads them.
With `use_viterbi: false` the classifier-only route is returned instead of the HMM route.
