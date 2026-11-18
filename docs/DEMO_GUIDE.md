# PathSnap Demo Setup Guide: Laptop 1 (API) + Laptop 2 (UI)

**Team 8. Final evaluation.** Follow this top to bottom. Total setup time is about 10 minutes.
Everything here was tested end to end on the 10 evaluator trips (all 10 OK, 0.3–2 s per trip).

| | Laptop 1: Model & API | Laptop 2: Evaluator UI |
| --- | --- | --- |
| Runs | FastAPI server (`api/app.py`): RF + HMM/Viterbi + server-side evaluation | `ui/index.html` in a browser (Leaflet map) |
| Needs | the repo, Python + requirements, the data folder (with `evaluator_test`) | the repo (or just `ui/index.html`), Chrome/Edge, **internet** (map tiles and Leaflet from a CDN) |
| Shows the evaluators | the **console log** of every request | the trip list, the route on the map and the **metrics panel** |

---

## Quick start: two double-clicks (recommended)

| Laptop | Double-click (in the repo folder) | What it does automatically |
| --- | --- | --- |
| **Laptop 1** | **`LAPTOP1_START_DEMO.bat`** | ① checks Python, installs missing packages ② adds the firewall rule for port 8000 (asks for admin **once**: click *Yes*) ③ finds the `evaluator_test` folder (copies it from `Hackathon\evaluator_test\evaluator_test` if needed) ④ pre-flight check of every trip (`10/10 trips ready`) ⑤ prints this laptop's IP, marking the one to use ⑥ starts the API and saves its log to `results\demo_logs\` |
| **Laptop 2** | **`LAPTOP2_START_DEMO.bat`** | ① asks for Laptop 1's IP (remembers the last one) ② tests the connection and shows the API status (model loaded, 10 trips) or what to fix ③ opens the UI **already connected**, with the trips in the dropdown |

**Order:**

1. Both laptops join the same hotspot and run `git pull`.
2. Laptop 1: double-click `LAPTOP1_START_DEMO.bat`, wait for **`API ready`**, and read out the IP
   marked `<-- USE THIS`.
3. Laptop 2: double-click `LAPTOP2_START_DEMO.bat`, type that IP, press Enter. The browser opens
   the connected UI.
4. Run the demo script (§4).

Rehearse with the student-test trips: `LAPTOP1_START_DEMO.bat student_test`. To skip the
question on Laptop 2: `LAPTOP2_START_DEMO.bat 192.168.x.x`.

The sections below are the manual equivalent and the background, if the launchers can't be used.

---

## 0. Before you start (both laptops)

- [ ] Both laptops **plugged in**; sleep disabled (Settings → System → Power → Screen and sleep → *Never*).
- [ ] One **phone hotspot** switched on, with mobile data, so Laptop 2 can load map tiles.
- [ ] Both laptops connected to that hotspot.
- [ ] `git pull` in the repo on both laptops (latest commit, model `models/saved/rf_compact.joblib` present).

---

## 1. Laptop 1: one-time setup

**1.1 Python packages** (once, from the repo folder):

```bat
cd <path>\repo
pip install -r requirements.txt
```

**1.2 Data folder layout.** The API looks for the trips next to the repo:

```text
Hackathon\
  repo\                        <- the git repository (run_api.bat is here)
  Student folder\
    Student folder\
      evaluator_test\          <- the 10 evaluator trips (already in place on our Laptop 1)
        00000015\00000015.track  .nodes  .arcs  .route
        00000017\...
      student_test\            <- optional (practice trips)
```

If the data is somewhere else, run `set PATHSNAP_DATA=<folder that contains evaluator_test>`
in the same window before step 2. `.route` files are only read by the server-side evaluation;
the prediction code never opens them.

**1.3 Firewall: allow port 8000** (once). Open **Command Prompt as Administrator**:

```bat
netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
```

If Windows instead shows an *"Allow access?"* pop-up for Python the first time the API starts,
tick **both Private and Public** networks and click Allow. Phone hotspots are often treated as
"Public".

---

## 2. Laptop 1: start the API

Double-click **`repo\LAPTOP1_START_DEMO.bat`** (recommended, see Quick start) or the simpler
**`repo\run_api.bat`**, or from a Command Prompt in the repo folder:

```bat
run_api.bat                    &rem serves the evaluator_test trips (default)
run_api.bat student_test       &rem practice with the student-test trips
```

What it does, in order:

1. **Pre-flight check.** Every trip in the folder is loaded and run through the full pipeline
   once. You should see 10 lines `[ OK ]` and `10/10 trips ready.`
2. **Prints this laptop's IP addresses.** Note the IPv4 of the Wi-Fi adapter connected to the
   hotspot, e.g. `192.168.43.152`. **Laptop 2 needs this.**
3. **Starts the API** on port 8000. Wait for these lines (about 15 s):

```text
| pathsnap.api | model loaded once at startup: models/saved/rf_compact.joblib (3.4s)
| pathsnap.api | data dir: ...\Student folder\Student folder | serving partition 'evaluator_test' (10 trips) | beta=5.0 gap_s=60.0
| pathsnap.api | warm-up: predicted 10 trips in 9.7s
| pathsnap.api | API ready
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Leave this window open and visible.** It is the "API logs on the receiving laptop" the
evaluators ask to see. Equivalent manual command:
`set PATHSNAP_TRIPS_PARTITION=evaluator_test` then
`python -m uvicorn api.app:app --host 0.0.0.0 --port 8000`.

**Quick self-test on Laptop 1:** open `http://localhost:8000/health` in a browser. You should see
`"status":"healthy","model_loaded":true,"partition":"evaluator_test","n_trips":10`.

---

## 3. Laptop 2: connect the UI

1. **Check the network path first.** On Laptop 2 open `http://<laptop-1-ip>:8000/health` in the
   browser. The same JSON as above means the network and firewall are fine.
2. Open **`repo\ui\index.html`** (double-click; it opens in the default browser).
3. In **Laptop 1 Host Address** type `http://<laptop-1-ip>:8000` (e.g. `http://192.168.43.152:8000`).
4. Click **Connect & Refresh Trips**. The badge turns green ("Connected to Laptop 1") and the
   **dropdown fills with the 10 evaluator trips**. They come from `GET /trips`, which scans the
   folder on Laptop 1; nothing is hard-coded.

---

## 4. Running the demo (script)

| Step | Who / where | Do | Say |
| --- | --- | --- | --- |
| 1 | Evaluator, Laptop 2 | Pick any trip in the dropdown | "These are the 10 evaluator trips, listed live from Laptop 1." |
| 2 | Laptop 2 | Click **Reconstruct Route** (HMM ticked) | "The UI sends only the trip id to the API." |
| 3 | Laptop 2 | Point at the map | "Blue = raw GPS, green = route predicted by RF + HMM/Viterbi, orange = each fix snapped to its road. Layers can be toggled." |
| 4 | Laptop 2 | Point at **Evaluation Results** | "Top-1 accuracy, route continuity and route F1 are computed on Laptop 1 against its reference route. The reference never leaves Laptop 1. Below: classifier alone vs classifier + HMM." |
| 5 | **Laptop 1** | Show the console | "Here is the request in the API log: trip, points, Top-1, continuity, route F1, timing." |
| 6 | Laptop 2 | Untick **Enable HMM**, Reconstruct again | "Without the HMM: the classifier's per-fix choice, noisier route and lower continuity." Then tick it again. |
| 7 | Anyone | Repeat for other trips the evaluator chooses | The system works for all 10 trips. |

**One-sentence explanation:** "A Random Forest scores every road within 100 m of each GPS point;
an HMM decoded with Viterbi picks the most likely *connected* sequence of roads, using the
difference between driving distance and GPS distance; accuracy is computed on the server, and
the prediction code never reads the reference route."

**What to expect on the evaluator trips** (our dry run of the frozen pipeline,
`results/hmm_vs_classifier_evaluator_test.csv`): about 1–2 s per request.

| Trip | Fixes | Top-1 classifier → HMM | Continuity (HMM) | Route F1 (HMM) |
| --- | --- | --- | --- | --- |
| 00000015 | 6,854 | 59.4 % → **92.0 %** | 99.7 % | 0.887 |
| 00000017 | 1,346 | 91.2 % → **94.0 %** | 87.1 % | 0.974 |
| 00000023 | 458 | 93.9 % → **94.5 %** | 97.6 % | 0.974 |
| 00000030 | 654 | 78.6 % → **83.5 %** | 90.7 % | 0.971 |
| 00000049 | 2,642 | 97.8 % → **99.3 %** | 99.3 % | 0.997 |
| 00000059 | 1,031 | 95.3 % → **97.6 %** | 93.1 % | 0.987 |
| 00000067 | 916 | 89.9 % → **97.1 %** | 99.1 % | 0.979 |
| 00000073 | 1,144 | 97.0 % → **98.6 %** | 99.0 % | 0.996 |
| 00000087 | 1,399 | 93.6 % → **99.6 %** | 99.0 % | 0.998 |
| 00000094 | 3,230 | 86.1 % → **95.3 %** | 97.9 % | 0.955 |
| **All (19,674 fixes)** | | **80.5 % → 94.9 %** | **97.7 %** | **0.972** |

Good trips to show first: **00000015** (largest HMM gain, 59 % → 92 %), **00000087** (99.6 %),
**00000067** (+7 pp). Trip 00000073 is decoded in 4 segments because of GPS gaps.

---

## 5. Screenshots to take during the demo (for report §7)

1. **Laptop 2:** the UI with a trip's route and the Evaluation Results panel visible (Win + Shift + S).
2. **Laptop 1:** the console window showing `API ready` and a few `trip … Top-1=…` lines.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Pre-flight says `ERROR: ...evaluator_test does not exist` | data folder not next to the repo | copy the folder (§1.2) or `set PATHSNAP_DATA=...` |
| `ModuleNotFoundError` when starting | packages missing | `pip install -r requirements.txt` (§1.1) |
| `[Errno 10048]` / address already in use | an old API is still running | close the other window, or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` |
| Laptop 2: `/health` does not load | firewall or different network | §1.3 firewall rule; both on the **same** hotspot; use the hotspot adapter's IPv4 |
| Still blocked on the hotspot | some phones isolate clients | turn on **Windows Mobile Hotspot** on Laptop 1 (Settings → Network → Mobile hotspot) and connect Laptop 2 to it; Laptop 1's IP is then usually `192.168.137.1` |
| UI badge "Offline / Unreachable" | wrong address in the box | must be `http://IP:8000`, with `http://` and port |
| Dropdown empty | wrong partition | `/health` shows `partition` and `n_trips`; restart with `run_api.bat evaluator_test` |
| Map grey / no tiles | Laptop 2 has no internet | the hotspot needs mobile data (tiles come from OpenStreetMap) |
| Map tiles say **"403 Access blocked"** | UI opened as a local file: OpenStreetMap blocks tile requests without a Referer | open the UI with `LAPTOP2_START_DEMO.bat` (serves it at `http://127.0.0.1:8080`); a local-file UI now uses the Esri street map automatically |
| First request slow | warm-up skipped | normal requests take 0.3–2 s; wait for `API ready` before demoing |
| HTTP 404 for a trip | trip not in the served folder | check the folder name and contents; `GET /trips` lists what is served |

**Stop the API:** press **Ctrl + C** in the Laptop 1 window.
