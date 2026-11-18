# PathSnap: Two-Laptop Distributed Setup & UI Guide

This guide contains the exact steps for deploying and demonstrating PathSnap across two laptops on a shared phone hotspot.

---

## 1. Laptop 1: Model & Backend API

### Step 1: Connect to Hotspot & Note IP Address
1. Connect Laptop 1 to your phone's Wi-Fi hotspot.
2. Open terminal/PowerShell and get Laptop 1's local IP address:
   ```powershell
   ipconfig
   ```
   Look for **IPv4 Address** under your Wi-Fi adapter (e.g., `192.168.43.152`).

### Step 2: Open Port 8000 in Windows Firewall
Allow inbound traffic on port 8000 so Laptop 2 can connect:
```powershell
netsh advfirewall firewall add rule name="PathSnap API Port 8000" dir=in action=allow protocol=TCP localport=8000
```

### Step 3: Start the FastAPI Server
From the repository root on Laptop 1:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
You should see:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 2. Laptop 2: Evaluator User Interface

### Step 1: Connect to the Same Hotspot
Connect Laptop 2 to the same phone Wi-Fi hotspot.

### Step 2: Verify Network Connectivity
Open terminal/cmd on Laptop 2 and test connection to Laptop 1:
```bash
curl http://<LAPTOP_1_IP>:8000/health
```
*(Expected response: `{"status":"healthy","timestamp":...,"model_loaded":true}`)*

### Step 3: Launch the UI
1. Open `ui/index.html` in any web browser (Chrome, Edge, Firefox, Safari).
2. Enter Laptop 1's address in the top input box (e.g. `http://192.168.43.152:8000`).
3. Click **Connect & Refresh Trips**. The status badge turns green (`Connected to Laptop 1`).
4. Select any of the 10 evaluator trips from the dropdown.
5. Click **Reconstruct Route**. The map renders the raw GPS dots (blue), the reconstructed Viterbi route (green) and the snapped points (orange). The **Evaluation Results** panel shows Top-1 accuracy, route continuity, route F1 and route mismatch, plus the classifier-only vs HMM comparison. All are computed on Laptop 1 by `evaluation/evaluate.py`.
6. Untick **Enable HMM / Viterbi** to show the classifier-only route for comparison.

There is no offline or mock mode: every result shown comes from the API on Laptop 1.

---

## 3. Integration Test Results Across the 10 Student-Test Trips (2:00 PM)

All 10 student-test trips were validated at 2:00 PM with the Random Forest + HMM/Viterbi engine, before the
API/UI fixes that added server-side metrics. The current per-trip accuracy is in `results/hmm_vs_classifier.csv`:

| Evaluator Trip ID | Total GPS Fixes | Reconstructed Route Arcs | Polylines | End-to-End Latency |
| :---: | :---: | :---: | :---: | :---: |
| **`00000000`** | 2,503 pts | 213 arcs | 3 | 1.83 s |
| **`00000007`** | 1,354 pts | 123 arcs | 1 | 0.60 s |
| **`00000009`** | 1,049 pts | 501 arcs | 1 | 0.72 s |
| **`00000040`** | 2,995 pts | 335 arcs | 1 | 0.87 s |
| **`00000052`** | 262 pts | 93 arcs | 1 | 0.53 s |
| **`00000056`** | 3,958 pts | 1,372 arcs | 3 | 1.28 s |
| **`00000065`** | 768 pts | 202 arcs | 2 | 0.69 s |
| **`00000076`** | 1,486 pts | 438 arcs | 1 | 0.92 s |
| **`00000077`** | 3,911 pts | 417 arcs | 1 | 0.80 s |
| **`00000091`** | 874 pts | 238 arcs | 1 | 0.47 s |
| **All Trips** | **19,160 pts** | **3,932 arcs** | **13** | **Average: 0.87 s** |
