# HMM / Viterbi route reconstruction: note for the report (Section 5)

All numbers come from `results/hmm_tuning.csv`, `results/hmm_tuning_per_trip.csv`,
`results/hmm_vs_classifier.csv` and `results/hmm_failure_cases.csv`. Code is in `inference/`
(prediction, never reads `.route`) and `evaluation/` (the only code that reads `.route`).

## 1. Hidden states

For each GPS fix, the hidden states are its **candidate arcs**: the same generator as round 1
(arcs within R = 100 m, top-k = 8 keeping distance ties). Each state also keeps the position of the
fix's projection along the arc (`proj_t`, 0 = start, 1 = end), which the transition model needs.
The candidate → arc correspondence is never re-ordered, so the decoded state is directly an arc id.
Fixes with no candidate are skipped and logged (0 on student-test; 1,413 of 208,417 on train).

## 2. Emission probabilities

`log e(arc | fix) = log( max( p_RF(arc | features), 1e-6 ) )`

`p_RF` is the Random Forest's candidate probability. The 1e-6 floor stops one very confident
"no" from making a state impossible. Model: `models/saved/rf_compact.joblib`, the selected RF
configuration with `min_samples_leaf = 50` instead of 5. The original is several GB in memory and
froze the laptop; the compact one is 19.5 MB with essentially the same student-test scores
(Top-1 0.8603 vs 0.8636, macro-F1 0.9177 vs 0.9180, log loss 0.0974 vs 0.0966;
`results/rf_compact_vs_round1.csv`).

## 3. Transition probabilities

Newson & Krumm style, using a directed graph of the trip's road network (`inference/road_graph.py`;
edges = arcs, weight = arc length in UTM metres):

* `d_gps` = straight-line distance between the two consecutive fixes.
* `d_net` = driving distance between the two projected positions:
  * same arc: `|t_b − t_a| · len(arc)`
  * different arcs: `(1 − t_a)·len(a) + shortest_path(to(a) → from(b)) + t_b·len(b)`
* `log A(a → b) = max( −|d_net − d_gps| / β , log(1e-12) )`

The idea: on the true route, the driven distance between two fixes ≈ the straight-line GPS
distance. Jumping to a parallel road, the wrong direction or a disconnected arc forces a long
detour, so `|d_net − d_gps|` is large and the transition is penalised.

Shortest paths use Dijkstra with a cutoff of `d_gps + 1000 m`, inside a 1.5 km corridor around the
track. If one arc **cannot reach** the next within the cutoff, the transition gets the floor
probability 1e-12: **very small, never zero**.

## 4. Gaps and dead ends

The chain is **split** and each segment is decoded independently when:

* the time gap between consecutive fixes exceeds **gap_s = 60 s**, or
* **no** candidate of one fix can reach **any** candidate of the next (no route possible).

At the chosen setting there are 70 gap splits and 154 dead-end splits across the 80 train trips.
**149 of the 154 dead ends are trip 00000054**, whose track runs where the supplied network has no
roads. Student-test has 1 gap split and 0 dead ends. Fixes without candidates are skipped and
listed in the API output (`skipped_points`).

## 5. Decoding (Viterbi)

```
delta_0(j) = log e_0(j)
delta_k(j) = log e_k(j) + max_i [ delta_{k-1}(i) + log A_k(i, j) ],   back_k(j) = argmax_i
```

Backtrack from `argmax_j delta_end(j)` in each segment. All in log space. The predicted route is
then made continuous by inserting shortest paths between consecutive matched arcs that are not
directly connected (`reconstruct_route`).

## 6. Tuning β and the gap threshold

* Grid: β ∈ {1, 2, 3, 5, 8, 12, 20, 35, 60} m × gap_s ∈ {10, 30, 60, 120, 300, ∞} s = 54 settings.
* **Tuned on the 80 train trips with out-of-fold RF probabilities** (GroupKFold(5) by trip). A
  model scoring its own training rows is over-confident, which would bias β. Student-test
  (final model) is the confirmation set.
* Criterion: pooled Top-1 on reliable train fixes. **Chosen: β = 5 m, gap_s = 60 s**
  (train Top-1 reliable 0.9185).
* **β matters, the gap threshold hardly does:** across gap thresholds Top-1 varies by < 0.1 pp at
  any β, while β moves it by 2.6 pp on train. Small β trusts the road network more, large β trusts
  the classifier more. Student-test alone would prefer β = 2 (0.9406 vs 0.9348); we kept the
  train-chosen value so the test set was not used for tuning. Plot: `results/hmm_tuning.png`.
* **Follow-up experiment** (`results/hmm_emission_experiment.csv`): a finer grid β ∈ {2, …, 7}
  confirms β = 5 on train. **Normalising emissions per fix** (p / Σ p over the fix's candidates)
  gives *identical* results. This is expected: it adds the same constant to every state's score
  at that time step, so the Viterbi argmax cannot change. The emission scale is effectively
  controlled by β alone.

## 7. Did the HMM help? Student-test, 10 trips (`results/hmm_vs_classifier.csv`)

| Metric (pooled over 19,160 fixes) | Classifier only | Classifier + HMM |
| --- | --- | --- |
| Top-1 accuracy (all fixes) | 0.8603 | **0.9339** |
| Top-1 accuracy (reliable fixes) | 0.8611 | **0.9348** |
| Route continuity (consecutive arcs equal or connected) | 0.9274 | **0.9819** |
| Route F1, length-weighted (whole sequence; mean over trips) | 0.7852 | **0.9729** |
| Route precision / recall (length-weighted) | 0.660 / 0.998 | **0.952** / 0.997 |
| Route mismatch fraction (Newson & Krumm; lower is better) | 0.588 | **0.058** |
| Prediction time per trip (s, full pipeline) | 0.27 | 0.27 |

* **Every one of the 10 student-test trips improved.** Gains range from +2.3 pp (trip 00000000:
  0.940 → 0.962) to +13.6 pp (trip 00000077: 0.704 → 0.840).
* On the 80 train trips (OOF): Top-1 reliable 0.8376 → 0.9185, route F1 0.759 → 0.951. Only one
  train trip got worse (00000063, −2.3 pp).
* **Why route precision jumps:** classifier-only flips between neighbouring arcs, so its
  reconstructed route contains many spurious detours (precision 0.66). The HMM removes them.
  Recall is ~1 for both because the true road is almost always among the candidates.

## 8. A case where the HMM did not help (`results/hmm_failure_cases.csv`)

**Trip 00000077, fixes 865–947** (three windows of 5–6 fixes): classifier right, HMM wrong.

* Arcs **181 and 6394 are the two directions of the same 38.8 m two-way street** (69 → 2890 and
  2890 → 69; identical geometry).
* The vehicle is **crawling** (1.2–4.6 m/s) and the GPS is 6–22 m off. The GPS heading is noise
  (heading_diff swings between 5° and 175°), so the classifier's probabilities for the two
  directions flip with small margins (e.g. 0.42 vs 0.36).
* For the HMM, switching between the two directions is a **U-turn**: the network distance
  (to the end of the arc and back) is far larger than the 2–4 m the GPS moved. With β = 5 m that
  transition is heavily penalised, so Viterbi **sticks with the direction it chose first (181)**
  and refuses the classifier's correct flips.
* The reference labels here are themselves uncertain: they are inferred by alignment, and at
  walking pace on a short two-way street the true direction is ambiguous. This is the HMM's known
  weakness: a strong smoothness prior over-rules weak, noisy emissions exactly where the vehicle
  manoeuvres slowly (U-turns, parking, stop-and-go).

## 9. Reproduce

```bash
python models/oof_probabilities.py --params models/rf_compact_params.json --name rf_compact   # ~9 min
python evaluation/tune_hmm.py --probs rf_compact     # ~2 min; writes beta/gap_s to inference/config.json
python evaluation/compare_hmm.py                     # student-test comparison CSV + failure cases
```
