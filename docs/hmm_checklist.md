# HMM / Viterbi requirements checklist

Every HMM requirement in *Models and Training Guidelines* §8–9 (items 1–16) and *Deliverables*
§5 (items 17–19), with where it is implemented and evidenced.

| # | Requirement (source) | Where it is met | Evidence |
| --- | --- | --- | --- |
| 1 | Use sequential information and road-network connectivity to get a coherent route (MTG §8) | `inference/hmm.py`, `inference/road_graph.py` (directed graph of the trip's arcs) | continuity 0.927 → 0.982 (`results/hmm_vs_classifier.csv`) |
| 2 | Use the selected model's outputs consistently (MTG §8) | emissions = selected RF family's `predict_proba` (`inference/predict.py`); same candidates and features as round 1 | `inference/config.json`, `results/rf_compact_vs_round1.csv` |
| 3 | Describe the hidden-state representation (MTG §9) | states = candidate arcs of each fix (R = 100 m, k = 8) + projection position `proj_t` | `docs/hmm_note.md` §1 |
| 4 | Preserve the candidate-to-road-segment correspondence (MTG §9) | `Lattice.arcs[k][j]` is the arc of state j; decoded index → arc id via `state_arcs()` | `inference/hmm.py` |
| 5 | Explain emission modelling (MTG §9) | `log max(p_RF, 1e-6)` | `docs/hmm_note.md` §2 |
| 6 | Explain transition modelling (MTG §9) | `max(−\|d_net − d_gps\| / β, log 1e-12)`, shortest paths with cutoff `d_gps + 1000 m` | `docs/hmm_note.md` §3 |
| 7 | Explain the Viterbi recurrence (MTG §9) | `delta_k(j) = log e_k(j) + max_i[delta_{k−1}(i) + log A(i,j)]` + backtracking | `docs/hmm_note.md` §5, `viterbi()` |
| 8 | Handle missing candidates (MTG §9) | fixes with no candidate are skipped, logged and returned as `skipped_points` (`-1`) | `inference/predict.py`; 1,413 train / 0 student-test fixes |
| 9 | Handle invalid transitions (MTG §9) | unreachable pairs get floor probability 1e-12 (never zero); if no pair is reachable the chain is split | `viterbi()`; 154 dead-end splits on train (149 in trip 54) |
| 10 | Handle GPS gaps (MTG §9) | chain split when Δt > gap_s (60 s); each segment decoded independently | 70 gap splits train, 1 student-test (`results/hmm_tuning_per_trip.csv`) |
| 11 | Return a predicted road segment for each relevant GPS observation (MTG §9) | `pred_arc_hmm` per fix (+ `matched_latlon`, `route_latlon`) | `inference/README.md` |
| 12 | Compare classifier-only with classifier + HMM (MTG §9) | `evaluation/compare_hmm.py` | `results/hmm_vs_classifier.csv`, `.png` |
| 13 | Report pointwise accuracy (MTG §9) | Top-1 per GPS fix: 0.860 → 0.934 | `results/hmm_vs_classifier.csv` |
| 14 | Report sequence-level performance (MTG §9) | length-weighted route F1 0.785 → 0.973; route mismatch 0.588 → 0.058 | `results/hmm_vs_classifier.csv` |
| 15 | Analyse route continuity (MTG §9) | continuity per trip and pooled (0.927 → 0.982; every trip improves); route fills/breaks | `results/hmm_vs_classifier.png` panel 2, `docs/hmm_note.md` §7 |
| 16 | Optimise decoder parameters on train and student-test only (MTG §9) | grid of β × gap on train (out-of-fold RF probabilities), student-test as confirmation | `evaluation/tune_hmm.py`, `results/hmm_tuning.csv`, `.png` |
| 17 | What happens when consecutive fixes suggest disconnected segments (Del §5 Q5) | transition at floor (strongly avoided); if nothing connects, segment split | `docs/hmm_note.md` §3–4 |
| 18 | Did HMM decoding improve the route, with experimental results (Del §5 Q6) | yes on all 10 student-test trips and 79/80 train trips | `results/hmm_vs_classifier.csv`, `results/hmm_tuning_per_trip.csv` |
| 19 | Discuss improvements and cases where HMM decoding did not help (Del §5) | trip 77: slow two-way street, the HMM refuses the U-turn the classifier correctly flips to; train trip 63 (−2.3 pp) | `results/hmm_failure_cases.csv`, `docs/hmm_note.md` §8 |

Also required: HMM configuration exported as an artifact (`inference/config.json`). Prediction
(`inference/predict.py`, never reads `.route`) and evaluation (`evaluation/evaluate.py`, the only
reader of `.route`) live in separate files, so reference labels stay server-side.
