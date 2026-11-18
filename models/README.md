# Model Training & Evaluation (Task B)

Training harness, model comparison and model selection for the PathSnap candidate classifier.

## Task framing

Each row of `data/processed/candidates_*.parquet` is one (GPS point, candidate arc) pair with 17
features. `label = 1` if the arc is the point's reference arc. A binary classifier scores every
candidate; for each GPS point the highest-scoring candidate is the predicted arc (Top-1). The
probabilities are later used as HMM emission scores.

## Protocol (`run_experiments.py`)

| Item | Choice | Why |
| --- | --- | --- |
| CV | `GroupKFold(n_splits=5)` grouped by `trip_id`, 80 train trips | consecutive points of a trip are near-duplicates; grouping prevents leakage and simulates unseen trips |
| Final evaluation | refit on all 80 train trips, evaluate on the 10 student_test trips | development-stage comparison as required |
| Fitting rows | `label_reliable == 1` only (9.9 % of train points excluded) | points > 100 m from their aligned reference arc come from route gaps; all their candidates would be labelled 0 |
| Evaluation rows | row metrics on reliable rows (`*_all` = all rows); Top-1 over **all points** of `points_*.parquet` | points with no candidate count as errors |
| Preprocessing | `StandardScaler` inside each sklearn `Pipeline` (LR, KNN, MLP) | scaler fitted on the training fold only |
| Seed | 42 for every model and subsample | reproducibility (`results/configs.json`) |
| Threshold | 0.5 for row-level metrics | Top-1 does not depend on a threshold |

## Models (five families) and key configuration

| Model | Family | Key configuration | Loss / objective |
| --- | --- | --- | --- |
| Logistic Regression | linear / probabilistic | L2, C=1, max_iter=1000, scaled | log loss (cross-entropy) |
| KNN | instance-based | k=25, scaled, fit on a 100k-row random subsample (inference cost) | none (majority vote of neighbours) |
| Random Forest | bagged trees | 100 trees, max_depth=18, min_samples_leaf=5 | Gini impurity per split |
| HistGradientBoosting | boosted trees | 200 iterations, lr=0.1, 31 leaves (LightGBM was not installed) | log loss, gradient boosting |
| MLP | neural network | (64, 32) ReLU, Adam, early stopping, ≤ 400k-row subsample (time cap) | log loss; loss curve in `results/mlp_loss_curve.png` |

Untrained rule baselines (nearest arc; nearest arc with a heading filter; part A's dist + 0.5·heading
rule) are reported alongside as the floor to beat. They are not counted among the five models.

## Metrics recorded

Accuracy, precision, recall, F1, macro-F1, log loss, confusion matrix counts, observation-level Top-1
(all points / reliable points / given the true arc is a candidate), candidate miss rate, route continuity
(consecutive predicted arcs equal or connected), fit time and inference time. Each is recorded per CV
fold (`results/cv_per_fold.csv`) and as CV mean + student-test (`results/model_comparison.csv`).

## Results and selection (`select_model.py` → `results/model_selection.md`)

| Model | CV Top-1 (reliable) | CV log loss | CV macro-F1 | test Top-1 | test macro-F1 | test log loss | inference (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Random Forest** | **0.8365** | **0.1081** | **0.9073 ± 0.017** | 0.8636 | **0.9180** | **0.0966** | 0.354 |
| HistGradientBoosting | 0.8354 | 0.1235 | 0.9055 ± 0.018 | 0.8602 | 0.9161 | 0.1414 | 0.447 |
| Logistic Regression | 0.8327 | 0.1337 | 0.8997 ± 0.018 | 0.8539 | 0.9103 | 0.1238 | 0.028 |
| KNN | 0.8271 | 0.2443 | 0.8975 ± 0.019 | 0.8557 | 0.9108 | 0.2052 | 13.547 |
| MLP | 0.8263 | 0.1272 | 0.8960 ± 0.015 | **0.8641** | 0.9148 | 0.1014 | 0.055 |

**Selected: Random Forest.**

* **Primary metric: CV Top-1 on reliable points.** It measures the route-matching objective and is
  computed on the training trips only, so the selection does not tune on the test set. RF is first.
* **Log loss as second criterion,** because the probabilities become HMM emissions. RF is best in all
  5 folds and on student-test.
* **Stability.** RF has the best macro-F1 in 4 of 5 folds, with fold std 0.017.
* **Generalisation.** RF is the best student-test macro-F1 and log loss. MLP's student-test Top-1 is
  0.06 pp higher (≈ 11 of 19,160 points), which is noise, and MLP is last on CV Top-1.
* **Over-fitting.** Train − CV macro-F1 is +0.034 (deep trees). CV and test agree, so this is mild.
  Bayesian tuning will target `max_depth`, `min_samples_leaf` and `max_features`.
* **Why the others lose.**
  * KNN: worst log loss (coarse vote probabilities) and ~40× slower inference.
  * LR: linear, so it cannot use interactions such as "heading only matters when moving".
  * HGB: a close runner-up, with worse calibration.

## Reproduce

```bash
python models/run_experiments.py     # writes results/model_comparison.csv, cv_per_fold.csv, configs.json,
                                     # mlp_loss_curve.png, models/saved/*.joblib
python results/plot_model_comparison.py
python models/select_model.py
```

`models/saved/*.joblib` (trained pipelines) are produced by the run. They are not committed yet: the
Random Forest file is large, so the final selected model will be committed or regenerated for the API.
