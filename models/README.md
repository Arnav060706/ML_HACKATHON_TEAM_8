# Model Training & Evaluation (Task B)

This directory contains the training harness, model artifacts, and evaluation scripts for the PathSnap map-matching classifier.

## Overview
The goal is to score candidate road segments within radius $R$ for each GPS observation point and select the true road arc traversed by the vehicle.

## Key Guidelines
1. **Validation Split:** 5-fold `GroupKFold` grouped strictly by `trip_id` across the 80 training trips to prevent spatial and trajectory leakage.
2. **Filtering:** Train models strictly on candidates where `label_reliable == 1` (filtering out points with disconnected routes or severe multipath > 100m from road network).
3. **Candidate Model Families (Minimum 5 required):**
   - Logistic Regression (Linear baseline)
   - K-Nearest Neighbors (KNN)
   - Random Forest Classifier (Ensemble bagging)
   - LightGBM / XGBoost (Gradient boosted decision trees)
   - Multi-Layer Perceptron (MLP Neural Network with loss curve)
4. **Evaluation Metrics:**
   - Top-1 Accuracy: Best scoring candidate matching the true road arc for each GPS observation.
   - Macro F1-score & Binary F1-score.
   - Log Loss / Binary Cross-Entropy.
   - Inference latency per observation (critical for real-time map matching).
