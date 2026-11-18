# Section 3.1 Consolidated Model Comparison Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | linear | StandardScaler + LogisticRegression | 0.8997 | 0.8539 | 0.929 | 0.9103 | 0.1238 | 0.028 |
| K-Nearest Neighbors | instance-based | StandardScaler + KNN k=25, fit on random subsample <= 100000 rows | 0.8975 | 0.8557 | 0.922 | 0.9108 | 0.2052 | 13.547 |
| Random Forest | bagged trees | RandomForest (no scaling needed) | 0.9073 | 0.8636 | 0.9302 | 0.918 | 0.0966 | 0.354 |
| Hist Gradient Boosting | boosted trees | HistGradientBoostingClassifier (lightgbm not installed) (no scaling needed) | 0.9055 | 0.8602 | 0.9178 | 0.9161 | 0.1414 | 0.447 |
| Multi-Layer Perceptron (MLP) | neural network | StandardScaler + MLP (64,32), early_stopping, max_iter=200, fit on random subsample <= 400000 rows (time cap) | 0.896 | 0.8641 | 0.9333 | 0.9148 | 0.1014 | 0.055 |
