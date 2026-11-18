"""
Generate Model Comparison Plots & Tables for Task D & Report Section 3
Plots:
1. Cross-validation Macro F1 across models
2. Student-test Top-1 Accuracy and Macro F1
3. Log Loss comparison across models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = r"C:\Users\Arnav Sharma\.gemini\antigravity\scratch\ML_HACKATHON_TEAM_8"
CSV_PATH = os.path.join(PROJECT_DIR, "results", "model_comparison.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

df = pd.read_csv(CSV_PATH)

# Clean model names for display
model_names = {
    "logreg": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "rf": "Random Forest",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "mlp": "Multi-Layer Perceptron (MLP)",
    "baseline_nearest_arc": "Baseline (Nearest Arc)",
    "baseline_nearest_arc_heading": "Baseline (Nearest + Heading)",
    "baseline_A_rule": "Baseline (Heuristic Rule)"
}

df["model_display"] = df["model"].map(model_names).fillna(df["model"])

# Filter out baselines for pure model comparison or include them as benchmarks
models_only_cv = df[(df["split"] == "cv_mean_train80") & (~df["family"].str.contains("baseline"))].copy()
models_only_test = df[(df["split"] == "student_test10") & (~df["family"].str.contains("baseline"))].copy()

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# 1. CV Macro F1 Comparison
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=models_only_cv, x="model_display", y="macro_f1", palette="Blues_r")
plt.title("5-Fold Cross-Validation Macro F1-Score Across Models")
plt.ylabel("Macro F1-Score (CV Mean)")
plt.xlabel("Model Family")
plt.ylim(0.85, 0.95)
plt.xticks(rotation=20, ha="right")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cv_macro_f1_comparison.png"), dpi=300)
plt.close()
print("Saved cv_macro_f1_comparison.png")

# 2. Student Test Top-1 Accuracy and Macro F1 Comparison
fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(models_only_test))
width = 0.35

rects1 = ax1.bar(x - width/2, models_only_test["top1_all_points"], width, label="Top-1 Accuracy", color="#1f77b4")
rects2 = ax1.bar(x + width/2, models_only_test["macro_f1"], width, label="Macro F1-Score", color="#2ca02c")

ax1.set_ylabel("Score")
ax1.set_title("Student-Testing Performance: Top-1 Accuracy & Macro F1-Score")
ax1.set_xticks(x)
ax1.set_xticklabels(models_only_test["model_display"], rotation=20, ha="right")
ax1.set_ylim(0.80, 0.95)
ax1.legend(loc="upper left")

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2., h),
                 ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2., h),
                 ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "student_test_top1_and_macro_f1.png"), dpi=300)
plt.close()
print("Saved student_test_top1_and_macro_f1.png")

# 3. Log Loss Comparison (lower is better)
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=models_only_test, x="model_display", y="log_loss", palette="Oranges_r")
plt.title("Student-Testing Log Loss Comparison (Lower is Better)")
plt.ylabel("Binary Log Loss / Cross-Entropy")
plt.xlabel("Model Family")
plt.xticks(rotation=20, ha="right")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "student_test_log_loss_comparison.png"), dpi=300)
plt.close()
print("Saved student_test_log_loss_comparison.png")

# 4. Generate Section 3.1 Consolidated Comparison Table
table_df = models_only_test[["model_display", "family", "key_config"]].copy()
# Merge CV macro F1
cv_map = dict(zip(models_only_cv["model"], models_only_cv["macro_f1"]))
test_map_top1 = dict(zip(models_only_test["model"], models_only_test["top1_all_points"]))
test_map_f1 = dict(zip(models_only_test["model"], models_only_test["macro_f1"]))
test_map_loss = dict(zip(models_only_test["model"], models_only_test["log_loss"]))
test_map_route = dict(zip(models_only_test["model"], models_only_test["route_continuity"]))
test_map_time = dict(zip(models_only_test["model"], models_only_test["predict_time_s"]))

summary_rows = []
for idx, r in models_only_test.iterrows():
    m = r["model"]
    summary_rows.append({
        "Model": r["model_display"],
        "Algorithm Family": r["family"],
        "Key Configuration": r["key_config"],
        "CV Macro F1": round(cv_map.get(m, np.nan), 4),
        "Top-1 Accuracy": round(r["top1_all_points"], 4),
        "Route Continuity": round(r["route_continuity"], 4),
        "Student-Test Macro F1": round(r["macro_f1"], 4),
        "Log Loss": round(r["log_loss"], 4),
        "Inference Time (s)": round(r["predict_time_s"], 3)
    })

df_summary = pd.DataFrame(summary_rows)
# Save CSV
df_summary.to_csv(os.path.join(RESULTS_DIR, "section_3_model_comparison_table.csv"), index=False)

# Format markdown manually without needing tabulate
headers = list(df_summary.columns)
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for _, row in df_summary.iterrows():
    lines.append("| " + " | ".join(str(val) for val in row) + " |")
md_table = "\n".join(lines)

with open(os.path.join(RESULTS_DIR, "section_3_model_comparison_table.md"), "w") as f:
    f.write("# Section 3.1 Consolidated Model Comparison Table\n\n" + md_table + "\n")
print("Saved section_3_model_comparison_table.md and .csv")
