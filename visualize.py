import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#F8F9FA", "axes.facecolor": "#F8F9FA",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
COLORS = ["#1F497D", "#E05C2A", "#2E8B57", "#8B2FC9", "#C9A02F"]

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCE PIPELINE (same logic as crime_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    r"data\crimedata_csv_AllNeighbourhoods_AllYears.csv",
    usecols=["TYPE", "YEAR", "NEIGHBOURHOOD"]
)
df.dropna(subset=["NEIGHBOURHOOD"], inplace=True)
df = df[df["NEIGHBOURHOOD"].str.strip() != ""]

WEIGHTS = {
    "Homicide": 10, "Offence Against a Person": 8,
    "Vehicle Collision or Pedestrian Struck (with Fatality)": 8,
    "Robbery": 7, "Break and Enter Commercial": 6,
    "Break and Enter Residential/Other": 6,
    "Vehicle Collision or Pedestrian Struck (with Injury)": 5,
    "Assault": 5, "Theft of Vehicle": 4, "Theft from Vehicle": 3,
    "Other Theft": 2, "Theft of Bicycle": 2, "Mischief": 1,
}
df["WEIGHT"] = df["TYPE"].map(WEIGHTS).fillna(2)

agg = (
    df.groupby(["NEIGHBOURHOOD", "YEAR"])
    .agg(crime_count=("TYPE", "count"), weighted_score=("WEIGHT", "sum"))
    .reset_index()
)
min_ws = agg["weighted_score"].min()
max_ws = agg["weighted_score"].max()
agg["safety_score"] = 100 * (1 - (agg["weighted_score"] - min_ws) / (max_ws - min_ws))

agg = agg.sort_values(["NEIGHBOURHOOD", "YEAR"]).reset_index(drop=True)

le = LabelEncoder()
agg["neighbourhood_enc"] = le.fit_transform(agg["NEIGHBOURHOOD"])

# Time-based split: train on 2003-2019, test on 2020+
SPLIT_YEAR = 2020
FEATURES   = ["neighbourhood_enc", "YEAR", "crime_count"]
train_mask = agg["YEAR"] < SPLIT_YEAR
test_mask  = agg["YEAR"] >= SPLIT_YEAR

X = agg[FEATURES]
y = agg["safety_score"]
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

models = {
    "Linear Regression":  LinearRegression(),
    "Decision Tree":      DecisionTreeRegressor(random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
}
results, predictions = {}, {}
for name, m in models.items():
    m.fit(X_train, y_train)
    yp = m.predict(X_test)
    predictions[name] = yp
    results[name] = {"MSE": mean_squared_error(y_test, yp),
                     "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
                     "R2":   r2_score(y_test, yp)}

import os
os.makedirs("outputs/plots/visualizations", exist_ok=True)

print("Data ready. Generating plots...\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Raw Data Overview  (Steps 1-2)
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle("Figure 1 — Raw Data Overview", fontsize=13, fontweight="bold", y=1.01)

# 1a. Total crimes per year
crimes_per_year = df.groupby("YEAR").size()
axes[0].bar(crimes_per_year.index, crimes_per_year.values, color=COLORS[0], alpha=0.85)
axes[0].set_title("Total Crimes per Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Number of Crimes")

# 1b. Crime type distribution (top 10)
top_types = df["TYPE"].value_counts().head(10)
# Fix readability: split stride+slice into two steps
reversed_index  = top_types.index[::-1]
reversed_values = top_types.values[::-1]
axes[1].barh(reversed_index, reversed_values, color=COLORS[1], alpha=0.85)
axes[1].set_title("Top 10 Crime Types")
axes[1].set_xlabel("Count")

# 1c. Crimes per neighbourhood (total across all years)
crimes_per_neigh = df.groupby("NEIGHBOURHOOD").size().sort_values(ascending=True)
axes[2].barh(crimes_per_neigh.index, crimes_per_neigh.values, color=COLORS[2], alpha=0.85)
axes[2].set_title("Total Crimes per Neighbourhood")
axes[2].set_xlabel("Count")

plt.tight_layout()
plt.savefig("outputs/plots/visualizations/fig1_raw_data.png", dpi=150, bbox_inches="tight")
print("Saved outputs/plots/visualizations/fig1_raw_data.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Crime Type Weights  (Step 3)
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 5))
fig2.suptitle("Figure 2 — Crime Type Severity Weights", fontsize=13, fontweight="bold")

sorted_w  = dict(sorted(WEIGHTS.items(), key=lambda x: x[1]))
bar_colors = [COLORS[0] if v >= 7 else COLORS[1] if v >= 4 else COLORS[2]
              for v in sorted_w.values()]
bars = ax.barh(list(sorted_w.keys()), list(sorted_w.values()), color=bar_colors, alpha=0.85)
ax.bar_label(bars, padding=3)
ax.set_xlabel("Severity Weight")
ax.set_xlim(0, 12)
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color=COLORS[0], alpha=0.85, label="High (7-10)"),
    plt.Rectangle((0,0),1,1, color=COLORS[1], alpha=0.85, label="Medium (4-6)"),
    plt.Rectangle((0,0),1,1, color=COLORS[2], alpha=0.85, label="Low (1-3)"),
], loc="lower right")

plt.tight_layout()
plt.savefig("outputs/plots/visualizations/fig2_weights.png", dpi=150, bbox_inches="tight")
print("Saved outputs/plots/visualizations/fig2_weights.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Feature Engineering  (Steps 4-6)
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle("Figure 3 — Feature Engineering", fontsize=13, fontweight="bold", y=1.01)

# 3a. Safety score distribution
axes[0].hist(agg["safety_score"], bins=30, color=COLORS[0], alpha=0.85, edgecolor="white")
axes[0].set_title("Safety Score Distribution")
axes[0].set_xlabel("Safety Score (0-100)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(agg["safety_score"].mean(), color=COLORS[1], linestyle="--",
                label=f"Mean: {agg['safety_score'].mean():.1f}")
axes[0].legend()

# 3b. Weighted score vs crime count scatter
axes[1].scatter(agg["crime_count"], agg["weighted_score"],
                alpha=0.3, s=15, color=COLORS[2])
axes[1].set_title("Crime Count vs Weighted Score")
axes[1].set_xlabel("Crime Count")
axes[1].set_ylabel("Weighted Score")

# 3c. Average safety score per neighbourhood (heatmap-style bar)
avg_safety = agg.groupby("NEIGHBOURHOOD")["safety_score"].mean().sort_values()
bar_cols    = [COLORS[0] if v >= 75 else COLORS[1] if v >= 50 else COLORS[3]
               for v in avg_safety.values]
axes[2].barh(avg_safety.index, avg_safety.values, color=bar_cols, alpha=0.85)
axes[2].axvline(75, color="green",  linestyle="--", linewidth=1, label="Safe (75)")
axes[2].axvline(50, color="orange", linestyle="--", linewidth=1, label="Moderate (50)")
axes[2].set_title("Avg Safety Score by Neighbourhood")
axes[2].set_xlabel("Average Safety Score")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/plots/visualizations/fig3_features.png", dpi=150, bbox_inches="tight")
print("Saved outputs/plots/visualizations/fig3_features.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Safety Score Trends Over Time  (Step 4-5)
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle("Figure 4 — Safety Score Trends Over Time", fontsize=13, fontweight="bold", y=1.01)

# 4a. Top 5 and bottom 5 neighbourhoods trend lines
avg_by_neigh = agg.groupby("NEIGHBOURHOOD")["safety_score"].mean()
top5    = avg_by_neigh.nlargest(5).index.tolist()
bottom5 = avg_by_neigh.nsmallest(5).index.tolist()

for i, n in enumerate(top5):
    d = agg[agg["NEIGHBOURHOOD"] == n].sort_values("YEAR")
    axes[0].plot(d["YEAR"], d["safety_score"], marker="o", markersize=3,
                 label=n, color=COLORS[i % len(COLORS)])
axes[0].set_title("Top 5 Safest Neighbourhoods Over Time")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Safety Score")
axes[0].legend(fontsize=7)

for i, n in enumerate(bottom5):
    d = agg[agg["NEIGHBOURHOOD"] == n].sort_values("YEAR")
    axes[1].plot(d["YEAR"], d["safety_score"], marker="o", markersize=3,
                 label=n, color=COLORS[i % len(COLORS)])
axes[1].set_title("Bottom 5 Most Dangerous Neighbourhoods Over Time")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Safety Score")
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig("outputs/plots/visualizations/fig4_trends.png", dpi=150, bbox_inches="tight")
print("Saved outputs/plots/visualizations/fig4_trends.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Train/Test Split  (Step 9)
# ─────────────────────────────────────────────────────────────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(14, 5))
fig5.suptitle("Figure 5 — Train / Test Split", fontsize=13, fontweight="bold", y=1.01)

# 5a. Pie chart
axes[0].pie([len(X_train), len(X_test)], labels=[f"Train (2003-{SPLIT_YEAR-1})", f"Test ({SPLIT_YEAR}+)"],
            colors=[COLORS[0], COLORS[1]], autopct="%1.0f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Dataset Split")

# 5b. Safety score distribution in train vs test
axes[1].hist(y_train, bins=25, alpha=0.6, color=COLORS[0], label=f"Train (n={len(y_train)})")
axes[1].hist(y_test,  bins=25, alpha=0.6, color=COLORS[1], label=f"Test  (n={len(y_test)})")
axes[1].set_title("Safety Score Distribution: Train vs Test")
axes[1].set_xlabel("Safety Score")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/plots/visualizations/fig5_split.png", dpi=150, bbox_inches="tight")
print("Saved outputs/plots/visualizations/fig5_split.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Model Results  (Steps 10-12)
# ─────────────────────────────────────────────────────────────────────────────
fig6 = plt.figure(figsize=(18, 10))
fig6.suptitle("Figure 6 — Model Results", fontsize=13, fontweight="bold")
gs   = gridspec.GridSpec(2, 3, figure=fig6, hspace=0.4, wspace=0.35)

model_names = list(results.keys())

# 6a-c. Predicted vs Actual for each model
for i, (name, yp) in enumerate(predictions.items()):
    ax = fig6.add_subplot(gs[0, i])
    ax.scatter(y_test, yp, alpha=0.35, s=15, color=COLORS[i])
    lims = [min(y_test.min(), yp.min()), max(y_test.max(), yp.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual Safety Score")
    ax.set_ylabel("Predicted Safety Score")
    ax.set_title(f"{name}\nRMSE={results[name]['RMSE']:.4f}  R²={results[name]['R2']:.4f}")
    ax.legend(fontsize=7)

# 6d. RMSE comparison bar chart
ax_rmse = fig6.add_subplot(gs[1, 0])
rmse_vals = [results[n]["RMSE"] for n in model_names]
bars = ax_rmse.bar(model_names, rmse_vals, color=COLORS[:3], alpha=0.85, edgecolor="white")
ax_rmse.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
ax_rmse.set_title("RMSE Comparison (lower = better)")
ax_rmse.set_ylabel("RMSE")
ax_rmse.set_ylim(0, max(rmse_vals) * 1.3)

# 6e. R² comparison bar chart
ax_r2 = fig6.add_subplot(gs[1, 1])
r2_vals = [results[n]["R2"] for n in model_names]
bars = ax_r2.bar(model_names, r2_vals, color=COLORS[:3], alpha=0.85, edgecolor="white")
ax_r2.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
ax_r2.set_title("R² Comparison (higher = better)")
ax_r2.set_ylabel("R²")
ax_r2.set_ylim(min(r2_vals) * 0.999, 1.001)

# 6f. Residuals for best model
best   = min(results, key=lambda n: results[n]["RMSE"])
resid  = y_test.values - predictions[best]
ax_res = fig6.add_subplot(gs[1, 2])
ax_res.hist(resid, bins=30, color=COLORS[3], alpha=0.85, edgecolor="white")
ax_res.axvline(0, color="red", linestyle="--", linewidth=1.2)
ax_res.set_title(f"Residuals — {best}")
ax_res.set_xlabel("Residual (Actual - Predicted)")
ax_res.set_ylabel("Frequency")

plt.savefig("outputs/plots/visualizations/fig6_model_results.png", dpi=150, bbox_inches="tight")
print("Saved outputs/plots/visualizations/fig6_model_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Safety Score Heatmap (Neighbourhood x Year)
# ─────────────────────────────────────────────────────────────────────────────
pivot = agg.pivot_table(index="NEIGHBOURHOOD", columns="YEAR", values="safety_score")

fig7, ax = plt.subplots(figsize=(20, 8))
fig7.suptitle("Figure 7 — Safety Score Heatmap (Neighbourhood x Year)",
              fontsize=13, fontweight="bold")

im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_xlabel("Year")
ax.set_ylabel("Neighbourhood")
plt.colorbar(im, ax=ax, label="Safety Score (0=Dangerous, 100=Safe)")

plt.tight_layout()
plt.savefig("outputs/plots/visualizations/fig7_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved outputs/plots/visualizations/fig7_heatmap.png")

plt.show()
print("\nAll 7 figures saved to outputs/plots/visualizations/")
