import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── 1. LOAD DATA ─────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
df = pd.read_csv(
    _ROOT / "data" / "crimedata_csv_AllNeighbourhoods_AllYears.csv",
    usecols=["TYPE", "YEAR", "NEIGHBOURHOOD"]
)

# ── 2. CLEAN DATA ────────────────────────────────────────────
df.dropna(subset=["NEIGHBOURHOOD"], inplace=True)
df = df[df["NEIGHBOURHOOD"].str.strip() != ""]

# ── 3. ASSIGN WEIGHTS ────────────────────────────────────────
WEIGHTS = {
    "Homicide": 10,
    "Offence Against a Person": 8,
    "Vehicle Collision or Pedestrian Struck (with Fatality)": 8,
    "Robbery": 7,
    "Break and Enter Commercial": 6,
    "Break and Enter Residential/Other": 6,
    "Vehicle Collision or Pedestrian Struck (with Injury)": 5,
    "Assault": 5,
    "Theft of Vehicle": 4,
    "Theft from Vehicle": 3,
    "Other Theft": 2,
    "Theft of Bicycle": 2,
    "Mischief": 1,
}
DEFAULT_WEIGHT = 2
df["WEIGHT"] = df["TYPE"].map(WEIGHTS).fillna(DEFAULT_WEIGHT)

# ── 4. AGGREGATE DATA ────────────────────────────────────────
agg = (
    df.groupby(["NEIGHBOURHOOD", "YEAR"])
    .agg(crime_count=("TYPE", "count"), weighted_score=("WEIGHT", "sum"))
    .reset_index()
)

# ── 5. CREATE SAFETY SCORE (TARGET) ──────────────────────────
min_ws = agg["weighted_score"].min()
max_ws = agg["weighted_score"].max()
agg["safety_score"] = 100 * (1 - (agg["weighted_score"] - min_ws) / (max_ws - min_ws))

# ── 6. ENCODE NEIGHBOURHOOD ──────────────────────────────────
le = LabelEncoder()
agg["neighbourhood_enc"] = le.fit_transform(agg["NEIGHBOURHOOD"])

# ── 7. BUILD ROLLING FEATURES (aligned with query.py/chatbot.py/gui.py) ──────
agg = agg.sort_values(["NEIGHBOURHOOD", "YEAR"]).reset_index(drop=True)
agg["rolling_avg_score"] = (
    agg.groupby("NEIGHBOURHOOD")["safety_score"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
agg["prev_year_score"] = agg.groupby("NEIGHBOURHOOD")["safety_score"].shift(1)
agg["crime_trend"] = agg.groupby("NEIGHBOURHOOD")["crime_count"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
agg = agg.dropna(subset=["prev_year_score"])

# ── 8. DEFINE FEATURES ───────────────────────────────────────
FEATURES = ["neighbourhood_enc", "YEAR", "prev_year_score", "rolling_avg_score", "crime_trend"]
TARGET   = "safety_score"

X = agg[FEATURES]
y = agg[TARGET]

# ── 9. TIME-BASED SPLIT (train: 2003-2019, test: 2020+) ──────
SPLIT_YEAR = 2020
train_mask = agg["YEAR"] < SPLIT_YEAR
test_mask  = agg["YEAR"] >= SPLIT_YEAR

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train: {len(X_train)} samples (2003-{SPLIT_YEAR-1})")
print(f"Test:  {len(X_test)} samples ({SPLIT_YEAR}-{agg['YEAR'].max()})")

# ── 10. TRAIN MODELS ─────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
}
results     = {}
predictions = {}

# ── 11. EVALUATE ─────────────────────────────────────────────
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    mse  = mean_squared_error(y_test, y_pred)
    results[name] = {"MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_test, y_pred)}

# ── 12. RESULTS TABLE ────────────────────────────────────────
results_df = pd.DataFrame(results).T.round(4)
print("\n" + "=" * 55)
print("         MODEL COMPARISON TABLE")
print("=" * 55)
print(results_df.to_string())
print("=" * 55)

best_model = results_df["RMSE"].idxmin()
print(f"\n>> Best model by RMSE : {best_model}")
print(f"   RMSE = {results_df.loc[best_model, 'RMSE']:.4f}  |  "
      f"R2 = {results_df.loc[best_model, 'R2']:.4f}")

# ── 13. SAVE BEST MODEL ──────────────────────────────────────
os.makedirs("outputs/plots/pipeline", exist_ok=True)
os.makedirs("model", exist_ok=True)
joblib.dump(models[best_model],                    "model/best_model.pkl")
joblib.dump(le,                                    "model/label_encoder.pkl")
joblib.dump({"min_ws": min_ws, "max_ws": max_ws},  "model/score_params.pkl")
print(f"Best model ({best_model}) saved to model/")

# ── 14. PLOT PREDICTED VS ACTUAL ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Predicted vs Actual Safety Score", fontsize=14, fontweight="bold")
for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.4)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name}\nRMSE={results[name]['RMSE']:.2f}, R2={results[name]['R2']:.3f}")
plt.tight_layout()
plt.savefig("outputs/plots/pipeline/predicted_vs_actual.png", dpi=150)
plt.show()
print("Plot saved to outputs/plots/pipeline/predicted_vs_actual.png")
