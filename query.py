import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
LE_PATH    = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
SCORE_PATH = os.path.join(BASE_DIR, "model", "score_params.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "crimedata_csv_AllNeighbourhoods_AllYears.csv")

# ── Load & prepare data ───────────────────────────────────────────────────────
print("Loading data and model...")

df = pd.read_csv(DATA_PATH, usecols=["TYPE", "YEAR", "NEIGHBOURHOOD"])
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

# Load saved normalisation params or compute from data
if os.path.exists(SCORE_PATH):
    score_params = joblib.load(SCORE_PATH)
    min_ws, max_ws = score_params["min_ws"], score_params["max_ws"]
else:
    min_ws, max_ws = agg["weighted_score"].min(), agg["weighted_score"].max()

agg["safety_score"] = 100 * (1 - (agg["weighted_score"] - min_ws) / (max_ws - min_ws))

# Load saved label encoder or fit a new one
if os.path.exists(LE_PATH):
    le = joblib.load(LE_PATH)
    agg["neighbourhood_enc"] = le.transform(agg["NEIGHBOURHOOD"])
else:
    le = LabelEncoder()
    agg["neighbourhood_enc"] = le.fit_transform(agg["NEIGHBOURHOOD"])

agg = agg.sort_values(["NEIGHBOURHOOD", "YEAR"]).reset_index(drop=True)
agg["rolling_avg_score"] = (
    agg.groupby("NEIGHBOURHOOD")["safety_score"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
agg["prev_year_score"] = agg.groupby("NEIGHBOURHOOD")["safety_score"].shift(1)
agg["crime_trend"] = agg.groupby("NEIGHBOURHOOD")["crime_count"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# Load the best saved model (Random Forest) or retrain as fallback
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Loaded saved Random Forest model.")
else:
    print("No saved model found — retraining...")
    from sklearn.ensemble import RandomForestRegressor
    train_data = agg.dropna(subset=["prev_year_score"])
    FEATURES   = ["neighbourhood_enc", "YEAR", "prev_year_score", "rolling_avg_score", "crime_trend"]
    model      = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train_data[FEATURES], train_data["safety_score"])

NEIGHBOURHOODS = sorted(agg["NEIGHBOURHOOD"].unique())
MAX_HIST_YEAR  = int(agg["YEAR"].max())

# ── Helpers ───────────────────────────────────────────────────────────────────
def divider():
    print("-" * 56)

def match_neighbourhood(query):
    q = query.strip().lower()
    exact = [n for n in NEIGHBOURHOODS if n.lower() == q]
    if exact:
        return exact[0]
    partial = [n for n in NEIGHBOURHOODS if q in n.lower()]
    return partial[0] if len(partial) == 1 else None

def safety_label(score):
    if score >= 75: return "SAFE"
    if score >= 50: return "MODERATE"
    if score >= 25: return "HIGH CRIME"
    return "VERY HIGH CRIME"

def get_historical(neighbourhood, year):
    row = agg[(agg["NEIGHBOURHOOD"] == neighbourhood) & (agg["YEAR"] == year)]
    return None if row.empty else row.iloc[0]

def predict_future(neighbourhood, target_year):
    enc          = le.transform([neighbourhood])[0]
    neigh_data   = agg[agg["NEIGHBOURHOOD"] == neighbourhood].sort_values("YEAR")
    known_scores = list(neigh_data["safety_score"].values[-3:])
    known_crimes = list(neigh_data["crime_count"].values[-3:])
    current_year = MAX_HIST_YEAR
    while current_year < target_year:
        current_year += 1
        X = pd.DataFrame([{
            "neighbourhood_enc": enc,
            "YEAR":              current_year,
            "prev_year_score":   known_scores[-1],
            "rolling_avg_score": float(np.mean(known_scores[-3:])),
            "crime_trend":       float(np.mean(known_crimes[-3:])),
        }])
        s = float(np.clip(model.predict(X)[0], 0, 100))
        known_scores.append(s)
        known_crimes.append(float(np.mean(known_crimes[-3:])))
    return s, current_year

# ── Main loop ─────────────────────────────────────────────────────────────────
print("Model ready.\n")
print("=" * 56)
print("  Vancouver Neighbourhood Crime & Safety Query Tool")
print("=" * 56)
print(f"  Historical data : 2003 – {MAX_HIST_YEAR}")
print(f"  Future years    : {MAX_HIST_YEAR + 1} onwards (ML prediction)")
print("  Type 'list' to see all neighbourhoods | 'quit' to exit")
print("=" * 56 + "\n")

while True:
    neighbourhood_input = input("Enter neighbourhood: ").strip()

    if neighbourhood_input.lower() == "quit":
        print("Goodbye!")
        break

    if neighbourhood_input.lower() == "list":
        print("\nAvailable neighbourhoods:")
        for n in NEIGHBOURHOODS:
            print(f"  - {n}")
        print()
        continue

    neighbourhood = match_neighbourhood(neighbourhood_input)
    if not neighbourhood:
        print(f"  '{neighbourhood_input}' not found. Try 'list' to see options.\n")
        continue

    year_input = input("Enter year: ").strip()
    if year_input.lower() == "quit":
        print("Goodbye!")
        break

    try:
        year = int(year_input)
    except ValueError:
        print("  Invalid year. Please enter a number.\n")
        continue

    if year < 2003:
        print("  No data available before 2003.\n")
        continue

    print()
    divider()

    # ── Historical year ───────────────────────────────────────────────────────
    if year <= MAX_HIST_YEAR:
        row = get_historical(neighbourhood, year)
        if row is None:
            print(f"  No records found for {neighbourhood} in {year}.\n")
            continue

        crime_count  = int(row["crime_count"])
        safety_score = float(row["safety_score"])
        breakdown    = (
            df[(df["NEIGHBOURHOOD"] == neighbourhood) & (df["YEAR"] == year)]
            .groupby("TYPE").size().sort_values(ascending=False)
            .reset_index(name="count")
        )
        year_rank = (
            agg[agg["YEAR"] == year]
            .sort_values("safety_score", ascending=False)
            .reset_index(drop=True)
        )
        rank  = year_rank[year_rank["NEIGHBOURHOOD"] == neighbourhood].index[0] + 1
        total = len(year_rank)

        print(f"  {neighbourhood.upper()}  |  {year}  [HISTORICAL]")
        divider()
        print(f"  Total crimes   : {crime_count}")
        print(f"  Safety score   : {safety_score:.1f} / 100  [{safety_label(safety_score)}]")
        print(f"  Rank (safest)  : #{rank} out of {total} neighbourhoods in {year}")
        divider()
        print("  Crime breakdown:")
        for _, r in breakdown.iterrows():
            print(f"    {r['TYPE']:<50} {r['count']}")

    # ── Future year ───────────────────────────────────────────────────────────
    else:
        years_ahead = year - MAX_HIST_YEAR
        print(f"  {neighbourhood.upper()}  |  {year}  [PREDICTED]")
        divider()
        print(f"  Years predicted ahead  : {years_ahead}")
        print(f"  Model used             : Random Forest (best model, trained on 2003-{MAX_HIST_YEAR})")
        divider()
        print("  Year-by-year forecast:")

        enc          = le.transform([neighbourhood])[0]
        neigh_data   = agg[agg["NEIGHBOURHOOD"] == neighbourhood].sort_values("YEAR")
        known_scores = list(neigh_data["safety_score"].values[-3:])
        known_crimes = list(neigh_data["crime_count"].values[-3:])
        cur_year     = MAX_HIST_YEAR

        while cur_year < year:
            cur_year += 1
            X = pd.DataFrame([{
                "neighbourhood_enc": enc,
                "YEAR":              cur_year,
                "prev_year_score":   known_scores[-1],
                "rolling_avg_score": float(np.mean(known_scores[-3:])),
                "crime_trend":       float(np.mean(known_crimes[-3:])),
            }])
            s = float(np.clip(model.predict(X)[0], 0, 100))
            print(f"    {cur_year}  ->  {s:.1f} / 100  [{safety_label(s)}]")
            known_scores.append(s)
            known_crimes.append(float(np.mean(known_crimes[-3:])))

        print(f"\n  Final predicted score  : {s:.1f} / 100  [{safety_label(s)}]")

    divider()
    print()
