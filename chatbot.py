import re
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import ollama

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ENGINE
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data and training model...")

df = pd.read_csv(
    r"archive\crimedata_csv_AllNeighbourhoods_AllYears.csv",
    usecols=["TYPE", "YEAR", "NEIGHBOURHOOD"]
)
df.dropna(subset=["NEIGHBOURHOOD"], inplace=True)
df = df[df["NEIGHBOURHOOD"].str.strip() != ""]

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
df["WEIGHT"] = df["TYPE"].map(WEIGHTS).fillna(2)

agg = (
    df.groupby(["NEIGHBOURHOOD", "YEAR"])
    .agg(crime_count=("TYPE", "count"), weighted_score=("WEIGHT", "sum"))
    .reset_index()
)
min_ws = agg["weighted_score"].min()
max_ws = agg["weighted_score"].max()
agg["safety_score"] = 100 * (1 - (agg["weighted_score"] - min_ws) / (max_ws - min_ws))

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

train_data = agg.dropna(subset=["prev_year_score"])
FEATURES   = ["neighbourhood_enc", "YEAR", "prev_year_score", "rolling_avg_score", "crime_trend"]
rf_model   = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(train_data[FEATURES], train_data["safety_score"])

NEIGHBOURHOODS = sorted(agg["NEIGHBOURHOOD"].unique())
MAX_YEAR       = int(agg["YEAR"].max())

print(f"Ready. {len(NEIGHBOURHOODS)} neighbourhoods, 2003-{MAX_YEAR}.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _safety_label(score):
    if score >= 75: return "Safe"
    if score >= 50: return "Moderate"
    if score >= 25: return "High Crime"
    return "Very High Crime"

def _match(query):
    """Fuzzy-match a neighbourhood name."""
    q = query.strip().lower()
    # exact
    for n in NEIGHBOURHOODS:
        if n.lower() == q:
            return n
    # partial
    matches = [n for n in NEIGHBOURHOODS if q in n.lower()]
    if len(matches) == 1:
        return matches[0]
    # word overlap
    q_words = set(q.split())
    scored  = [(n, len(q_words & set(n.lower().split()))) for n in NEIGHBOURHOODS]
    scored  = [(n, s) for n, s in scored if s > 0]
    if scored:
        return max(scored, key=lambda x: x[1])[0]
    return None

def _predict_chain(neighbourhood, target_year):
    enc    = int(le.transform([neighbourhood])[0])
    nd     = agg[agg["NEIGHBOURHOOD"] == neighbourhood].sort_values("YEAR")
    scores = list(nd["safety_score"].values[-3:])
    crimes = list(nd["crime_count"].values[-3:])
    cur    = MAX_YEAR
    yearly = {}
    while cur < target_year:
        cur += 1
        X = pd.DataFrame([{
            "neighbourhood_enc": enc,
            "YEAR":              cur,
            "prev_year_score":   scores[-1],
            "rolling_avg_score": float(np.mean(scores[-3:])),
            "crime_trend":       float(np.mean(crimes[-3:])),
        }])
        s = float(np.clip(rf_model.predict(X)[0], 0, 100))
        yearly[cur] = round(s, 2)
        scores.append(s)
        crimes.append(float(np.mean(crimes[-3:])))
    return yearly

def _sanitize_neighbourhood(name):
    """Validate neighbourhood against the known whitelist to prevent injection."""
    matched = _match(name)
    if matched not in NEIGHBOURHOODS:
        return None
    return matched

def _sanitize_year(year):
    """Ensure year is a plain integer within a safe range."""
    return int(year) if 1900 <= int(year) <= 2100 else None

def get_neighbourhood_stats(neighbourhood, year):
    n    = _sanitize_neighbourhood(neighbourhood)
    year = _sanitize_year(year)
    if not n:
        return {"error": f"Neighbourhood '{neighbourhood}' not found.", "available": NEIGHBOURHOODS}
    if not year or year < 2003:
        return {"error": "No data before 2003."}
    if year <= MAX_YEAR:
        # filter using sanitized whitelist value, not raw user input
        mask = (agg["NEIGHBOURHOOD"] == n) & (agg["YEAR"] == year)
        row  = agg[mask]
        if row.empty:
            return {"error": f"No records for {n} in {year}."}
        r            = row.iloc[0]
        df_mask      = (df["NEIGHBOURHOOD"] == n) & (df["YEAR"] == year)
        breakdown    = df[df_mask].groupby("TYPE").size().sort_values(ascending=False).to_dict()
        yr_data      = agg[agg["YEAR"] == year].sort_values("safety_score", ascending=False).reset_index(drop=True)
        rank         = int(yr_data[yr_data["NEIGHBOURHOOD"] == n].index[0]) + 1
        return {
            "neighbourhood": n, "year": year, "type": "historical",
            "crime_count":   int(r["crime_count"]),
            "safety_score":  round(float(r["safety_score"]), 2),
            "safety_label":  _safety_label(float(r["safety_score"])),
            "rank_safest":   rank,
            "total_neighbourhoods": len(yr_data),
            "crime_breakdown": breakdown,
        }
    else:
        chain = _predict_chain(n, year)
        return {
            "neighbourhood": n, "year": year, "type": "predicted",
            "predicted_score": chain[year],
            "safety_label":    _safety_label(chain[year]),
            "year_by_year":    chain,
            "model": "Random Forest", "trained_up_to": MAX_YEAR,
        }

def compare_neighbourhoods(neighbourhoods, year):
    results = [
        {"neighbourhood": s["neighbourhood"],
         "safety_score":  s.get("safety_score") or s.get("predicted_score", 0),
         "safety_label":  _safety_label(s.get("safety_score") or s.get("predicted_score", 0)),
         "type":          s["type"]}
        for name in neighbourhoods
        for s in [get_neighbourhood_stats(name, year)]
        if "error" not in s
    ]
    results.sort(key=lambda x: x["safety_score"], reverse=True)
    return {"comparison": results, "year": year}

def get_safest_neighbourhoods(year, top_n=5):
    if year <= MAX_YEAR:
        yr  = agg[agg["YEAR"] == year].sort_values("safety_score", ascending=False).reset_index(drop=True)
        top = yr.head(top_n)[["NEIGHBOURHOOD", "safety_score", "crime_count"]].copy()
        top["safety_label"] = top["safety_score"].apply(_safety_label)
        return {"year": year, "type": "historical", "rankings": top.to_dict(orient="records")}
    else:
        scores = []
        for n in NEIGHBOURHOODS:
            chain = _predict_chain(n, year)
            scores.append({"neighbourhood": n, "safety_score": chain[year],
                           "safety_label": _safety_label(chain[year])})
        scores.sort(key=lambda x: x["safety_score"], reverse=True)
        return {"year": year, "type": "predicted", "rankings": scores[:top_n]}

def get_crime_trend(neighbourhood, start_year, end_year):
    n = _match(neighbourhood)
    if not n:
        return {"error": f"Neighbourhood '{neighbourhood}' not found."}
    trend = {}
    hist  = agg[(agg["NEIGHBOURHOOD"] == n) &
                (agg["YEAR"] >= start_year) &
                (agg["YEAR"] <= min(end_year, MAX_YEAR))]
    for _, row in hist.iterrows():
        trend[int(row["YEAR"])] = {
            "safety_score": round(float(row["safety_score"]), 2),
            "crime_count":  int(row["crime_count"]), "type": "historical"
        }
    if end_year > MAX_YEAR:
        chain = _predict_chain(n, end_year)
        for yr, score in chain.items():
            if yr >= start_year:
                trend[yr] = {"safety_score": score, "type": "predicted"}
    return {"neighbourhood": n, "trend": dict(sorted(trend.items()))}

def list_neighbourhoods():
    return {"neighbourhoods": NEIGHBOURHOODS, "count": len(NEIGHBOURHOODS)}

# ─────────────────────────────────────────────────────────────────────────────
# 3. PYTHON INTENT PARSER  (no LLM needed for routing — 100% reliable)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_year(text):
    """Pull the first 4-digit year from text."""
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return int(m.group()) if m else None

def _extract_years(text):
    """Pull all years from text (for trend queries)."""
    return [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]

def _extract_neighbourhood(text):
    """Find the best matching neighbourhood in the user's message."""
    t = text.lower()
    # try longest match first so "west point grey" beats "west"
    for n in sorted(NEIGHBOURHOODS, key=len, reverse=True):
        if n.lower() in t:
            return n
    # fallback: fuzzy word match
    return _match(text)

def _extract_multiple_neighbourhoods(text):
    """Find all neighbourhood mentions in a comparison query."""
    t       = text.lower()
    found   = []
    for n in sorted(NEIGHBOURHOODS, key=len, reverse=True):
        if n.lower() in t and n not in found:
            found.append(n)
    return found

def _extract_top_n(text):
    """Extract a number like 'top 3' or 'top ten'."""
    word_map = {"one":1,"two":2,"three":3,"four":4,"five":5,
                "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    m = re.search(r"top\s+(\d+)", text.lower())
    if m: return int(m.group(1))
    for word, num in word_map.items():
        if f"top {word}" in text.lower():
            return num
    return 5

# ── Intent helpers (split out to reduce cyclomatic complexity) ────────────────

LIST_TRIGGERS    = ["list", "what neighbourhoods", "which neighbourhoods", "all neighbourhoods",
                    "available", "can i ask", "ask you about", "do you know", "what areas",
                    "which areas", "what cities"]
RANKING_TRIGGERS = ["safest", "most dangerous", "worst", "best", "ranking", "top",
                    "least safe", "highest crime", "lowest crime"]
TREND_TRIGGERS   = ["trend", "over the years", "from", "between", "history",
                    "changed", "getting", "improve", "worse", "progress"]
COMPARE_TRIGGERS = ["compare", "vs", "versus", "between", "and"]

def _intent_list(t):
    return any(w in t for w in LIST_TRIGGERS)

def _intent_ranking(t):
    return any(w in t for w in RANKING_TRIGGERS)

def _intent_trend(t):
    return any(w in t for w in TREND_TRIGGERS)

def _intent_compare(t):
    return any(w in t for w in COMPARE_TRIGGERS)

def _route_trend(user_input, t):
    neighbourhood = _extract_neighbourhood(user_input)
    years         = _extract_years(t)
    if not neighbourhood:
        return None, None
    if len(years) >= 2:
        return "get_crime_trend", get_crime_trend(neighbourhood, min(years), max(years))
    start = years[0] if years else 2015
    end   = years[0] if len(years) == 1 and years[0] > MAX_YEAR else MAX_YEAR
    return "get_crime_trend", get_crime_trend(neighbourhood, start, end)

def _route_compare(user_input, t):
    neighbourhoods = _extract_multiple_neighbourhoods(user_input)
    year           = _extract_year(t) or MAX_YEAR
    if len(neighbourhoods) >= 2:
        return "compare_neighbourhoods", compare_neighbourhoods(neighbourhoods, year)
    return None, None

def _route_single(user_input, t):
    neighbourhood = _extract_neighbourhood(user_input)
    year          = _extract_year(t)
    if neighbourhood and year:
        return "get_neighbourhood_stats", get_neighbourhood_stats(neighbourhood, year)
    if neighbourhood:
        return "get_neighbourhood_stats", get_neighbourhood_stats(neighbourhood, MAX_YEAR)
    if year:
        return "get_safest_neighbourhoods", get_safest_neighbourhoods(year)
    return None, None

def parse_and_execute(user_input):
    """Route user input to the correct tool using Python intent detection."""
    t = user_input.lower()
    if _intent_list(t):
        return "list_neighbourhoods", list_neighbourhoods()
    if _intent_ranking(t):
        return "get_safest_neighbourhoods", get_safest_neighbourhoods(_extract_year(t) or MAX_YEAR, _extract_top_n(t))
    if _intent_trend(t):
        result = _route_trend(user_input, t)
        if result[0]:
            return result
    if _intent_compare(t):
        result = _route_compare(user_input, t)
        if result[0]:
            return result
    return _route_single(user_input, t)

# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM  (only used to format the final answer, never for routing)
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "llama3.2"

chat_history = []

MAX_TOKENS = 512

def _build_prompt(user_input, data):
    """Build the prompt string depending on whether data is a list or stats."""
    data_str = json.dumps(data, indent=2)
    if isinstance(data, dict) and "neighbourhoods" in data:
        return (
            f"The user asked: \"{user_input}\"\n\n"
            f"Here are the available neighbourhoods:\n{data_str}\n\n"
            f"Simply list all the neighbourhood names clearly, one per line. "
            f"Do not add scores, opinions, or any extra commentary."
        )
    return (
        f"The user asked: \"{user_input}\"\n\n"
        f"Here is the exact data retrieved from the Vancouver crime database:\n{data_str}\n\n"
        f"Instructions:\n"
        f"- Answer the user's question conversationally and helpfully.\n"
        f"- You CAN give opinions and recommendations BUT only based strictly on the numbers above.\n"
        f"- If a safety score is above 75, you can say it is a safe area. If trending upward, say it is improving.\n"
        f"- If data is marked 'predicted', always mention it is an ML prediction and results may vary.\n"
        f"- Never invent crime statistics, neighbourhood facts, or advice not supported by the data.\n"
        f"- Keep it concise, 3 to 5 sentences max unless the user asked for a trend or list."
    )

def ask_llm(user_input, data):
    """Pass the real data to the LLM and return a grounded natural language answer."""
    prompt = _build_prompt(user_input, data)
    chat_history.append({"role": "user", "content": prompt})
    resp   = ollama.chat(model=MODEL, messages=chat_history, options={"num_predict": MAX_TOKENS})
    answer = resp["message"]["content"]
    chat_history.append({"role": "assistant", "content": answer})
    return answer

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 56)
print("  Vancouver Crime & Safety Chatbot  (free & local)")
print(f"  Historical: 2003-{MAX_YEAR}  |  Future: ML predicted")
print("  Type 'quit' to exit")
print("=" * 56)
print("Bot: Hey! Ask me anything about crime in Vancouver.")
print("     e.g. 'How is downtown in 2026?'")
print("          'Which neighbourhood will be safest in 2030?'")
print("          'Compare Kitsilano and West End in 2020'")
print("          'Show the trend in Strathcona from 2015 to 2027'\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBot: Goodbye!")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "bye"):
        print("Bot: Goodbye!")
        break

    tool_name, data = parse_and_execute(user_input)

    if data is None:
        # Nothing matched — let the LLM handle it as a general question
        chat_history.append({"role": "user", "content": user_input})
        resp   = ollama.chat(model=MODEL, messages=chat_history, options={"num_predict": MAX_TOKENS})
        answer = resp["message"]["content"]
        chat_history.append({"role": "assistant", "content": answer})
    else:
        answer = ask_llm(user_input, data)

    print(f"Bot: {answer}\n")
