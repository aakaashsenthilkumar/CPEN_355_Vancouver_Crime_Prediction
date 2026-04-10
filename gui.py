import re
import json
import sys
import os
import joblib
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ollama

# Resolve base path whether running as .py or bundled .exe
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH  = os.path.join(BASE_DIR, "data", "crimedata_csv_AllNeighbourhoods_AllYears.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
LE_PATH    = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
SCORE_PATH = os.path.join(BASE_DIR, "model", "score_params.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ENGINE
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data and model...")

df = pd.read_csv(
    DATA_PATH,
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

# Load saved score normalisation params
if os.path.exists(SCORE_PATH):
    score_params = joblib.load(SCORE_PATH)
    min_ws = score_params["min_ws"]
    max_ws = score_params["max_ws"]
else:
    min_ws = agg["weighted_score"].min()
    max_ws = agg["weighted_score"].max()

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

# Load the best saved model or retrain as fallback
if os.path.exists(MODEL_PATH):
    rf_model = joblib.load(MODEL_PATH)
    print("Loaded saved Random Forest model.")
else:
    print("No saved model found — retraining...")
    from sklearn.ensemble import RandomForestRegressor
    train_data = agg.dropna(subset=["prev_year_score"])
    FEATURES   = ["neighbourhood_enc", "YEAR", "prev_year_score", "rolling_avg_score", "crime_trend"]
    rf_model   = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(train_data[FEATURES], train_data["safety_score"])

NEIGHBOURHOODS = sorted(agg["NEIGHBOURHOOD"].unique())
MAX_YEAR       = int(agg["YEAR"].max())
print(f"Ready. {len(NEIGHBOURHOODS)} neighbourhoods loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOL FUNCTIONS  (identical to chatbot.py)
# ─────────────────────────────────────────────────────────────────────────────

def _safety_label(score):
    if score >= 75: return "Safe"
    if score >= 50: return "Moderate"
    if score >= 25: return "High Crime"
    return "Very High Crime"

def _match(query):
    q = query.strip().lower()
    for n in NEIGHBOURHOODS:
        if n.lower() == q: return n
    matches = [n for n in NEIGHBOURHOODS if q in n.lower()]
    if len(matches) == 1: return matches[0]
    q_words = set(q.split())
    scored  = [(n, len(q_words & set(n.lower().split()))) for n in NEIGHBOURHOODS]
    scored  = [(n, s) for n, s in scored if s > 0]
    return max(scored, key=lambda x: x[1])[0] if scored else None

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
            "neighbourhood_enc": enc, "YEAR": cur,
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
    matched = _match(name)
    return matched if matched in NEIGHBOURHOODS else None

def _sanitize_year(year):
    return int(year) if 1900 <= int(year) <= 2100 else None

def get_neighbourhood_stats(neighbourhood, year):
    n    = _sanitize_neighbourhood(neighbourhood)
    year = _sanitize_year(year)
    if not n:
        return {"error": f"Neighbourhood '{neighbourhood}' not found."}
    if not year or year < 2003:
        return {"error": "No data before 2003."}
    if year <= MAX_YEAR:
        mask    = (agg["NEIGHBOURHOOD"] == n) & (agg["YEAR"] == year)
        row     = agg[mask]
        if row.empty: return {"error": f"No records for {n} in {year}."}
        r        = row.iloc[0]
        df_mask  = (df["NEIGHBOURHOOD"] == n) & (df["YEAR"] == year)
        breakdown = df[df_mask].groupby("TYPE").size().sort_values(ascending=False).to_dict()
        yr_data  = agg[agg["YEAR"] == year].sort_values("safety_score", ascending=False).reset_index(drop=True)
        rank     = int(yr_data[yr_data["NEIGHBOURHOOD"] == n].index[0]) + 1
        return {
            "neighbourhood": n, "year": year, "type": "historical",
            "crime_count": int(r["crime_count"]),
            "safety_score": round(float(r["safety_score"]), 2),
            "safety_label": _safety_label(float(r["safety_score"])),
            "rank_safest": rank, "total_neighbourhoods": len(yr_data),
            "crime_breakdown": breakdown,
        }
    else:
        chain = _predict_chain(n, year)
        return {
            "neighbourhood": n, "year": year, "type": "predicted",
            "predicted_score": chain[year],
            "safety_label": _safety_label(chain[year]),
            "year_by_year": chain, "model": "Random Forest", "trained_up_to": MAX_YEAR,
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
        scores = [{"neighbourhood": n, "safety_score": _predict_chain(n, year)[year],
                   "safety_label": _safety_label(_predict_chain(n, year)[year])}
                  for n in NEIGHBOURHOODS]
        scores.sort(key=lambda x: x["safety_score"], reverse=True)
        return {"year": year, "type": "predicted", "rankings": scores[:top_n]}

def get_crime_trend(neighbourhood, start_year, end_year):
    n = _match(neighbourhood)
    if not n: return {"error": f"Neighbourhood '{neighbourhood}' not found."}
    trend = {}
    hist  = agg[(agg["NEIGHBOURHOOD"] == n) &
                (agg["YEAR"] >= start_year) &
                (agg["YEAR"] <= min(end_year, MAX_YEAR))]
    for _, row in hist.iterrows():
        trend[int(row["YEAR"])] = {
            "safety_score": round(float(row["safety_score"]), 2),
            "crime_count": int(row["crime_count"]), "type": "historical"
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
# 3. INTENT PARSER  (identical to chatbot.py)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_year(text):
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return int(m.group()) if m else None

def _extract_years(text):
    return [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]

def _extract_neighbourhood(text):
    t = text.lower()
    for n in sorted(NEIGHBOURHOODS, key=len, reverse=True):
        if n.lower() in t: return n
    return _match(text)

def _extract_multiple_neighbourhoods(text):
    t, found = text.lower(), []
    for n in sorted(NEIGHBOURHOODS, key=len, reverse=True):
        if n.lower() in t and n not in found:
            found.append(n)
    return found

def _extract_top_n(text):
    word_map = {"one":1,"two":2,"three":3,"four":4,"five":5,
                "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    m = re.search(r"top\s+(\d+)", text.lower())
    if m: return int(m.group(1))
    for word, num in word_map.items():
        if f"top {word}" in text.lower(): return num
    return 5

LIST_TRIGGERS    = ["list","what neighbourhoods","which neighbourhoods","all neighbourhoods",
                    "available","can i ask","ask you about","do you know","what areas","which areas"]
RANKING_TRIGGERS = ["safest","most dangerous","worst","best","ranking","top","least safe",
                    "highest crime","lowest crime"]
TREND_TRIGGERS   = ["trend","over the years","from","between","history","changed",
                    "getting","improve","worse","progress"]
COMPARE_TRIGGERS = ["compare","vs","versus","between","and"]

def parse_and_execute(user_input):
    t = user_input.lower()
    if any(w in t for w in LIST_TRIGGERS):
        return "list_neighbourhoods", list_neighbourhoods()
    if any(w in t for w in RANKING_TRIGGERS):
        return "get_safest_neighbourhoods", get_safest_neighbourhoods(_extract_year(t) or MAX_YEAR, _extract_top_n(t))
    if any(w in t for w in TREND_TRIGGERS):
        neighbourhood = _extract_neighbourhood(user_input)
        years         = _extract_years(t)
        if neighbourhood:
            if len(years) >= 2:
                return "get_crime_trend", get_crime_trend(neighbourhood, min(years), max(years))
            start = years[0] if years else 2015
            end   = years[0] if len(years) == 1 and years[0] > MAX_YEAR else MAX_YEAR
            return "get_crime_trend", get_crime_trend(neighbourhood, start, end)
    if any(w in t for w in COMPARE_TRIGGERS):
        neighbourhoods = _extract_multiple_neighbourhoods(user_input)
        year           = _extract_year(t) or MAX_YEAR
        if len(neighbourhoods) >= 2:
            return "compare_neighbourhoods", compare_neighbourhoods(neighbourhoods, year)
    neighbourhood = _extract_neighbourhood(user_input)
    year          = _extract_year(t)
    if neighbourhood and year:
        return "get_neighbourhood_stats", get_neighbourhood_stats(neighbourhood, year)
    if neighbourhood:
        return "get_neighbourhood_stats", get_neighbourhood_stats(neighbourhood, MAX_YEAR)
    if year:
        return "get_safest_neighbourhoods", get_safest_neighbourhoods(year)
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM
# ─────────────────────────────────────────────────────────────────────────────
MODEL      = "llama3.2"
MAX_TOKENS = 512
chat_history = []

def _build_prompt(user_input, data):
    data_str = json.dumps(data, indent=2)
    if isinstance(data, dict) and "neighbourhoods" in data:
        return (f"The user asked: \"{user_input}\"\n\n"
                f"Here are the available neighbourhoods:\n{data_str}\n\n"
                f"Simply list all the neighbourhood names clearly, one per line. "
                f"Do not add scores, opinions, or any extra commentary.")
    return (f"The user asked: \"{user_input}\"\n\n"
            f"Here is the exact data retrieved from the Vancouver crime database:\n{data_str}\n\n"
            f"Instructions:\n"
            f"- Answer conversationally and helpfully.\n"
            f"- You CAN give opinions and recommendations BUT only based strictly on the numbers above.\n"
            f"- If a safety score is above 75, say it is a safe area. If trending upward, say it is improving.\n"
            f"- If data is marked 'predicted', always mention it is an ML prediction.\n"
            f"- Never invent facts not in the data.\n"
            f"- Keep it concise, 3 to 5 sentences max unless the user asked for a trend or list.")

def get_response(user_input):
    tool_name, data = parse_and_execute(user_input)
    if data is None:
        chat_history.append({"role": "user", "content": user_input})
        resp   = ollama.chat(model=MODEL, messages=chat_history, options={"num_predict": MAX_TOKENS})
        answer = resp["message"]["content"]
    else:
        prompt = _build_prompt(user_input, data)
        chat_history.append({"role": "user", "content": prompt})
        resp   = ollama.chat(model=MODEL, messages=chat_history, options={"num_predict": MAX_TOKENS})
        answer = resp["message"]["content"]
    chat_history.append({"role": "assistant", "content": answer})
    return answer

# ─────────────────────────────────────────────────────────────────────────────
# 5. GUI
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = "#1A1A2E"   # dark navy background
SIDEBAR_BG  = "#16213E"   # slightly lighter sidebar
BUBBLE_USER = "#0F3460"   # user bubble
BUBBLE_BOT  = "#2A2A4A"   # bot bubble
ACCENT      = "#E94560"   # red accent
TEXT        = "#E0E0E0"   # main text
TEXT_DIM    = "#888888"   # dimmed text
FONT_MAIN   = ("Segoe UI", 11)
FONT_BOLD   = ("Segoe UI", 11, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_TITLE  = ("Segoe UI", 14, "bold")

SUGGESTIONS = [
    "How safe is Kitsilano in 2022?",
    "Which neighbourhood will be safest in 2030?",
    "Compare Downtown and West End in 2019",
    "Show crime trend in Strathcona from 2015 to 2027",
    "What are the top 5 safest neighbourhoods in 2023?",
    "Should I move to West Point Grey?",
]

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vancouver Crime & Safety Chatbot")
        self.root.geometry("1100x720")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self._build_ui()
        self._welcome()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Sidebar
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=240)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

        # Main chat area
        main = tk.Frame(self.root, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self._build_header(main)
        self._build_chat(main)
        self._build_input(main)

    def _build_sidebar(self, parent):
        # Title
        tk.Label(parent, text="Vancouver\nSafety Bot", font=FONT_TITLE,
                 bg=SIDEBAR_BG, fg=ACCENT, justify="center").pack(pady=(24, 4))
        tk.Label(parent, text=f"Data: 2003–{MAX_YEAR}  |  Future: ML",
                 font=FONT_SMALL, bg=SIDEBAR_BG, fg=TEXT_DIM).pack(pady=(0, 20))

        tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x", padx=16)

        # Suggestions
        tk.Label(parent, text="Try asking:", font=("Segoe UI", 10, "bold"),
                 bg=SIDEBAR_BG, fg=TEXT).pack(anchor="w", padx=16, pady=(16, 6))

        for s in SUGGESTIONS:
            btn = tk.Button(
                parent, text=s, font=FONT_SMALL, bg=SIDEBAR_BG, fg=TEXT_DIM,
                activebackground=BUBBLE_USER, activeforeground=TEXT,
                relief="flat", cursor="hand2", wraplength=200, justify="left",
                anchor="w", padx=8, pady=4,
                command=lambda txt=s: self._inject(txt)
            )
            btn.pack(fill="x", padx=8, pady=2)
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg=TEXT))
            btn.bind("<Leave>", lambda e, b=btn: b.config(fg=TEXT_DIM))

        # Clear button at bottom
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x", padx=16, side="bottom", pady=(0, 8))
        tk.Button(
            parent, text="Clear Chat", font=FONT_SMALL, bg=SIDEBAR_BG, fg=ACCENT,
            activebackground=ACCENT, activeforeground="white",
            relief="flat", cursor="hand2", command=self._clear_chat
        ).pack(side="bottom", pady=8)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=SIDEBAR_BG, height=56)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        tk.Label(header, text="Vancouver Crime & Safety Chatbot",
                 font=FONT_BOLD, bg=SIDEBAR_BG, fg=TEXT).pack(side="left", padx=20, pady=14)
        self.status_label = tk.Label(header, text="● Ready", font=FONT_SMALL,
                                     bg=SIDEBAR_BG, fg="#2ECC71")
        self.status_label.pack(side="right", padx=20)

    def _build_chat(self, parent):
        chat_frame = tk.Frame(parent, bg=BG)
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, state="disabled",
            bg=BG, fg=TEXT, font=FONT_MAIN,
            relief="flat", padx=20, pady=16,
            selectbackground=BUBBLE_USER,
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew")

        # Tag styles for bubbles
        self.chat_display.tag_config("user_name",  foreground=ACCENT,      font=FONT_BOLD)
        self.chat_display.tag_config("user_text",  foreground=TEXT,        font=FONT_MAIN,
                                     lmargin1=20, lmargin2=20, rmargin=80,
                                     background=BUBBLE_USER, relief="flat",
                                     spacing1=4, spacing3=8)
        self.chat_display.tag_config("bot_name",   foreground="#5DADE2",   font=FONT_BOLD)
        self.chat_display.tag_config("bot_text",   foreground=TEXT,        font=FONT_MAIN,
                                     lmargin1=20, lmargin2=20, rmargin=80,
                                     background=BUBBLE_BOT, relief="flat",
                                     spacing1=4, spacing3=8)
        self.chat_display.tag_config("typing",     foreground=TEXT_DIM,    font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_config("divider",    foreground=SIDEBAR_BG)
        self.chat_display.tag_config("timestamp",  foreground=TEXT_DIM,    font=FONT_SMALL)

    def _build_input(self, parent):
        input_frame = tk.Frame(parent, bg=SIDEBAR_BG, pady=12)
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)

        self.input_box = tk.Text(
            input_frame, height=2, font=FONT_MAIN,
            bg="#0D1B2A", fg=TEXT, insertbackground=TEXT,
            relief="flat", padx=12, pady=8, wrap=tk.WORD,
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(16, 8), pady=4)
        self.input_box.bind("<Return>",       self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)  # allow newline with shift

        self.send_btn = tk.Button(
            input_frame, text="Send", font=FONT_BOLD,
            bg=ACCENT, fg="white", activebackground="#C0392B",
            relief="flat", cursor="hand2", padx=20, pady=8,
            command=self._send
        )
        self.send_btn.grid(row=0, column=1, padx=(0, 16), pady=4)

        tk.Label(input_frame, text="Enter to send  ·  Shift+Enter for new line",
                 font=FONT_SMALL, bg=SIDEBAR_BG, fg=TEXT_DIM).grid(
            row=1, column=0, columnspan=2, pady=(0, 4))

    # ── Chat logic ────────────────────────────────────────────────────────────
    def _welcome(self):
        self._append_bot(
            f"Hey! I'm your Vancouver crime and safety analyst.\n\n"
            f"I have real VPD data from 2003–{MAX_YEAR} and can predict future years "
            f"using a trained Random Forest model.\n\n"
            f"Try one of the suggestions on the left, or ask me anything!"
        )

    def _inject(self, text):
        """Click a suggestion — fill input and send."""
        self.input_box.delete("1.0", tk.END)
        self.input_box.insert("1.0", text)
        self._send()

    def _on_enter(self, event):
        if not event.state & 0x1:   # Shift not held
            self._send()
            return "break"

    def _send(self):
        user_input = self.input_box.get("1.0", tk.END).strip()
        if not user_input:
            return
        self.input_box.delete("1.0", tk.END)
        self._append_user(user_input)
        self._set_status("● Thinking...", "#F39C12")
        self.send_btn.config(state="disabled")
        self._show_typing()
        threading.Thread(target=self._worker, args=(user_input,), daemon=True).start()

    def _worker(self, user_input):
        try:
            answer = get_response(user_input)
        except Exception as e:
            answer = f"Sorry, something went wrong: {e}"
        self.root.after(0, self._on_response, answer)

    def _on_response(self, answer):
        self._hide_typing()
        self._append_bot(answer)
        self._set_status("● Ready", "#2ECC71")
        self.send_btn.config(state="normal")

    # ── Display helpers ───────────────────────────────────────────────────────
    def _append_user(self, text):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, "You\n", "user_name")
        self.chat_display.insert(tk.END, text + "\n", "user_text")
        self.chat_display.insert(tk.END, "\n", "divider")
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)

    def _append_bot(self, text):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, "Bot\n", "bot_name")
        self.chat_display.insert(tk.END, text + "\n", "bot_text")
        self.chat_display.insert(tk.END, "\n", "divider")
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)

    def _show_typing(self):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, "Bot is thinking...\n", "typing")
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)
        self._typing_index = self.chat_display.index(tk.END)

    def _hide_typing(self):
        self.chat_display.config(state="normal")
        # Remove the last "Bot is thinking..." line
        content = self.chat_display.get("1.0", tk.END)
        if "Bot is thinking..." in content:
            start = content.rfind("Bot is thinking...")
            line  = self.chat_display.index(f"1.0 + {start} chars linestart")
            self.chat_display.delete(line, f"{line} lineend +1c")
        self.chat_display.config(state="disabled")

    def _clear_chat(self):
        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state="disabled")
        chat_history.clear()
        self._welcome()

    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)


# ─────────────────────────────────────────────────────────────────────────────
# 6. LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = ChatApp(root)
    root.mainloop()
