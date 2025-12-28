# ============================================================
# WEEKLY BATCH JOB (AUTO RUN)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

PORTFOLIO = {
    "ANANTRAJ": "ANANTRAJ.NS",
    "ARVINDFASN": "ARVINDFASN.NS",
    "HAVELLS": "HAVELLS.NS",
    "HCLINSYS": "HCLINSYS.NS",
    "HDFCGOLD": "HDFCGOLD.NS",
    "SONATSOFTW": "SONATSOFTW.NS",
    "SULA": "SULA.NS",
    "SUZLON": "SUZLON.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "UCOBANK": "UCOBANK.NS",
    "YESBANK": "YESBANK.NS"
}

OUT_FILE = "weekly_predictions_log.csv"

def make_features(close):
    f = pd.DataFrame(index=close.index)
    f["ret1"] = close.pct_change()
    f["ret4"] = close.pct_change(4)
    f["ma4"] = close.rolling(4).mean()
    f["ma8"] = close.rolling(8).mean()
    f["ma4_8"] = f["ma4"] / f["ma8"] - 1
    return f

rows = []

for name, symbol in PORTFOLIO.items():
    df = yf.download(symbol, period="3y", interval="1d", progress=False)
    if df.empty:
        continue

    weekly = df["Close"].resample("W").last().dropna()
    feats = make_features(weekly)
    y = weekly.shift(-1)

    data = feats.copy()
    data["y"] = y
    data.dropna(inplace=True)

    if len(data) < 60:
        continue

    X = data.drop(columns=["y"])
    y = data["y"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    pred = float(model.predict(X.iloc[[-1]])[0])
    last = float(weekly.iloc[-1])

    rows.append({
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "stock": name,
        "last_week": round(last, 2),
        "pred_next_week": round(pred, 2)
    })

if rows:
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
