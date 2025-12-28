# ============================================================
# DAILY BATCH JOB – RUNS ONCE PER DAY (REAL DATA)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os

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

LOG_FILE = "daily_log.csv"
TODAY = datetime.now().date()

if os.path.exists(LOG_FILE):
    log = pd.read_csv(LOG_FILE, parse_dates=["date"])
else:
    log = pd.DataFrame(columns=["date","stock","predicted","actual"])

for stock, symbol in PORTFOLIO.items():
    df = yf.download(symbol, period="3y", interval="1d", progress=False)
    df = df[['Close']].dropna()

    df["prev_close"] = df["Close"].shift(1)
    df.dropna(inplace=True)

    X = df[["prev_close"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    last_close = float(df.iloc[-1]["Close"])
    X_pred = pd.DataFrame([[last_close]], columns=["prev_close"])
    prediction = float(model.predict(X_pred)[0])

    # only ONE row per stock per day
    exists = (
        (log["date"].dt.date == TODAY).any() and
        ((log["stock"] == stock) & (log["date"].dt.date == TODAY)).any()
    )

    if not exists:
        log.loc[len(log)] = [
            TODAY,
            stock,
            round(prediction, 2),
            round(last_close, 2)
        ]

log.to_csv(LOG_FILE, index=False)
