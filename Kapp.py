# ============================================================
# MULTI-STOCK DAILY ROLLING PREDICTION APP
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os

st.set_page_config(page_title="Portfolio Daily Predictor", layout="wide")
st.title("📊 Portfolio – Daily Prediction vs Actual")
st.write("Runs daily | Grows day by day | Honest ML")

# ------------------------------------------------------------
# PORTFOLIO (Your holdings)
# ------------------------------------------------------------
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

LOG_FILE = "daily_portfolio_log.csv"
TODAY = pd.Timestamp.today().normalize()

# ------------------------------------------------------------
# LOAD LOG
# ------------------------------------------------------------
if os.path.exists(LOG_FILE):
    log = pd.read_csv(LOG_FILE, parse_dates=["date"])
else:
    log = pd.DataFrame(columns=["date", "stock", "predicted", "actual"])

# ------------------------------------------------------------
# FUNCTION: DAILY PREDICTION
# ------------------------------------------------------------
def daily_predict(symbol):
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

    return last_close, prediction

# ------------------------------------------------------------
# DAILY UPDATE (AUTO-SAFE)
# ------------------------------------------------------------
for name, symbol in PORTFOLIO.items():
    last_close, prediction = daily_predict(symbol)

    # Add row once per day per stock
    exists = (
        (log["date"] == TODAY).any() and
        (log["stock"] == name).any()
    )

    if not exists:
        log = pd.concat([
            log,
            pd.DataFrame([{
                "date": TODAY,
                "stock": name,
                "predicted": round(prediction, 2),
                "actual": round(last_close, 2)
            }])
        ], ignore_index=True)

log.to_csv(LOG_FILE, index=False)

# ------------------------------------------------------------
# UI: SHOW EACH STOCK
# ------------------------------------------------------------
for name in PORTFOLIO.keys():
    st.markdown("---")
    st.subheader(name)

    stock_log = log[log["stock"] == name].sort_values("date")

    if stock_log.empty:
        st.info("Waiting for data…")
        continue

    last_row = stock_log.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Last Actual", last_row["actual"])
    c2.metric("Predicted", last_row["predicted"])

    diff = (last_row["predicted"] - last_row["actual"]) / last_row["actual"] * 100
    c3.metric("Predicted Change %", f"{diff:.2f}%")

    st.line_chart(
        stock_log.set_index("date")[["predicted", "actual"]]
    )

st.success("Daily predictions updated for all stocks.")
