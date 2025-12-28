# ============================================================
# WEEKLY STOCK PREDICTION APP (FREE – STREAMLIT CLOUD)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import streamlit as st

st.set_page_config(page_title="Weekly Share Predictor", layout="centered")

st.title("📈 Weekly Share Price Prediction")
st.write("Free | Learns weekly | Web-based")

SYMBOL = "HDFCGOLD.NS"
HISTORY_FILE = "weekly_predictions.csv"

# -----------------------------
# Fetch web data
# -----------------------------
df = yf.download(SYMBOL, period="3y", interval="1d", progress=False)
df = df[['Close']].dropna()

weekly = df.resample('W').last()
weekly['prev_close'] = weekly['Close'].shift(1)
weekly.dropna(inplace=True)

# -----------------------------
# Train model
# -----------------------------
X = weekly[['prev_close']]
y = weekly['Close']

model = LinearRegression()
model.fit(X, y)

last_price = weekly.iloc[-1]['Close']
prediction = model.predict([[last_price]])[0]

# -----------------------------
# Learning history
# -----------------------------
if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
else:
    history = pd.DataFrame(columns=[
        "date", "last_week_price", "predicted_next_week", "actual_next_week"
    ])

if not history.empty and pd.isna(history.iloc[-1]["actual_next_week"]):
    history.loc[history.index[-1], "actual_next_week"] = last_price

new_row = {
    "date": datetime.today().strftime("%Y-%m-%d"),
    "last_week_price": round(last_price, 2),
    "predicted_next_week": round(prediction, 2),
    "actual_next_week": np.nan
}

history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
history.to_csv(HISTORY_FILE, index=False)

# -----------------------------
# UI
# -----------------------------
st.metric("Last Week Price", round(last_price, 2))
st.metric("Next Week Prediction", round(prediction, 2))

st.subheader("Prediction History")
st.dataframe(history.tail(10))
