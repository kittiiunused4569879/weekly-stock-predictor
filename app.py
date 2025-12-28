# ============================================================
# WEEKLY STOCK PREDICTION APP (STREAMLIT CLOUD - FINAL FIXED)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import streamlit as st

# ------------------------------------------------------------
# Streamlit UI setup
# ------------------------------------------------------------
st.set_page_config(page_title="Weekly Share Predictor", layout="centered")

st.title(" Weekly Share Price Prediction")
st.write("Free | Web-based | Learns weekly")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SYMBOL = "HDFCGOLD.NS"
HISTORY_FILE = "weekly_predictions.csv"

# ------------------------------------------------------------
# 1. Fetch historical data from web (Yahoo Finance)
# ------------------------------------------------------------
df = yf.download(
    SYMBOL,
    period="3y",
    interval="1d",
    progress=False
)

if df.empty:
    st.error("No data received from Yahoo Finance")
    st.stop()

df = df[['Close']].dropna()

# ------------------------------------------------------------
# 2. Convert daily → weekly data
# ------------------------------------------------------------
weekly = df.resample('W').last()
weekly['prev_close'] = weekly['Close'].shift(1)
weekly.dropna(inplace=True)

# ------------------------------------------------------------
# 3. Train regression model
# ------------------------------------------------------------
X = weekly[['prev_close']]   # IMPORTANT: named column
y = weekly['Close']

model = LinearRegression()
model.fit(X, y)

# ------------------------------------------------------------
# 4. Predict next week (FIXED FEATURE FORMAT)
# ------------------------------------------------------------
last_price = weekly.iloc[-1]['Close']

#  CRITICAL FIX: prediction input must be a DataFrame
X_pred = pd.DataFrame([[last_price]], columns=["prev_close"])
prediction = model.predict(X_pred)[0]

# ------------------------------------------------------------
# 5. Load or create learning history
# ------------------------------------------------------------
if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
else:
    history = pd.DataFrame(columns=[
        "date",
        "last_week_price",
        "predicted_next_week",
        "actual_next_week"
    ])

# ------------------------------------------------------------
# 6. Learning step (update last week's actual)
# ------------------------------------------------------------
if not history.empty:
    if pd.isna(history.iloc[-1]["actual_next_week"]):
        history.loc[history.index[-1], "actual_next_week"] = last_price

# ------------------------------------------------------------
# 7. Save new prediction
# ------------------------------------------------------------
new_row = {
    "date": datetime.today().strftime("%Y-%m-%d"),
    "last_week_price": round(last_price, 2),
    "predicted_next_week": round(float(prediction), 2),
    "actual_next_week": np.nan
}

history = pd.concat(
    [history, pd.DataFrame([new_row])],
    ignore_index=True
)

history.to_csv(HISTORY_FILE, index=False)

# ------------------------------------------------------------
# 8. UI Output
# ------------------------------------------------------------
st.metric("Last Week Price", round(last_price, 2))
st.metric("Predicted Next Week Price", round(prediction, 2))

st.subheader("Prediction History (Learning Over Time)")
st.dataframe(history.tail(10))
