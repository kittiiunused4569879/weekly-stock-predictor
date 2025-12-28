# ============================================================
# PORTFOLIO STOCK PREDICTION APP (STREAMLIT CLOUD)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
from datetime import datetime
import os

st.set_page_config(page_title="Portfolio Prediction", layout="wide")
st.title("📊 My Portfolio – Weekly Prediction Dashboard")

# -------------------------
# Your Portfolio
# -------------------------
PORTFOLIO = {
    "ANANTRAJ": "ANANTRAJ.NS",
    "ARVINDFASN": "ARVINDFASN.NS",
    "HAVELLS": "HAVELLS.NS",
    "HCL-INSYS": "HCLINSYS.NS",
    "HDFCGOLD": "HDFCGOLD.NS",
    "SONATSOFTW": "SONATSOFTW.NS",
    "SULA": "SULA.NS",
    "SUZLON": "SUZLON.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "UCOBANK": "UCOBANK.NS",
    "YESBANK": "YESBANK.NS"
}

HISTORY_FILE = "portfolio_predictions.csv"

# -------------------------
# Load history
# -------------------------
if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
else:
    history = pd.DataFrame(columns=[
        "date", "stock", "actual_price", "predicted_price"
    ])

# -------------------------
# Helper function
# -------------------------
def predict_next_week(symbol):
    df = yf.download(symbol, period="3y", interval="1d", progress=False)
    df = df[['Close']].dropna()

    weekly = df.resample('W').last()
    weekly['prev_close'] = weekly['Close'].shift(1)
    weekly.dropna(inplace=True)

    X = weekly[['prev_close']]
    y = weekly['Close']

    model = LinearRegression()
    model.fit(X, y)

    last_price = float(weekly.iloc[-1]['Close'])
    X_pred = pd.DataFrame([[last_price]], columns=["prev_close"])
    pred = float(model.predict(X_pred)[0])

    return weekly, last_price, pred

# -------------------------
# UI LOOP
# -------------------------
for name, symbol in PORTFOLIO.items():
    st.markdown("---")
    st.subheader(name)

    weekly, last_price, prediction = predict_next_week(symbol)

    col1, col2, col3 = st.columns(3)

    col1.metric("Last Week Price", round(last_price, 2))
    col2.metric("Next Week Prediction", round(prediction, 2))
    col3.metric(
        "Predicted Change %",
        f"{round((prediction-last_price)/last_price*100,2)}%"
    )

    # Save history
    history = pd.concat([
        history,
        pd.DataFrame([{
            "date": datetime.today().strftime("%Y-%m-%d"),
            "stock": name,
            "actual_price": round(last_price, 2),
            "predicted_price": round(prediction, 2)
        }])
    ], ignore_index=True)

    # -------------------------
    # GRAPH: Actual vs Prediction
    # -------------------------
    chart_df = weekly[['Close']].copy()
    chart_df = chart_df.rename(columns={"Close": "Actual"})
    chart_df["Predicted"] = np.nan
    chart_df.iloc[-1, chart_df.columns.get_loc("Predicted")] = prediction

    st.line_chart(chart_df)

# -------------------------
# Save portfolio history
# -------------------------
history.to_csv(HISTORY_FILE, index=False)

st.success("Portfolio predictions updated successfully")
