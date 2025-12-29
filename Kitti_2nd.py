# ============================================================
# Kitti_Final_NewML.py
# New ML Logic:
#   - Predict RETURNS instead of PRICE
#   - Model selector: Gradient Boosting / Random Forest
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Weekly Stock Predictor", layout="wide")
st.title("📊 Weekly Stock Predictor (New ML Logic: Return Forecasting)")

# ------------------------------------------------------------
# PORTFOLIO
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
    "YESBANK": "YESBANK.NS",
}

# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.header("Controls")

MODEL_TYPE = st.sidebar.selectbox(
    "ML Model",
    ["Gradient Boosting (Returns)", "Random Forest (Returns)"]
)

N_EST = st.sidebar.slider("n_estimators", 100, 500, 300, 50)
MAX_DEPTH = st.sidebar.slider("max_depth", 2, 6, 3)

HORIZON_LABEL = st.sidebar.selectbox(
    "Prediction Horizon",
    ["1 Week", "1 Month", "3 Months", "6 Months"]
)
HORIZON_MAP = {"1 Week": 1, "1 Month": 4, "3 Months": 12, "6 Months": 26}
PRED_HORIZON = HORIZON_MAP[HORIZON_LABEL]

BUY_TH = st.sidebar.slider("BUY threshold (%)", 1.0, 5.0, 2.0, 0.5)
SELL_TH = st.sidebar.slider("SELL threshold (%)", 1.0, 5.0, 2.0, 0.5)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def ensure_series(x):
    if isinstance(x, pd.Series):
        return x
    return x.iloc[:, 0]

def make_features(close):
    close = ensure_series(close).astype(float)
    df = pd.DataFrame(index=close.index)
    df["ret1"] = close.pct_change()
    df["ret4"] = close.pct_change(4)
    df["ret8"] = close.pct_change(8)
    df["vol12"] = df["ret1"].rolling(12).std()
    df["mom12"] = close / close.shift(12) - 1
    return df

def get_signal(last, pred):
    pct = (pred - last) / last * 100
    if pct >= BUY_TH:
        return "BUY", "green"
    elif pct <= -SELL_TH:
        return "SELL", "red"
    else:
        return "HOLD", "orange"

# ------------------------------------------------------------
# TRAIN + ACCURACY + FORECAST (RETURN-BASED ML)
# ------------------------------------------------------------
def train_predict_returns(weekly):
    feats = make_features(weekly)
    returns = weekly.pct_change().shift(-1)   # 🔥 TARGET = RETURN

    data = feats.copy()
    data["y"] = returns
    data = data.dropna()

    X = data.drop(columns="y")
    y = data["y"]

    if MODEL_TYPE.startswith("Gradient"):
        model = GradientBoostingRegressor(
            n_estimators=N_EST,
            max_depth=MAX_DEPTH,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=N_EST,
            max_depth=MAX_DEPTH,
            random_state=42,
            n_jobs=-1
        )

    # Accuracy via TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], preds))

    accuracy = max(0.0, 1.0 - np.mean(mape_scores))

    # Final training
    model.fit(X, y)

    # Recursive forecasting (returns → prices)
    preds_price = []
    hist = weekly.copy()

    for _ in range(PRED_HORIZON):
        f = make_features(hist).iloc[[-1]]
        if f.isna().any(axis=1).iloc[0]:
            break

        ret_pred = model.predict(f)[0]
        next_price = hist.iloc[-1] * (1 + ret_pred)

        preds_price.append(float(next_price))
        hist.loc[hist.index[-1] + pd.offsets.Week(1)] = next_price

    return preds_price, accuracy

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
for name, symbol in PORTFOLIO.items():
    st.markdown("---")
    st.subheader(f"{name} ({symbol})")

    df = yf.download(symbol, period="max", interval="1d", progress=False)
    if df.empty or "Close" not in df:
        st.warning("No data")
        continue

    weekly = ensure_series(df["Close"]).resample("W-FRI").last().dropna()
    if len(weekly) < 80:
        st.warning("Not enough data")
        continue

    preds, acc = train_predict_returns(weekly)

    last = float(weekly.iloc[-1])
    final_pred = float(preds[-1])

    signal_text, signal_color = get_signal(last, final_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"{last:.2f}")
    c2.metric("Predicted", f"{final_pred:.2f}")
    c3.metric("Accuracy (1 − MAPE)", f"{acc*100:.2f}%")
    c4.markdown(
        f"<h3 style='color:{signal_color};'>{signal_text}</h3>",
        unsafe_allow_html=True,
    )

    # Plot
    future_idx = pd.date_range(
        weekly.index[-1] + pd.offsets.Week(1),
        periods=len(preds),
        freq="W-FRI",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weekly.index, weekly.values, label="Actual", linewidth=2)
    ax.plot(future_idx, preds, label="Predicted", linewidth=4, color="orange")

    ax.set_title(f"Return-Based Forecast — {signal_text}")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
