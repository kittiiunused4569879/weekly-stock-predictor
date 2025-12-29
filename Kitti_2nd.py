# ============================================================
# Kitti_2nd.py — Stable Portfolio Weekly Stock Predictor
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Weekly Stock Predictor", layout="wide")
st.title("📊 Weekly Stock Predictor (Stable Version)")

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

N_EST = st.sidebar.slider("n_estimators", 100, 500, 300, 50)
LR = st.sidebar.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1], value=0.05)
MAX_DEPTH = st.sidebar.slider("max_depth", 2, 4, 3)
SUBSAMPLE = st.sidebar.select_slider("subsample", [0.7, 0.8, 0.9, 1.0], value=0.8)

HORIZON_LABEL = st.sidebar.selectbox(
    "Prediction Horizon",
    ["1 Week", "1 Month", "3 Months", "6 Months", "12 Months"]
)

HORIZON_MAP = {
    "1 Week": 1,
    "1 Month": 4,
    "3 Months": 12,
    "6 Months": 26,
    "12 Months": 52,
}

PRED_HORIZON = HORIZON_MAP[HORIZON_LABEL]

BUY_TH = st.sidebar.slider("BUY threshold (%)", 1.0, 5.0, 2.0, 0.5)
SELL_TH = st.sidebar.slider("SELL threshold (%)", 1.0, 5.0, 2.0, 0.5)

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
def make_features(close):
    df = pd.DataFrame(index=close.index)
    df["ret1"] = close.pct_change()
    df["ret4"] = close.pct_change(4)
    df["ret8"] = close.pct_change(8)
    df["vol12"] = df["ret1"].rolling(12).std()
    df["mom12"] = close / close.shift(12) - 1
    return df

# ------------------------------------------------------------
# TRAIN + FORECAST (NO NaNs POSSIBLE)
# ------------------------------------------------------------
def train_predict(weekly):
    feats = make_features(weekly)
    target = weekly.shift(-1)

    data = feats.copy()
    data["y"] = target
    data = data.dropna()

    if len(data) < 60:
        return None

    X = data.drop(columns="y")
    y = data["y"]

    model = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42,
    )
    model.fit(X, y)

    preds = []
    hist = weekly.copy()

    for _ in range(PRED_HORIZON):
        f = make_features(hist).iloc[[-1]]
        p = float(model.predict(f)[0])
        preds.append(p)
        hist.loc[hist.index[-1] + pd.offsets.Week(1)] = p

    return preds

# ------------------------------------------------------------
# SIGNAL
# ------------------------------------------------------------
def signal(last, pred):
    pct = (pred - last) / last * 100
    if pct >= BUY_TH:
        return "BUY"
    if pct <= -SELL_TH:
        return "SELL"
    return "HOLD"

# ------------------------------------------------------------
# MAIN LOOP (SAFE)
# ------------------------------------------------------------
for name, symbol in PORTFOLIO.items():
    st.markdown("---")
    st.subheader(name)

    try:
        df = yf.download(symbol, period="max", interval="1d", progress=False)
    except Exception as e:
        st.warning(f"Download failed: {e}")
        continue

    if df.empty or "Close" not in df:
        st.warning("No data")
        continue

    weekly = df["Close"].resample("W-FRI").last().dropna()

    out = train_predict(weekly)
    if out is None:
        st.warning("Not enough data")
        continue

    last = weekly.iloc[-1]
    final_pred = out[-1]

    st.metric("Last Close", f"{last:.2f}")
    st.metric("Predicted", f"{final_pred:.2f}")
    st.write("Signal:", signal(last, final_pred))

    future_idx = pd.date_range(
        weekly.index[-1] + pd.offsets.Week(1),
        periods=len(out),
        freq="W-FRI",
    )

    plot_df = pd.concat([
        pd.DataFrame({"Actual": weekly}),
        pd.DataFrame({"Predicted": out}, index=future_idx),
    ])

    st.line_chart(plot_df)
