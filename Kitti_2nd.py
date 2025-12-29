# ============================================================
# Kapp.py — Portfolio Weekly Stock Predictor (ENSEMBLE MODEL)
# Gradient Boosting + Random Forest (Averaged)
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
st.set_page_config(page_title="Portfolio Weekly Predictor (Ensemble)", layout="wide")
st.title("📊 Portfolio Weekly Predictor — Ensemble Model")
st.write("Gradient Boosting + Random Forest | Price-based | 1 − MAPE accuracy")

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
# CONTROLS
# ------------------------------------------------------------
BUY_THRESHOLD = st.slider("BUY threshold (+%)", 0.5, 10.0, 2.0, 0.5)
SELL_THRESHOLD = st.slider("SELL threshold (-%)", 0.5, 10.0, 2.0, 0.5)
MIN_WEEKS = st.slider("Minimum weekly samples", 52, 200, 80, 4)

HORIZON_LABEL = st.selectbox(
    "Prediction Horizon",
    ["1 Week", "1 Month", "3 Months"]
)
HORIZON_MAP = {"1 Week": 1, "1 Month": 4, "3 Months": 12}
PRED_HORIZON = HORIZON_MAP[HORIZON_LABEL]

with st.expander("Model parameters"):
    N_EST = st.slider("n_estimators", 100, 600, 300, 50)
    LR = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.08, 0.1], value=0.05)
    MAX_DEPTH = st.slider("max_depth", 2, 6, 3)
    SUBSAMPLE = st.select_slider("subsample (GB)", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def ensure_series(x):
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return pd.Series(x)

# ------------------------------------------------------------
# FEATURES (UNCHANGED)
# ------------------------------------------------------------
def make_features(close):
    close = ensure_series(close).astype(float)

    f = pd.DataFrame(index=close.index)
    f["ret1"] = close.pct_change(1)
    f["ret2"] = close.pct_change(2)
    f["ret4"] = close.pct_change(4)
    f["ret8"] = close.pct_change(8)

    ma4 = close.rolling(4).mean()
    ma8 = close.rolling(8).mean()
    ma12 = close.rolling(12).mean()
    ma20 = close.rolling(20).mean()

    f["ma4_ma8"] = (ma4 / (ma8 + 1e-9)) - 1
    f["ma8_ma12"] = (ma8 / (ma12 + 1e-9)) - 1
    f["ma12_ma20"] = (ma12 / (ma20 + 1e-9)) - 1

    f["vol8"] = f["ret1"].rolling(8).std()
    f["vol12"] = f["ret1"].rolling(12).std()

    f["mom4"] = close / close.shift(4) - 1
    f["mom12"] = close / close.shift(12) - 1

    return f

# ------------------------------------------------------------
# ENSEMBLE TRAIN + PREDICT
# ------------------------------------------------------------
def train_predict(weekly_close):
    weekly_close = ensure_series(weekly_close).dropna()
    if len(weekly_close) < MIN_WEEKS:
        return None

    feats = make_features(weekly_close)
    y = weekly_close.shift(-PRED_HORIZON)

    data = feats.copy()
    data["y"] = y
    data = data.dropna()

    if len(data) < MIN_WEEKS:
        return None

    X = data.drop(columns="y")
    y = data["y"]

    gb = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=N_EST,
        max_depth=MAX_DEPTH,
        random_state=42,
        n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []

    for tr, te in tscv.split(X):
        gb.fit(X.iloc[tr], y.iloc[tr])
        rf.fit(X.iloc[tr], y.iloc[tr])

        gb_p = gb.predict(X.iloc[te])
        rf_p = rf.predict(X.iloc[te])

        ensemble_p = (gb_p + rf_p) / 2.0
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], ensemble_p))

    acc = max(0.0, 1.0 - np.mean(mape_scores))

    gb.fit(X, y)
    rf.fit(X, y)

    pred = float((gb.predict(X.iloc[[-1]])[0] + rf.predict(X.iloc[[-1]])[0]) / 2.0)

    return pred, acc

def signal(last, pred):
    pct = (pred - last) / last * 100
    if pct >= BUY_THRESHOLD:
        return "BUY", "green"
    if pct <= -SELL_THRESHOLD:
        return "SELL", "red"
    return "HOLD", "orange"

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
for name, symbol in PORTFOLIO.items():
    st.markdown("---")
    st.subheader(f"{name} ({symbol})")

    df = yf.download(symbol, period="3y", interval="1d", progress=False)
    if df.empty or "Close" not in df:
        st.warning("No data")
        continue

    weekly = ensure_series(df["Close"]).resample("W-FRI").last().dropna()
    out = train_predict(weekly)

    if out is None:
        st.warning("Not enough data")
        continue

    pred, acc = out
    last = float(weekly.iloc[-1])
    sig, color = signal(last, pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"{last:.2f}")
    c2.metric("Predicted", f"{pred:.2f}")
    c3.metric("Accuracy (1 − MAPE)", f"{acc*100:.2f}%")
    c4.markdown(f"<h3 style='color:{color}'>{sig}</h3>", unsafe_allow_html=True)

    # Plot (recent window + continuation)
    wk_plot = weekly.iloc[-60:]
    future_idx = pd.date_range(wk_plot.index[-1] + pd.offsets.Week(1), periods=1, freq="W-FRI")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wk_plot.index, wk_plot.values, label="Actual", linewidth=2)
    ax.plot(future_idx, [pred], label="Predicted (Ensemble)", linewidth=4, color="orange")
    ax.set_title(f"Weekly Forecast ({HORIZON_LABEL}) — {sig}")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
