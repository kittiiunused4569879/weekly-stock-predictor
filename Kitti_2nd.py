# ============================================================
# Kapp.py — Full Portfolio Predictor with Confidence Bands
# ML: Gradient Boosting + Random Forest (Ensemble)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Portfolio Weekly Predictor", layout="wide")
st.title("📊 Portfolio Weekly Predictor — Interactive with Confidence Bands")

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
c1, c2, c3 = st.columns(3)
with c1:
    HORIZON_LABEL = st.selectbox("Prediction Horizon", ["1 Week", "4 Weeks", "12 Weeks"])
with c2:
    BUY_TH = st.slider("BUY threshold (+%)", 0.5, 10.0, 2.0, 0.5)
with c3:
    SELL_TH = st.slider("SELL threshold (-%)", 0.5, 10.0, 2.0, 0.5)

HORIZON_MAP = {"1 Week": 1, "4 Weeks": 4, "12 Weeks": 12}
PRED_HORIZON = HORIZON_MAP[HORIZON_LABEL]

MIN_WEEKS = st.slider("Minimum weekly samples", 52, 200, 80, 4)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def ensure_series(x):
    if isinstance(x, pd.Series):
        return x
    return x.iloc[:, 0]

def make_features(close):
    f = pd.DataFrame(index=close.index)
    f["ret1"] = close.pct_change()
    f["ret4"] = close.pct_change(4)
    f["ret8"] = close.pct_change(8)

    ma4 = close.rolling(4).mean()
    ma8 = close.rolling(8).mean()
    ma12 = close.rolling(12).mean()
    ma20 = close.rolling(20).mean()

    f["ma4_ma8"] = (ma4 / (ma8 + 1e-9)) - 1
    f["ma8_ma12"] = (ma8 / (ma12 + 1e-9)) - 1
    f["ma12_ma20"] = (ma12 / (ma20 + 1e-9)) - 1

    f["vol12"] = f["ret1"].rolling(12).std()
    f["mom12"] = close / close.shift(12) - 1
    return f

def signal(last, pred):
    pct = (pred - last) / last * 100
    if pct >= BUY_TH:
        return "BUY", "lime"
    if pct <= -SELL_TH:
        return "SELL", "red"
    return "HOLD", "orange"

# ------------------------------------------------------------
# ENSEMBLE TRAIN + FORECAST
# ------------------------------------------------------------
def train_predict_ensemble(weekly):
    feats = make_features(weekly)
    y = weekly.shift(-PRED_HORIZON)

    data = feats.copy()
    data["y"] = y
    data = data.dropna()

    X, y = data.drop(columns="y"), data["y"]

    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=5)
    mape = []

    for tr, te in tscv.split(X):
        gb.fit(X.iloc[tr], y.iloc[tr])
        rf.fit(X.iloc[tr], y.iloc[tr])
        preds = (gb.predict(X.iloc[te]) + rf.predict(X.iloc[te])) / 2
        mape.append(mean_absolute_percentage_error(y.iloc[te], preds))

    acc = max(0.0, 1 - np.mean(mape))

    gb.fit(X, y)
    rf.fit(X, y)

    future = []
    hist = weekly.copy()
    for _ in range(PRED_HORIZON):
        f = make_features(hist).iloc[[-1]]
        p = (gb.predict(f)[0] + rf.predict(f)[0]) / 2
        future.append(float(p))
        hist.loc[hist.index[-1] + pd.offsets.Week(1)] = p

    return future, acc

def backtest_predictions(weekly, lookback=80):
    preds, dates = [], []
    for i in range(-lookback, -PRED_HORIZON):
        train = weekly.iloc[:i]
        out, _ = train_predict_ensemble(train)
        preds.append(out[-1])
        dates.append(weekly.index[i + PRED_HORIZON])
    return pd.Series(preds, index=dates)

def confidence_band(actual, predicted, z=1.96):
    residuals = actual.loc[predicted.index] - predicted
    sigma = residuals.std()
    return predicted - z * sigma, predicted + z * sigma

# ------------------------------------------------------------
# DATA LOAD
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_weekly(symbol):
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    close = ensure_series(df["Close"]).dropna()
    return close.resample("W-FRI").last().dropna()

# ------------------------------------------------------------
# PORTFOLIO COMPUTE
# ------------------------------------------------------------
results = []
for name, symbol in PORTFOLIO.items():
    try:
        weekly = load_weekly(symbol)
        if len(weekly) < MIN_WEEKS:
            continue
        future, acc = train_predict_ensemble(weekly)
        last = weekly.iloc[-1]
        pred = future[-1]
        sig, _ = signal(last, pred)
        results.append({
            "name": name,
            "symbol": symbol,
            "weekly": weekly,
            "future": future,
            "pred": pred,
            "acc": acc,
            "signal": sig,
        })
    except:
        pass

# ------------------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------------------
st.markdown("---")
df_sum = pd.DataFrame([{
    "Stock": r["name"],
    "Signal": r["signal"],
    "Last": round(r["weekly"].iloc[-1], 2),
    "Predicted": round(r["pred"], 2),
    "Accuracy %": round(r["acc"] * 100, 2)
} for r in results])

st.dataframe(df_sum, use_container_width=True)

# ------------------------------------------------------------
# INTERACTIVE CHART
# ------------------------------------------------------------
st.markdown("---")
selected = st.selectbox("Select Stock", df_sum["Stock"])
r = next(x for x in results if x["name"] == selected)

weekly = r["weekly"]
bt = backtest_predictions(weekly)
low, high = confidence_band(weekly, bt)

future_idx = pd.date_range(weekly.index[-1] + pd.offsets.Week(1),
                           periods=len(r["future"]), freq="W-FRI")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=weekly.index, y=weekly.values,
    mode="lines", name="Actual",
    line=dict(color="lime", width=2)
))

fig.add_trace(go.Scatter(
    x=bt.index, y=bt.values,
    mode="lines", name="Prediction (Past)",
    line=dict(color="orange", dash="dot")
))

fig.add_trace(go.Scatter(
    x=high.index, y=high.values,
    line=dict(width=0), showlegend=False
))
fig.add_trace(go.Scatter(
    x=low.index, y=low.values,
    fill="tonexty",
    fillcolor="rgba(255,165,0,0.25)",
    line=dict(width=0),
    name="95% Confidence"
))

fig.add_trace(go.Scatter(
    x=future_idx, y=r["future"],
    mode="lines+markers",
    name="Forecast",
    line=dict(color="orange", width=4)
))

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True)),
    height=600,
    title=f"{selected} — {HORIZON_LABEL} Forecast ({r['signal']})"
)

st.plotly_chart(fig, use_container_width=True)
