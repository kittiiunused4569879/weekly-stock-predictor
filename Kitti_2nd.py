# ============================================================
# Kapp.py — Portfolio Weekly Stock Predictor (Gradient Boosting)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Portfolio Weekly Predictor (GB)", layout="wide")
st.title("📊 Portfolio Weekly Predictor – Gradient Boosting")
st.write("Equity Curve + Drawdown + Directional Accuracy + Sharpe Ranking")

# ------------------------------------------------------------
# FULL PORTFOLIO
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
st.sidebar.header("Model Controls")

N_EST = st.sidebar.slider("n_estimators", 100, 800, 300, 50)
LR = st.sidebar.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.08, 0.1, 0.2], value=0.05)
MAX_DEPTH = st.sidebar.slider("max_depth", 1, 6, 3)
SUBSAMPLE = st.sidebar.select_slider("subsample", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)

PRED_HORIZON = st.sidebar.selectbox(
    "Prediction Horizon",
    {"1 Week": 1, "1 Month": 4, "3 Months": 12, "6 Months": 26, "12 Months": 52},
)

BUY_TH = st.sidebar.slider("BUY threshold (%)", 0.5, 10.0, 2.0, 0.5)
SELL_TH = st.sidebar.slider("SELL threshold (%)", 0.5, 10.0, 2.0, 0.5)

MIN_WEEKS = st.sidebar.slider("Minimum weekly samples", 52, 200, 80, 4)

force_rerun = st.sidebar.button("🔄 Re-train from scratch")

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if force_rerun:
    st.session_state.run_id += 1
    st.cache_data.clear()

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def make_features(close):
    f = pd.DataFrame(index=close.index)
    f["ret1"] = close.pct_change(1)
    f["ret4"] = close.pct_change(4)
    f["ret8"] = close.pct_change(8)
    f["vol12"] = f["ret1"].rolling(12).std()
    f["mom12"] = close / close.shift(12) - 1
    return f

def signal(last, pred):
    pct = (pred - last) / last * 100
    if pct >= BUY_TH:
        return 1    # BUY
    if pct <= -SELL_TH:
        return -1   # SELL
    return 0        # HOLD

# ------------------------------------------------------------
# TRAIN + FORECAST
# ------------------------------------------------------------
def train_predict(weekly):
    feats = make_features(weekly).dropna()
    y = weekly.shift(-1).loc[feats.index]

    model = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42,
    )
    model.fit(feats, y)

    preds, hist = [], weekly.copy()
    for _ in range(PRED_HORIZON):
        f = make_features(hist).iloc[[-1]]
        p = model.predict(f)[0]
        preds.append(p)
        hist.loc[hist.index[-1] + pd.offsets.Week(1)] = p

    return preds

# ------------------------------------------------------------
# EQUITY CURVE & DRAWDOWN
# ------------------------------------------------------------
def equity_and_drawdown(weekly):
    returns = weekly.pct_change().dropna()
    equity = (1 + returns).cumprod()

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return equity, drawdown, drawdown.min()

# ------------------------------------------------------------
# MAIN COMPUTE
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_portfolio(items, run_id):
    results = []

    for name, symbol in items:
        df = yf.download(symbol, period="max", interval="1d", progress=False)
        if df.empty or "Close" not in df:
            continue

        weekly = df["Close"].resample("W-FRI").last().dropna()
        if len(weekly) < MIN_WEEKS:
            continue

        preds = train_predict(weekly)
        last = weekly.iloc[-1]
        final_pred = preds[-1]

        sig = signal(last, final_pred)

        # Strategy returns
        strat_returns = weekly.pct_change().shift(-1) * sig
        strat_returns = strat_returns.dropna()

        equity = (1 + strat_returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak

        results.append({
            "name": name,
            "weekly": weekly,
            "preds": preds,
            "signal": sig,
            "equity": equity,
            "drawdown": drawdown,
            "max_dd": drawdown.min(),
        })

    return results

# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------
results = compute_portfolio(list(PORTFOLIO.items()), st.session_state.run_id)

for r in results:
    st.markdown("---")
    st.subheader(r["name"])

    sig_label = {1: "BUY", -1: "SELL", 0: "HOLD"}[r["signal"]]
    st.write("Signal:", sig_label)
    st.metric("Max Drawdown", f"{r['max_dd']*100:.2f}%")

    c1, c2 = st.columns(2)

    with c1:
        st.caption("📊 Equity Curve (Strategy P&L)")
        st.line_chart(r["equity"])

    with c2:
        st.caption("📉 Drawdown Curve")
        st.line_chart(r["drawdown"])

# ------------------------------------------------------------
# PORTFOLIO-LEVEL SUMMARY
# ------------------------------------------------------------
st.markdown("---")
st.header("📈 Portfolio Risk Summary")

summary = pd.DataFrame([{
    "Stock": r["name"],
    "Max Drawdown (%)": r["max_dd"] * 100,
    "Final Equity": r["equity"].iloc[-1] if len(r["equity"]) > 0 else np.nan,
} for r in results])

st.dataframe(summary, use_container_width=True)
