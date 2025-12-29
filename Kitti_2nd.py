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
# FEATURE ENGINEERING
# ------------------------------------------------------------
def make_features(close: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index)
    f["ret1"] = close.pct_change(1)
    f["ret4"] = close.pct_change(4)
    f["ret8"] = close.pct_change(8)
    f["vol12"] = f["ret1"].rolling(12).std()
    f["mom12"] = close / close.shift(12) - 1
    return f

# ------------------------------------------------------------
# BUY / HOLD / SELL SIGNAL
# ------------------------------------------------------------
def signal(last, pred):
    pct = (pred - last) / last * 100
    if pct >= BUY_TH:
        return "BUY"
    if pct <= -SELL_TH:
        return "SELL"
    return "HOLD"

# ------------------------------------------------------------
# TRAIN + FORECAST (NaN-SAFE)
# ------------------------------------------------------------
def train_predict(weekly):
    feats = make_features(weekly)
    target = weekly.shift(-1)

    data = feats.copy()
    data["y"] = target
    data = data.dropna()

    if len(data) < MIN_WEEKS:
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

    residuals = y - model.predict(X)
    sigma = residuals.std()

    preds = []
    hist = weekly.copy()

    for _ in range(PRED_HORIZON):
        f = make_features(hist).iloc[[-1]]
        p = float(model.predict(f)[0])
        preds.append(p)
        hist.loc[hist.index[-1] + pd.offsets.Week(1)] = p

    return preds, sigma

# ------------------------------------------------------------
# DIRECTIONAL ACCURACY (NaN-SAFE)
# ------------------------------------------------------------
def directional_accuracy(series, horizon):
    feats = make_features(series)
    target = series.shift(-horizon)

    data = feats.copy()
    data["y"] = target
    data = data.dropna()

    correct, total = 0, 0

    for i in range(100, len(data)):
        train = data.iloc[:i]
        X = train.drop(columns="y")
        y = train["y"]

        model = GradientBoostingRegressor(
            n_estimators=N_EST,
            learning_rate=LR,
            max_depth=MAX_DEPTH,
            subsample=SUBSAMPLE,
            random_state=42,
        )
        model.fit(X, y)

        pred = model.predict(X.iloc[[-1]])[0]

        last_price = series.loc[X.index[-1]]
        actual_price = series.loc[X.index[-1] + horizon * pd.offsets.Week(1)]

        if np.sign(actual_price - last_price) == np.sign(pred - last_price):
            correct += 1
        total += 1

    return correct / total if total > 0 else np.nan

# ------------------------------------------------------------
# SHARPE SCORE
# ------------------------------------------------------------
def sharpe_score(weekly, predicted_price):
    returns = weekly.pct_change().dropna()
    vol = returns.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    exp_ret = (predicted_price - weekly.iloc[-1]) / weekly.iloc[-1]
    return exp_ret / vol

# ------------------------------------------------------------
# MAIN COMPUTE (CACHED)
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

        out = train_predict(weekly)
        if out is None:
            continue

        preds, sigma = out
        last = weekly.iloc[-1]
        final_pred = preds[-1]

        sig = signal(last, final_pred)

        strat_ret = weekly.pct_change().shift(-1)
        strat_ret = strat_ret * (1 if sig == "BUY" else -1 if sig == "SELL" else 0)
        strat_ret = strat_ret.dropna()

        equity = (1 + strat_ret).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak

        results.append({
            "name": name,
            "weekly": weekly,
            "preds": preds,
            "upper": [p + 1.96 * sigma for p in preds],
            "lower": [p - 1.96 * sigma for p in preds],
            "signal": sig,
            "dir_acc": directional_accuracy(weekly, PRED_HORIZON),
            "sharpe": sharpe_score(weekly, final_pred),
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

    st.write("Signal:", r["signal"])
    st.metric("Directional Accuracy", f"{r['dir_acc']*100:.2f}%")
    st.metric("Sharpe Score", f"{r['sharpe']:.2f}")
    st.metric("Max Drawdown", f"{r['max_dd']*100:.2f}%")

    wk = r["weekly"]
    future_idx = pd.date_range(
        wk.index[-1] + pd.offsets.Week(1),
        periods=len(r["preds"]),
        freq="W-FRI",
    )

    price_df = pd.concat([
        pd.DataFrame({"Actual": wk}),
        pd.DataFrame({
            "Predicted": r["preds"],
            "Upper Band": r["upper"],
            "Lower Band": r["lower"],
        }, index=future_idx),
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.caption("📈 Price Forecast + Confidence Bands")
        st.line_chart(price_df)

    with c2:
        st.caption("📊 Equity Curve")
        st.line_chart(r["equity"])

    st.caption("📉 Drawdown Curve")
    st.line_chart(r["drawdown"])

# ------------------------------------------------------------
# SHARPE RANKING
# ------------------------------------------------------------
st.markdown("---")
st.header("📈 Risk-Adjusted Ranking (Sharpe)")

rank_df = pd.DataFrame([{
    "Stock": r["name"],
    "Signal": r["signal"],
    "Sharpe": r["sharpe"],
    "Directional Accuracy (%)": r["dir_acc"] * 100,
    "Max Drawdown (%)": r["max_dd"] * 100,
} for r in results])

rank_df = rank_df.sort_values("Sharpe", ascending=False)
st.dataframe(rank_df, use_container_width=True)
