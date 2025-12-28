# ============================================================
# PORTFOLIO WEEKLY STOCK PREDICTOR – GRADIENT BOOSTING (STABLE)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Portfolio Weekly Predictor (GB)", layout="wide")
st.title("📊 Portfolio Weekly Predictor – Gradient Boosting")
st.write("Next-week prediction + accuracy + BUY/HOLD/SELL (3 years real data)")

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
    "YESBANK": "YESBANK.NS"
}

# ------------------------------------------------------------
# CONTROLS
# ------------------------------------------------------------
BUY_THRESHOLD = st.slider("BUY threshold (+%)", 0.5, 10.0, 2.0, 0.5)
SELL_THRESHOLD = st.slider("SELL threshold (-%)", 0.5, 10.0, 2.0, 0.5)
MIN_WEEKS = st.slider("Minimum weekly samples", 52, 200, 80, 4)

with st.expander("Model parameters"):
    N_EST = st.slider("n_estimators", 100, 800, 300, 50)
    LR = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1], value=0.05)
    MAX_DEPTH = st.slider("max_depth", 1, 5, 3)
    SUBSAMPLE = st.select_slider("subsample", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
def make_features(close: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index)
    f["ret1"] = close.pct_change(1)
    f["ret2"] = close.pct_change(2)
    f["ret4"] = close.pct_change(4)
    f["mom4"] = close / close.shift(4) - 1
    f["mom12"] = close / close.shift(12) - 1
    f["vol8"] = f["ret1"].rolling(8).std()
    return f

# ------------------------------------------------------------
# MODEL TRAIN + PREDICT
# ------------------------------------------------------------
def train_predict(weekly_close: pd.Series):
    feats = make_features(weekly_close)
    y = weekly_close.shift(-1)

    data = feats.copy()
    data["y"] = y
    data.dropna(inplace=True)

    if len(data) < MIN_WEEKS:
        return None

    X = data.drop(columns="y")
    y = data["y"]

    model = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], preds))

    accuracy = max(0.0, 1.0 - float(np.mean(mape_scores)))

    model.fit(X, y)
    pred_next = float(model.predict(X.iloc[[-1]])[0])

    return pred_next, accuracy

def signal(last_price, pred_price):
    pct = (pred_price - last_price) / last_price * 100
    if pct >= BUY_THRESHOLD:
        return "BUY"
    if pct <= -SELL_THRESHOLD:
        return "SELL"
    return "HOLD"

# ------------------------------------------------------------
# CACHE COMPUTATION (THIS FIXES YOUR LOOP ISSUE)
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_portfolio():
    results = []

    for name, symbol in PORTFOLIO.items():
        try:
            df = yf.download(symbol, period="3y", interval="1d", progress=False)
            if df.empty or "Close" not in df:
                continue

            daily = df["Close"].dropna()
            weekly = daily.resample("W-FRI").last().dropna()

            if len(weekly) < MIN_WEEKS:
                continue

            out = train_predict(weekly)
            if out is None:
                continue

            pred, acc = out
            last = float(weekly.iloc[-1])

            results.append({
                "name": name,
                "symbol": symbol,
                "daily": daily,
                "weekly": weekly,
                "pred": pred,
                "last": last,
                "acc": acc,
                "signal": signal(last, pred)
            })

        except Exception:
            continue

    return results

# ------------------------------------------------------------
# RENDER UI
# ------------------------------------------------------------
results = compute_portfolio()

summary = []

for r in results:
    st.markdown("---")
    st.subheader(f"{r['name']} ({r['symbol']})")

    pct = (r["pred"] - r["last"]) / r["last"] * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Week Close", f"{r['last']:.2f}")
    c2.metric("Predicted Next Week", f"{r['pred']:.2f}")
    c3.metric("Change %", f"{pct:.2f}%")
    c4.metric("Accuracy", f"{r['acc']*100:.2f}%")

    st.write(f"**Signal:** {r['signal']}")

    left, right = st.columns(2)
    with left:
        st.caption("📈 Daily Close (3 Years)")
        st.line_chart(r["daily"])

    with right:
        st.caption("📊 Weekly Close + Prediction")
        pw = r["weekly"].to_frame("Actual")
        pw["Prediction"] = np.nan
        pw.iloc[-1, 1] = r["pred"]
        st.line_chart(pw)

    summary.append({
        "Stock": r["name"],
        "Last": round(r["last"], 2),
        "Predicted": round(r["pred"], 2),
        "Change %": round(pct, 2),
        "Signal": r["signal"],
        "Accuracy %": round(r["acc"] * 100, 2)
    })

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
st.markdown("---")
st.header("📌 Portfolio Summary")

if summary:
    df_sum = pd.DataFrame(summary).sort_values("Change %", ascending=False)
    st.dataframe(df_sum, use_container_width=True)
else:
    st.warning("No stocks produced results.")
