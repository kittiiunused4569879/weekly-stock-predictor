# ============================================================
# Kapp.py — Portfolio Weekly Stock Predictor (Gradient Boosting)
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
    "YESBANK": "YESBANK.NS",
}

# ------------------------------------------------------------
# CONTROLS
# ------------------------------------------------------------
BUY_THRESHOLD = st.slider("BUY threshold (+%)", 0.5, 10.0, 2.0, 0.5)
SELL_THRESHOLD = st.slider("SELL threshold (-%)", 0.5, 10.0, 2.0, 0.5)
MIN_WEEKS = st.slider("Minimum weekly samples", 52, 200, 80, 4)

with st.expander("Model parameters"):
    N_EST = st.slider("n_estimators", 100, 800, 300, 50)
    LR = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.08, 0.1, 0.2], value=0.05)
    MAX_DEPTH = st.slider("max_depth", 1, 6, 3, 1)
    SUBSAMPLE = st.select_slider("subsample", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)

# ------------------------------------------------------------
# HELPERS (CRITICAL: normalize to Series)
# ------------------------------------------------------------
def ensure_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0] if x.shape[1] else pd.Series(dtype=float)
    return pd.Series(x)

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------
def make_features(close: pd.Series) -> pd.DataFrame:
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

    f["mom4"] = close / (close.shift(4) + 1e-9) - 1
    f["mom12"] = close / (close.shift(12) + 1e-9) - 1

    return f

# ------------------------------------------------------------
# TRAIN + PREDICT
# ------------------------------------------------------------
def train_predict(weekly_close: pd.Series):
    weekly_close = ensure_series(weekly_close).astype(float).dropna()
    if len(weekly_close) < MIN_WEEKS:
        return None

    feats = make_features(weekly_close)
    y_next = weekly_close.shift(-1)

    data = feats.copy()
    data["y"] = y_next
    data = data.dropna()

    if len(data) < MIN_WEEKS:
        return None

    X = data.drop(columns=["y"])
    y = data["y"]

    model = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], preds))

    acc = max(0.0, 1.0 - float(np.mean(mape_scores)))

    model.fit(X, y)
    pred_next = float(model.predict(X.iloc[[-1]])[0])
    return pred_next, acc

def signal(last_price: float, pred_price: float) -> str:
    if not np.isfinite(last_price) or last_price <= 0 or not np.isfinite(pred_price):
        return "N/A"
    pct = (pred_price - last_price) / last_price * 100
    if pct >= BUY_THRESHOLD:
        return "BUY"
    if pct <= -SELL_THRESHOLD:
        return "SELL"
    return "HOLD"

# ------------------------------------------------------------
# CACHE COMPUTE (prevents Streamlit rerun issues)
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_portfolio(items):
    results = []

    for name, symbol in items:
        try:
            df = yf.download(symbol, period="3y", interval="1d", progress=False)

            if df is None or df.empty:
                results.append({"name": name, "symbol": symbol, "error": "No data"})
                continue

            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            if "Close" not in df.columns:
                results.append({"name": name, "symbol": symbol, "error": "Close missing"})
                continue

            daily = ensure_series(df["Close"]).dropna()
            if daily.empty:
                results.append({"name": name, "symbol": symbol, "error": "Empty Close"})
                continue

            weekly = ensure_series(daily.resample("W-FRI").last()).dropna()
            if len(weekly) < MIN_WEEKS:
                results.append({"name": name, "symbol": symbol, "error": f"Only {len(weekly)} weeks"})
                continue

            out = train_predict(weekly)
            if out is None:
                results.append({"name": name, "symbol": symbol, "error": "Model not trained"})
                continue

            pred, acc = out
            last = float(weekly.iloc[-1])
            pct = (pred - last) / last * 100 if last > 0 else np.nan

            results.append({
                "name": name,
                "symbol": symbol,
                "daily": daily,
                "weekly": weekly,     # ALWAYS Series
                "pred": float(pred),
                "last": float(last),
                "acc": float(acc),
                "pct": float(pct),
                "signal": signal(last, pred),
                "error": None,
            })

        except Exception as e:
            results.append({"name": name, "symbol": symbol, "error": str(e)})

    return results

# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------
results = compute_portfolio(list(PORTFOLIO.items()))
summary_rows = []

for r in results:
    st.markdown("---")
    st.subheader(f"{r['name']} ({r['symbol']})")

    if r["error"]:
        st.warning(r["error"])
        continue

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Week Close", f"{r['last']:.2f}")
    c2.metric("Predicted Next Week", f"{r['pred']:.2f}")
    c3.metric("Change %", f"{r['pct']:.2f}%")
    c4.metric("Accuracy (1 - MAPE)", f"{r['acc']*100:.2f}%")

    st.write(f"**Signal:** {r['signal']}")

    left, right = st.columns(2)
    with left:
        st.caption(" Daily Close (3 Years)")
        st.line_chart(r["daily"])

    with right:
        st.caption(" Weekly Close + Prediction")
        pw = r["weekly"].to_frame(name="Actual")   # SAFE
        pw["Prediction"] = np.nan
        pw.iloc[-1, pw.columns.get_loc("Prediction")] = r["pred"]
        st.line_chart(pw)

    summary_rows.append({
        "Stock": r["name"],
        "Symbol": r["symbol"],
        "Last": round(r["last"], 2),
        "Predicted": round(r["pred"], 2),
        "Change %": round(r["pct"], 2),
        "Signal": r["signal"],
        "Accuracy %": round(r["acc"] * 100, 2),
    })

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
st.markdown("---")
st.header(" Portfolio Summary")

if summary_rows:
    df_sum = pd.DataFrame(summary_rows)
    order = {"BUY": 0, "HOLD": 1, "SELL": 2, "N/A": 3}
    df_sum["SignalRank"] = df_sum["Signal"].map(order).fillna(99).astype(int)
    df_sum = df_sum.sort_values(["SignalRank", "Change %"], ascending=[True, False]).drop(columns="SignalRank")
    st.dataframe(df_sum, use_container_width=True)
else:
    st.info("No stocks produced results.")
