# ============================================================
# PORTFOLIO WEEKLY STOCK PREDICTOR (GRADIENT BOOSTING) - HARDENED
# - Next-week numeric prediction (weekly close)
# - Accuracy via time-series CV (MAPE -> Accuracy = 1 - MAPE)
# - BUY/HOLD/SELL signal
# - Robust loop: one failure won't stop others
# - Robust plots: no scalar DataFrame crash
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Portfolio Weekly Predictor (GB)", layout="wide")
st.title("📊 Portfolio Weekly Predictor – Gradient Boosting")
st.write("Next-week prediction + accuracy + BUY/HOLD/SELL + plots (3y real data).")

# -----------------------------
# PORTFOLIO
# -----------------------------
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

# -----------------------------
# SETTINGS
# -----------------------------
BUY_THRESHOLD_PCT = st.slider("BUY threshold (+%)", 0.5, 10.0, 2.0, 0.5)
SELL_THRESHOLD_PCT = st.slider("SELL threshold (-%)", 0.5, 10.0, 2.0, 0.5)
MIN_WEEKS_REQUIRED = st.slider("Min weekly samples required", 52, 200, 80, 4)

with st.expander("Model parameters (optional)"):
    N_ESTIMATORS = st.slider("n_estimators", 100, 1000, 300, 50)
    LEARNING_RATE = st.select_slider(
        "learning_rate",
        options=[0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
        value=0.05
    )
    MAX_DEPTH = st.slider("max_depth", 1, 6, 3, 1)
    SUBSAMPLE = st.select_slider("subsample", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)

# -----------------------------
# UTIL: SAFE FLOAT
# -----------------------------
def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (np.ndarray, list, tuple)):
            x = np.asarray(x).ravel()[0]
        return float(x)
    except Exception:
        return float(default)

# -----------------------------
# FEATURE ENGINEERING (WEEKLY)
# -----------------------------
def make_features(close: pd.Series) -> pd.DataFrame:
    # Ensure Series
    close = pd.Series(close).astype(float)

    f = pd.DataFrame(index=close.index)
    # returns
    f["ret1"] = close.pct_change(1)
    f["ret2"] = close.pct_change(2)
    f["ret4"] = close.pct_change(4)
    f["ret8"] = close.pct_change(8)

    # moving averages
    ma4 = close.rolling(4).mean()
    ma8 = close.rolling(8).mean()
    ma12 = close.rolling(12).mean()
    ma20 = close.rolling(20).mean()

    f["ma4_ma8"] = (ma4 / (ma8 + 1e-9)) - 1
    f["ma8_ma12"] = (ma8 / (ma12 + 1e-9)) - 1
    f["ma12_ma20"] = (ma12 / (ma20 + 1e-9)) - 1

    # volatility
    f["vol8"] = f["ret1"].rolling(8).std()
    f["vol12"] = f["ret1"].rolling(12).std()

    # momentum
    f["mom4"] = close / (close.shift(4) + 1e-9) - 1
    f["mom12"] = close / (close.shift(12) + 1e-9) - 1

    return f

# -----------------------------
# MODEL + ACCURACY + PREDICTION
# -----------------------------
def fit_predict_with_accuracy(weekly_close: pd.Series):
    weekly_close = pd.Series(weekly_close).astype(float).dropna()
    if len(weekly_close) < MIN_WEEKS_REQUIRED:
        return None

    feats = make_features(weekly_close)
    y_next = weekly_close.shift(-1)  # next-week close

    data = feats.copy()
    data["y"] = y_next
    data = data.dropna()

    # after feature engineering, need enough rows
    if len(data) < MIN_WEEKS_REQUIRED:
        return None

    X = data.drop(columns=["y"])
    y = data["y"]

    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42
    )

    # TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []

    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        # MAPE can blow up if y has zeros; close prices are > 0, but guard anyway
        mape = mean_absolute_percentage_error(y.iloc[test_idx], preds)
        mape_scores.append(mape)

    mean_mape = safe_float(np.mean(mape_scores), default=np.nan)
    accuracy = float(max(0.0, 1.0 - mean_mape)) if np.isfinite(mean_mape) else np.nan

    # Fit full and predict next
    model.fit(X, y)
    next_week_pred = safe_float(model.predict(X.iloc[[-1]]), default=np.nan)

    return next_week_pred, accuracy

def signal_from_pred(last_price: float, pred_price: float) -> str:
    if not np.isfinite(last_price) or not np.isfinite(pred_price) or last_price <= 0:
        return "N/A"
    change_pct = (pred_price - last_price) / last_price * 100.0
    if change_pct >= BUY_THRESHOLD_PCT:
        return "BUY"
    if change_pct <= -SELL_THRESHOLD_PCT:
        return "SELL"
    return "HOLD"

# -----------------------------
# MAIN LOOP (HARDENED)
# -----------------------------
summary_rows = []
progress = st.progress(0)
total = len(PORTFOLIO)
idx = 0

for stock_name, symbol in PORTFOLIO.items():
    idx += 1
    progress.progress(int(idx / total * 100))

    try:
        st.markdown("---")
        st.subheader(f"{stock_name} ({symbol})")

        # Fetch daily
        df_daily = yf.download(symbol, period="3y", interval="1d", progress=False)

        if df_daily is None or df_daily.empty:
            st.warning("No data received from Yahoo Finance for this symbol.")
            continue

        # Ensure Close exists (sometimes yfinance gives multiindex columns)
        if "Close" not in df_daily.columns:
            # try to flatten if multiindex
            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = [c[0] for c in df_daily.columns]
            if "Close" not in df_daily.columns:
                st.warning("Close column missing for this symbol.")
                continue

        df_daily = df_daily[["Close"]].dropna()
        if df_daily.empty:
            st.warning("No usable Close values after dropna.")
            continue

        # Weekly close
        weekly_close = df_daily["Close"].resample("W").last().dropna()
        if weekly_close.empty or len(weekly_close) < MIN_WEEKS_REQUIRED:
            st.warning(f"Not enough weekly data ({len(weekly_close)} weeks).")
            continue

        # Model
        out = fit_predict_with_accuracy(weekly_close)
        if out is None:
            st.warning("Not enough usable samples after feature engineering.")
            continue

        pred_price, acc = out
        last_week_price = safe_float(weekly_close.iloc[-1], default=np.nan)

        sig = signal_from_pred(last_week_price, pred_price)
        change_pct = ((pred_price - last_week_price) / last_week_price * 100.0) if (
            np.isfinite(pred_price) and np.isfinite(last_week_price) and last_week_price > 0
        ) else np.nan

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Week Close", "N/A" if not np.isfinite(last_week_price) else f"{last_week_price:.2f}")
        c2.metric("Predicted Next Week", "N/A" if not np.isfinite(pred_price) else f"{pred_price:.2f}")
        c3.metric("Predicted Change %", "N/A" if not np.isfinite(change_pct) else f"{change_pct:.2f}%")
        c4.metric("Accuracy (1 - MAPE)", "N/A" if not np.isfinite(acc) else f"{acc*100:.2f}%")

        st.write(f"**Signal:** {sig}")

        # Plots (never crash)
        left, right = st.columns(2)

        with left:
            st.caption("📈 Daily Close (Last 3 Years)")
            st.line_chart(df_daily["Close"])

        with right:
            st.caption("📊 Weekly Close + Next-week Prediction Point")
            plot_weekly = weekly_close.to_frame(name="Actual")  # SAFE construction
            plot_weekly["Prediction"] = np.nan
            if np.isfinite(pred_price) and len(plot_weekly) > 0:
                plot_weekly.iloc[-1, plot_weekly.columns.get_loc("Prediction")] = pred_price
            st.line_chart(plot_weekly)

        summary_rows.append({
            "Stock": stock_name,
            "Symbol": symbol,
            "Last Week": None if not np.isfinite(last_week_price) else round(last_week_price, 2),
            "Pred Next Week": None if not np.isfinite(pred_price) else round(pred_price, 2),
            "Change %": None if not np.isfinite(change_pct) else round(change_pct, 2),
            "Signal": sig,
            "Accuracy %": None if not np.isfinite(acc) else round(acc * 100, 2)
        })

    except Exception as e:
        # This ensures loop continues for other stocks
        st.error(f"{stock_name} failed: {e}")
        continue

progress.progress(100)

# -----------------------------
# SUMMARY TABLE
# -----------------------------
st.markdown("---")
st.header("📌 Portfolio Summary")

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)

    # Sort BUY first, HOLD, then SELL, then N/A
    order = {"BUY": 0, "HOLD": 1, "SELL": 2, "N/A": 3}
    summary_df["SignalRank"] = summary_df["Signal"].map(order).fillna(99).astype(int)
    summary_df = summary_df.sort_values(by=["SignalRank", "Change %"], ascending=[True, False]).drop(columns=["SignalRank"])

    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("No stocks produced results. Check symbols or try lowering min weekly samples.")
