# ============================================================
# Kitti_2nd.py — Stable Portfolio Weekly Stock Predictor
# Fixes:
#  1) NaN in y (train on combined X+y then dropna once)
#  2) PRED_HORIZON string bug (explicit mapping)
#  3) Series formatting error (force weekly to Series + float())
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
    ["1 Week", "1 Month", "3 Months", "6 Months", "12 Months"],
)

HORIZON_MAP = {
    "1 Week": 1,
    "1 Month": 4,
    "3 Months": 12,
    "6 Months": 26,
    "12 Months": 52,
}
PRED_HORIZON = int(HORIZON_MAP[HORIZON_LABEL])  # ALWAYS int

BUY_TH = st.sidebar.slider("BUY threshold (%)", 1.0, 5.0, 2.0, 0.5)
SELL_TH = st.sidebar.slider("SELL threshold (%)", 1.0, 5.0, 2.0, 0.5)

MIN_WEEKS = st.sidebar.slider("Minimum weekly samples", 52, 200, 80, 4)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def ensure_series(x) -> pd.Series:
    """Force input to be a Series (yfinance sometimes returns 1-col DataFrame)."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        return x.iloc[:, 0]
    return pd.Series(x)

def make_features(close: pd.Series) -> pd.DataFrame:
    close = ensure_series(close).astype(float)
    df = pd.DataFrame(index=close.index)
    df["ret1"] = close.pct_change()
    df["ret4"] = close.pct_change(4)
    df["ret8"] = close.pct_change(8)
    df["vol12"] = df["ret1"].rolling(12).std()
    df["mom12"] = close / (close.shift(12) + 1e-9) - 1
    return df

def signal(last: float, pred: float) -> str:
    pct = (pred - last) / last * 100 if last > 0 else np.nan
    if np.isnan(pct):
        return "N/A"
    if pct >= BUY_TH:
        return "BUY"
    if pct <= -SELL_TH:
        return "SELL"
    return "HOLD"

# ------------------------------------------------------------
# TRAIN + FORECAST (NaN-safe)
# ------------------------------------------------------------
def train_predict(weekly: pd.Series):
    weekly = ensure_series(weekly).astype(float).dropna()

    feats = make_features(weekly)
    target = weekly.shift(-1)

    data = feats.copy()
    data["y"] = target
    data = data.dropna()  # drop once, after combining X and y

    if len(data) < MIN_WEEKS:
        return None

    X = data.drop(columns="y")
    y = data["y"]

    # Double-safety: eliminate any accidental NaN in y (should be none after dropna)
    if y.isna().any():
        return None

    model = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42,
    )
    model.fit(X, y)

    # Recursive multi-step forecast
    preds = []
    hist = weekly.copy()

    for _ in range(PRED_HORIZON):
        f = make_features(hist).iloc[[-1]]
        # If f has NaNs (e.g., not enough history), stop safely
        if f.isna().any(axis=1).iloc[0]:
            break
        p = float(model.predict(f)[0])
        preds.append(p)
        hist.loc[hist.index[-1] + pd.offsets.Week(1)] = p

    return preds if preds else None

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
for name, symbol in PORTFOLIO.items():
    st.markdown("---")
    st.subheader(f"{name} ({symbol})")

    try:
        df = yf.download(symbol, period="max", interval="1d", progress=False)
    except Exception as e:
        st.warning(f"Download failed: {e}")
        continue

    if df is None or df.empty:
        st.warning("No data returned.")
        continue

    # yfinance sometimes yields MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Close" not in df.columns:
        st.warning("Close column missing.")
        continue

    close = ensure_series(df["Close"]).dropna()
    if close.empty:
        st.warning("Empty Close series.")
        continue

    weekly = ensure_series(close.resample("W-FRI").last()).dropna()
    if len(weekly) < MIN_WEEKS:
        st.warning(f"Not enough weekly data: {len(weekly)} weeks.")
        continue

    preds = train_predict(weekly)
    if preds is None:
        st.warning("Model not trained (insufficient clean samples).")
        continue

    last = float(weekly.iloc[-1])
    final_pred = float(preds[-1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Last Close", f"{last:.2f}")
    c2.metric("Predicted (Horizon End)", f"{final_pred:.2f}")
    c3.metric("Signal", signal(last, final_pred))

    # Plot actual + predicted on the same chart
    future_idx = pd.date_range(
        start=weekly.index[-1] + pd.offsets.Week(1),
        periods=len(preds),
        freq="W-FRI",
    )

    plot_df = pd.concat(
        [
            pd.DataFrame({"Actual": weekly}),
            pd.DataFrame({"Predicted": preds}, index=future_idx),
        ],
        axis=0,
    )

    st.line_chart(plot_df)
