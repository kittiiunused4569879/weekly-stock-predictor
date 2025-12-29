# ============================================================
# Kapp.py — Full Portfolio + Interactive Chart (Plotly)
# ML: Gradient Boosting + Random Forest Ensemble (Price-based)
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
st.set_page_config(page_title="Portfolio Weekly Predictor (Interactive)", layout="wide")
st.title("📊 Portfolio Weekly Predictor — Interactive")
st.write("Full portfolio summary + click/select a stock for interactive zoomable chart + dynamic horizon forecast.")

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
    HORIZON_LABEL = st.selectbox("Prediction Horizon", ["1 Week", "4 Weeks", "12 Weeks"], index=0)
with c2:
    BUY_TH = st.slider("BUY threshold (+%)", 0.5, 10.0, 2.0, 0.5)
with c3:
    SELL_TH = st.slider("SELL threshold (-%)", 0.5, 10.0, 2.0, 0.5)

HORIZON_MAP = {"1 Week": 1, "4 Weeks": 4, "12 Weeks": 12}
PRED_HORIZON = int(HORIZON_MAP[HORIZON_LABEL])

MIN_WEEKS = st.slider("Minimum weekly samples", 52, 200, 80, 4)

with st.expander("Model parameters"):
    N_EST = st.slider("n_estimators (both models)", 100, 600, 300, 50)
    LR = st.select_slider("learning_rate (GB)", [0.01, 0.03, 0.05, 0.08, 0.1, 0.2], value=0.05)
    GB_DEPTH = st.slider("max_depth (GB)", 1, 6, 3, 1)
    RF_DEPTH = st.slider("max_depth (RF)", 2, 20, 10, 1)
    SUBSAMPLE = st.select_slider("subsample (GB)", [0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def ensure_series(x):
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0] if x.shape[1] else pd.Series(dtype=float)
    return pd.Series(x)

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

def signal(last_price: float, pred_price: float) -> str:
    if not np.isfinite(last_price) or last_price <= 0 or not np.isfinite(pred_price):
        return "N/A"
    pct = (pred_price - last_price) / last_price * 100
    if pct >= BUY_TH:
        return "BUY"
    if pct <= -SELL_TH:
        return "SELL"
    return "HOLD"

def signal_color(sig: str) -> str:
    if sig == "BUY":
        return "lime"
    if sig == "SELL":
        return "red"
    return "orange"

# ------------------------------------------------------------
# TRAIN + CV ACCURACY + MULTI-STEP FORECAST (ENSEMBLE)
# ------------------------------------------------------------
def train_predict_ensemble(weekly_close: pd.Series):
    weekly_close = ensure_series(weekly_close).astype(float).dropna()
    if len(weekly_close) < MIN_WEEKS:
        return None

    feats = make_features(weekly_close)
    y_future = weekly_close.shift(-PRED_HORIZON)

    data = feats.copy()
    data["y"] = y_future
    data = data.dropna()

    if len(data) < MIN_WEEKS:
        return None

    X = data.drop(columns=["y"])
    y = data["y"]

    gb = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=GB_DEPTH,
        subsample=SUBSAMPLE,
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=N_EST,
        max_depth=RF_DEPTH,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validated accuracy (1 - MAPE)
    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []

    for tr, te in tscv.split(X):
        gb.fit(X.iloc[tr], y.iloc[tr])
        rf.fit(X.iloc[tr], y.iloc[tr])
        pred_te = (gb.predict(X.iloc[te]) + rf.predict(X.iloc[te])) / 2.0
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], pred_te))

    acc = max(0.0, 1.0 - float(np.mean(mape_scores)))

    # Fit full + recursive multi-step forecast (weekly)
    gb.fit(X, y)
    rf.fit(X, y)

    preds = []
    hist = weekly_close.copy()

    for _ in range(PRED_HORIZON):
        f_last = make_features(hist).iloc[[-1]]
        p = float((gb.predict(f_last)[0] + rf.predict(f_last)[0]) / 2.0)
        preds.append(p)
        hist.loc[hist.index[-1] + pd.offsets.Week(1)] = p

    return preds, acc

# ------------------------------------------------------------
# DATA DOWNLOAD (CACHED)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_daily(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

# ------------------------------------------------------------
# COMPUTE PORTFOLIO (CACHED, depends on controls)
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_portfolio(items, pred_horizon, min_weeks, n_est, lr, gb_depth, rf_depth, subsample, buy_th, sell_th):
    results = []
    for name, symbol in items:
        try:
            df = load_daily(symbol)
            if df.empty or "Close" not in df.columns:
                results.append({"name": name, "symbol": symbol, "error": "No data"})
                continue

            close = ensure_series(df["Close"]).dropna()
            weekly = ensure_series(close.resample("W-FRI").last()).dropna()

            if len(weekly) < min_weeks:
                results.append({"name": name, "symbol": symbol, "error": f"Only {len(weekly)} weeks"})
                continue

            out = train_predict_ensemble(weekly)
            if out is None:
                results.append({"name": name, "symbol": symbol, "error": "Model not trained"})
                continue

            preds, acc = out
            last = float(weekly.iloc[-1])
            final_pred = float(preds[-1])
            pct = (final_pred - last) / last * 100 if last > 0 else np.nan
            sig = signal(last, final_pred)

            results.append({
                "name": name,
                "symbol": symbol,
                "weekly": weekly,          # Series
                "preds": preds,            # list of future prices
                "pred": final_pred,        # horizon-end prediction
                "last": last,
                "pct": float(pct),
                "acc": float(acc),
                "signal": sig,
                "error": None,
            })

        except Exception as e:
            results.append({"name": name, "symbol": symbol, "error": str(e)})

    return results

# ------------------------------------------------------------
# RUN COMPUTE
# ------------------------------------------------------------
results = compute_portfolio(
    list(PORTFOLIO.items()),
    PRED_HORIZON,
    MIN_WEEKS,
    N_EST, LR, GB_DEPTH, RF_DEPTH, SUBSAMPLE,
    BUY_TH, SELL_TH
)

# ------------------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------------------
st.markdown("---")
st.header("📌 Portfolio Summary")

summary_rows = []
for r in results:
    if r.get("error"):
        summary_rows.append({
            "Stock": r["name"],
            "Symbol": r["symbol"],
            "Signal": "N/A",
            "Last": np.nan,
            "Predicted": np.nan,
            "Change %": np.nan,
            "Accuracy %": np.nan,
            "Error": r["error"]
        })
    else:
        summary_rows.append({
            "Stock": r["name"],
            "Symbol": r["symbol"],
            "Signal": r["signal"],
            "Last": round(r["last"], 2),
            "Predicted": round(r["pred"], 2),
            "Change %": round(r["pct"], 2),
            "Accuracy %": round(r["acc"] * 100, 2),
            "Error": ""
        })

df_sum = pd.DataFrame(summary_rows)

# Rank: BUY first, then HOLD, then SELL, then N/A
order = {"BUY": 0, "HOLD": 1, "SELL": 2, "N/A": 3}
df_sum["SignalRank"] = df_sum["Signal"].map(order).fillna(99).astype(int)
df_sum = df_sum.sort_values(["SignalRank", "Change %"], ascending=[True, False]).drop(columns="SignalRank")

st.dataframe(df_sum, use_container_width=True)

# ------------------------------------------------------------
# STOCK SELECTION FOR INTERACTIVE CHART
# ------------------------------------------------------------
valid_names = [r["name"] for r in results if not r.get("error")]
st.markdown("---")
st.header("📈 Interactive Chart (Zoom / Pan)")

if not valid_names:
    st.info("No valid stocks available to plot.")
    st.stop()

selected = st.selectbox("Select a stock to view chart", valid_names)
r = next(x for x in results if x["name"] == selected)

# ------------------------------------------------------------
# METRICS (TOP)
# ------------------------------------------------------------
sig = r["signal"]
color = signal_color(sig)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last Close", f"{r['last']:.2f}")
m2.metric(f"Predicted ({HORIZON_LABEL})", f"{r['pred']:.2f}")
m3.metric("Accuracy (1 − MAPE)", f"{r['acc']*100:.2f}%")
m4.markdown(f"<h3 style='color:{color}'>{sig}</h3>", unsafe_allow_html=True)

# ------------------------------------------------------------
# INTERACTIVE PLOTLY CHART
# ------------------------------------------------------------
weekly = r["weekly"]
preds = r["preds"]

# show recent window but allow zoom out via range slider
plot_weeks = 120
hist_plot = weekly.iloc[-plot_weeks:]

future_idx = pd.date_range(
    hist_plot.index[-1] + pd.offsets.Week(1),
    periods=len(preds),
    freq="W-FRI",
)

fig = go.Figure()

# Actual = GREEN
fig.add_trace(go.Scatter(
    x=hist_plot.index,
    y=hist_plot.values,
    mode="lines",
    name="Actual",
    line=dict(color="lime", width=2)
))

# Forecast = ORANGE
fig.add_trace(go.Scatter(
    x=future_idx,
    y=preds,
    mode="lines+markers",
    name="Forecast",
    line=dict(color="orange", width=3)
))

fig.update_layout(
    title=f"{selected} ({r['symbol']}) — {HORIZON_LABEL} Forecast — {sig}",
    template="plotly_dark",
    hovermode="x unified",
    height=550,
    xaxis=dict(rangeslider=dict(visible=True)),
)

st.plotly_chart(fig, use_container_width=True)
