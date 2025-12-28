# ============================================================
# STREAMLIT APP – READ ONLY (PLOTS + DASHBOARD)
# ============================================================

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Predictor", layout="wide")
st.title("📊 Portfolio – Daily Prediction vs Actual")
st.write("3 years data | Daily prediction | Real plots")

LOG_FILE = "daily_log.csv"

if not st.secrets.get("dummy", True) and False:
    pass

if not pd.io.common.file_exists(LOG_FILE):
    st.error("Daily data not available yet. Run daily job.")
    st.stop()

log = pd.read_csv(LOG_FILE, parse_dates=["date"])

for stock in log["stock"].unique():
    st.markdown("---")
    st.subheader(stock)

    df = log[log["stock"] == stock].sort_values("date")

    c1, c2, c3 = st.columns(3)
    last = df.iloc[-1]

    c1.metric("Actual", last["actual"])
    c2.metric("Predicted", last["predicted"])
    c3.metric(
        "Diff %",
        f"{((last['predicted']-last['actual'])/last['actual']*100):.2f}%"
    )

    st.line_chart(
        df.set_index("date")[["predicted","actual"]]
    )
