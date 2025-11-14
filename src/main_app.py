# src/main_app.py
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

import yfinance as yf

from tech_indicators import calculate_technical_indicators
from gmm_model import run_gmm_model
from plot_results import plot_results, plot_results_sub

# ==============================
# App config
# ==============================
st.set_page_config(
    page_title="Probabilistic Stock Forecasting",
    page_icon="📈",
    layout="wide"
)

ROOT = Path(__file__).parent.parent

# ==============================
# Cache: artifacts
# ==============================
@st.cache_resource

# Project root (one level above src/)
def load_artifacts():
    models_dir = models_dir = ROOT / "models"
    rf = joblib.load(models_dir / "rf_model.pkl")
    rfe = joblib.load(models_dir / "rfe.pkl")
    feature_list = json.loads((models_dir / "feature_list.json").read_text())
    return rf, rfe, feature_list

@st.cache_data(ttl=24*3600)
def load_stock_yf(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date,
                       progress=False, auto_adjust=False)

def load_local_snapshot(symbol):
    data_dir = ROOT / "data"
    candidates = [
        data_dir / f"{symbol}.csv",
        data_dir / f"{symbol}.parquet",
        data_dir / f"{symbol.lower()}.csv",
        data_dir / f"{symbol.lower()}.parquet",
    ]

    for fp in candidates:
        if not fp.exists():
            continue
        if fp.suffix == ".csv":
            df = pd.read_csv(fp)
        else:
            df = pd.read_parquet(fp)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.set_index(df.columns[0])

        df = df[~df.index.isna()].sort_index()
        return df
    return None

# ==============================
# Small metrics
# ==============================
def pi_coverage(y_true, lo, hi):
    y = pd.Series(y_true).reset_index(drop=True)
    lo = pd.Series(lo).reset_index(drop=True)
    hi = pd.Series(hi).reset_index(drop=True)
    return float(((y >= lo) & (y <= hi)).mean())

def pinball_loss(y_true, y_pred, q=0.5):
    y = np.asarray(y_true)
    f = np.asarray(y_pred)
    diff = y - f
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Ticker", ["AMZN"], index=0)
    viz_months = st.selectbox(
        "Visualization range (display only)",
        [12, 24, 36],
        index=1,
        help="Only the displayed window — models are pre-trained."
    )
    want_refresh = st.checkbox(
        "If local snapshot is missing, allow Yahoo Finance fetch",
        value=False
    )

# Remove sidebar diagnostics switch
show_diag = False    # Diagnostics only via 'Forecast → Advanced diagnostics'

# ==============================
# Visualization Window
# ==============================
today = datetime.today()
start_date = datetime(today.year, today.month, today.day) - pd.DateOffset(months=viz_months)
end_date = today

# ==============================
# Title
# ==============================
st.title("📈 Probability Distribution Prediction of Stock Prices — AMZN")
st.caption("Inference-only demo: pre-trained Random Forest + RFE + GMM distribution. No live training.")

# ==============================
# Load Data
# ==============================
df = load_local_snapshot(symbol)

if df is None:
    if want_refresh:
        with st.spinner("Fetching from Yahoo Finance…"):
            df = load_stock_yf(symbol, start_date, end_date)
    else:
        st.error("No local data found. Place `data/AMZN.csv` or enable refresh.")
        st.stop()

if df.empty:
    st.error("Data empty. YFinance might be rate-limited.")
    st.stop()

# Flatten columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Ensure numeric
num_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for c in [c for c in num_cols if c in df.columns]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

# Clip to window
df = df.loc[df.index >= (today - pd.DateOffset(months=viz_months))].copy()

# Calendar features
df["day"] = df.index.day
df["month"] = df.index.month
df["quarter"] = df.index.quarter
df["year"] = df.index.year

# Technical indicators
df = calculate_technical_indicators(df)
df["Date"] = df.index

# Train/Test split
if len(df) < 14:
    st.warning("Not enough rows after indicators.")
    st.stop()

test_df = df.tail(7).copy()
train_df = df.iloc[:-7].copy()

# ==============================
# Load Artifacts
# ==============================
rf_model, rfe, feature_list = load_artifacts()

missing = [f for f in feature_list if f not in df.columns]
if missing:
    st.error(f"Missing required features: {missing}")
    st.stop()

X_test = test_df[feature_list].values
y_test = test_df["Close"].values
X_test_sel = rfe.transform(X_test)

# ==============================
# Run GMM
# ==============================
(predicted_prices,
 predicted_samples,
 interval_probs,
 rel_mode_probs,
 interval_80_lower,
 interval_80_upper,
 heat_dates,
 heat_prices,
 prob_matrix,
 display_top_prices,
 display_actuals,
 gmm_avg_pred_prices,
 heatmap_predicted_probs
) = run_gmm_model(
    rf_model,
    X_test_sel,
    y_test,
    test_df,
    max_components=3,
    delta=1.0,
    pred_interval_level=0.80,
    show_plots=show_diag
)

# Ensure numeric
y_test = np.asarray(y_test, float)
predicted_prices = np.asarray(predicted_prices, float)
gmm_avg_pred_prices = np.asarray(gmm_avg_pred_prices, float)

# ==============================
# KPIs
# ==============================
rmse_mode = float(np.sqrt(np.mean((y_test - predicted_prices)**2)))
mae_mode  = float(np.mean(np.abs(y_test - predicted_prices)))
rmse_avg  = float(np.sqrt(np.mean((y_test - gmm_avg_pred_prices)**2)))
mae_avg   = float(np.mean(np.abs(y_test - gmm_avg_pred_prices)))

coverage80 = pi_coverage(y_test, interval_80_lower, interval_80_upper)
pinball50  = pinball_loss(y_test, gmm_avg_pred_prices)

# ==============================
# Tabs
# ==============================
tab_forecast, tab_cal = st.tabs(["🔮 Forecast", "🧪 Calibration"])

# ---------- Forecast Tab ----------
with tab_forecast:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE (mode)", f"{rmse_mode:,.2f}")
    col2.metric("MAE (mode)", f"{mae_mode:,.2f}")
    col3.metric("RMSE (avg)", f"{rmse_avg:,.2f}")
    col4.metric("MAE (avg)", f"{mae_avg:,.2f}")

    st.subheader("Last 7 Days — Interval Summary")

    summary_df = pd.DataFrame({
        "Date": test_df['Date'].dt.strftime('%Y-%m-%d'),
        "Predicted Price": predicted_prices,
        "Probability in ±$1": interval_probs,
        "Relative Probability": rel_mode_probs,
        "80% Lower": interval_80_lower,
        "80% Upper": interval_80_upper
    })

    st.dataframe(summary_df.style.format({
        "Predicted Price": "{:.2f}",
        "Probability in ±$1": "{:.2%}",
        "Relative Probability": "{:.2%}",
        "80% Lower": "{:.2f}",
        "80% Upper": "{:.2f}",
    }), use_container_width=True)

    st.subheader("Prices")
    plot_results(train_df, test_df, y_test, gmm_avg_pred_prices, predicted_prices)
    plot_results_sub(test_df, y_test, gmm_avg_pred_prices, predicted_prices)

    with st.expander("Advanced diagnostics (slow)"):
        st.caption("Optional plots and heatmaps. Disabled by default to keep the app fast.")
        colA, colB = st.columns(2)
        show_dist = colA.checkbox("Show per-day BIC & distribution plots", value=False)
        show_heatmap = colB.checkbox("Show probability heatmap (last 7 days)", value=True)

        # 1) Re-run GMM with plotting only if user requests
        if show_dist:
            try:
                _ = run_gmm_model(
                    rf_model,
                    X_test_sel,
                    y_test,
                    test_df,
                    max_components=3,
                    delta=1.0,
                    pred_interval_level=0.80,
                    show_plots=True
                )
                st.success("Per-day BIC and distribution plots rendered above.")
            except Exception as e:
                st.error(f"Could not render distribution plots: {e}")

        # 2) Heatmap from returned arrays (no re-fit needed)
        if show_heatmap:
            try:
                # Use the values already returned earlier on the Forecast tab call
                # If your variables are named differently, adjust here.
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    x=heat_dates, y=heat_prices, z=prob_matrix,
                    colorscale='YlGnBu',
                    colorbar=dict(title='Probability'),
                    zmin=0, zmax=prob_matrix.max()
                ))
                fig.add_trace(go.Scatter(
                    x=heat_dates, y=display_top_prices,
                    mode='markers', marker=dict(symbol='x', size=16, color='red'),
                    name='GMM Relative Probability'
                ))
                fig.add_trace(go.Scatter(
                    x=heat_dates, y=display_actuals,
                    mode='markers', marker=dict(symbol='circle', size=10, color='green'),
                    name='Actual Values'
                ))
                fig.update_layout(
                    title="Probability Distribution Heatmap (last 7 days)",
                    xaxis=dict(title="Date", tickangle=-45),
                    yaxis=dict(title="Price"),
                    height=520, margin=dict(l=60, r=20, t=60, b=60),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render heatmap: {e}")


# ---------- Calibration Tab ----------
with tab_cal:
    st.subheader("Headline Calibration")

    c1, c2 = st.columns(2)
    c1.metric("80% PI Coverage", f"{coverage80:.0%}")
    c2.metric("Pinball Loss (q=0.5)", f"{pinball50:,.3f}")

    st.markdown("""
    **Interpretation**
    - **PI Coverage (80%)**: fraction of actual closes inside the 80% interval.
    - **Pinball Loss (q=0.5)**: median quantile loss.
    """)

    st.subheader("Residuals (Actual − Predicted)")
    fig, ax = plt.subplots()
    ax.plot(test_df["Date"], y_test - gmm_avg_pred_prices, marker='o')
    ax.axhline(0, linestyle='--', color='gray')
    st.pyplot(fig)

st.caption("This dashboard is for educational purposes only and not investment advice.")