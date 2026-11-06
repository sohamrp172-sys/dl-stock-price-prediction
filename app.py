"""AI-Powered Stock Price Predictor — Streamlit dashboard.

A thin presentation layer over the `src` package. All data, model, and
indicator logic lives in reusable modules; this file only orchestrates the UI.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import __version__
from src.config import (
    DEFAULT_FUTURE_DAYS,
    DEFAULT_START_DATE,
    MAX_FUTURE_DAYS,
    MIN_FUTURE_DAYS,
)
from src.data import InvalidTickerError, date_range_for_period, load_stock_data
from src.indicators import adx_strength, bollinger_position, enrich
from src.model import evaluate, forecast_confidence, forecast_future, load_prediction_model

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Stock Price Predictor | AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 3rem; font-weight: 800;
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-align: center; padding: 0.5rem 0 0 0;
        }
        .sub-header { text-align:center; color:#64748b; font-size:1.1rem; margin-bottom:1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">📈 AI-Powered Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Deep LSTM neural network for stock market analysis & forecasting</p>',
    unsafe_allow_html=True,
)

TEMPLATE = "plotly_dark"


# --------------------------------------------------------------------------- #
# Cached wrappers (Streamlit-specific)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def cached_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    return load_stock_data(ticker, start, end)


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_prediction_model()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📊 Stock Selection")
    stock_input = st.text_input(
        "Enter Stock Ticker", value="GOOG",
        help="e.g. AAPL, MSFT, TSLA, BRK.B",
    )

    st.subheader("📅 Time Period")
    period_option = st.selectbox(
        "Quick Select",
        ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "Max"],
        index=6,
    )
    if period_option == "Custom":
        start_date = st.date_input("Start Date", value=pd.to_datetime(DEFAULT_START_DATE)).strftime("%Y-%m-%d")
        end_date = st.date_input("End Date", value=pd.to_datetime("today")).strftime("%Y-%m-%d")
    else:
        start_date, end_date = date_range_for_period(period_option)

    st.subheader("🔮 Prediction Settings")
    future_days = st.slider("Future Prediction Days", MIN_FUTURE_DAYS, MAX_FUTURE_DAYS, DEFAULT_FUTURE_DAYS)

    st.subheader("🤖 Model Information")
    st.info(
        "**LSTM Neural Network**\n\n"
        "- 4 stacked LSTM layers\n"
        "- Dropout regularization\n"
        "- Adam optimizer\n"
        "- 100-day lookback window"
    )
    st.markdown("---")
    st.caption(f"Version {__version__}")


# --------------------------------------------------------------------------- #
# Load data + model with graceful error handling
# --------------------------------------------------------------------------- #
try:
    with st.spinner(f"🔄 Loading {stock_input.upper()} data..."):
        data = cached_data(stock_input, start_date, end_date)
except InvalidTickerError as exc:
    st.error(f"❌ {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"❌ Unexpected error loading data: {exc}")
    st.stop()

try:
    with st.spinner("🧠 Loading model..."):
        model = cached_model()
except FileNotFoundError as exc:
    st.error(f"❌ {exc}")
    st.stop()

stock = stock_input.upper()
data = enrich(data)

current_price = float(data["Close"].iloc[-1])
prev_price = float(data["Close"].iloc[-2])
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100


# --------------------------------------------------------------------------- #
# Top metrics
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{stock} Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
c2.metric("Day High", f"${float(data['High'].iloc[-1]):.2f}")
c3.metric("Day Low", f"${float(data['Low'].iloc[-1]):.2f}")
c4.metric("Volume", f"{float(data['Volume'].iloc[-1]):,.0f}")
volatility = float(data["Close"].pct_change().std() * 100)
c5.metric("Volatility", f"{volatility:.2f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔮 Predictions", "📈 Technical Analysis", "🤖 Model Performance", "ℹ️ About"]
)

# --------------------------------------------------------------------------- #
# TAB 1 — Overview
# --------------------------------------------------------------------------- #
with tab1:
    st.header(f"{stock} Stock Analysis Overview")

    st.subheader("📊 Interactive Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="OHLC",
    )])
    fig.update_layout(title=f"{stock} Stock Price", yaxis_title="Price (USD)",
                      xaxis_title="Date", height=500, template=TEMPLATE, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Trading Volume")
    colors = np.where(data["Close"] < data["Open"], "#ef4444", "#10b981")
    vol = go.Figure(go.Bar(x=data.index, y=data["Volume"], marker_color=colors, name="Volume"))
    vol.update_layout(title="Trading Volume", yaxis_title="Volume", xaxis_title="Date",
                      height=300, template=TEMPLATE)
    st.plotly_chart(vol, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("📊 Dataset Statistics")
        stats = pd.DataFrame({
            "Metric": ["Total Days", "Start Date", "End Date", "Avg Price", "Max Price", "Min Price"],
            "Value": [
                len(data), data.index[0].strftime("%Y-%m-%d"), data.index[-1].strftime("%Y-%m-%d"),
                f"${data['Close'].mean():.2f}", f"${data['Close'].max():.2f}", f"${data['Close'].min():.2f}",
            ],
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)
    with right:
        st.subheader("📈 Recent Data")
        st.dataframe(data[["Open", "High", "Low", "Close", "Volume"]].tail(10), use_container_width=True)


# --------------------------------------------------------------------------- #
# TAB 2 — Predictions
# --------------------------------------------------------------------------- #
with tab2:
    st.header("🔮 Stock Price Predictions")
    try:
        with st.spinner("🧠 Back-testing model and generating forecast..."):
            result = evaluate(data, model=model)
            future_predictions = forecast_future(data, future_days, model=model)
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=future_days)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Accuracy", f"{result.accuracy:.2f}%")
        m2.metric("📊 RMSE", f"${result.rmse:.2f}")
        m3.metric("📈 MAE", f"${result.mae:.2f}")
        m4.metric("📉 MAPE", f"{result.mape:.2f}%")

        st.subheader("📊 Historical Predictions vs Actual Prices")
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=result.test_dates, y=result.y_true, mode="lines",
                                      name="Actual", line=dict(color="#10b981", width=2)))
        pred_fig.add_trace(go.Scatter(x=result.test_dates, y=result.y_pred, mode="lines",
                                      name="Predicted", line=dict(color="#ef4444", width=2, dash="dash")))
        pred_fig.update_layout(title="Prediction vs Actual", xaxis_title="Date", yaxis_title="Price (USD)",
                               height=500, template=TEMPLATE, hovermode="x unified")
        st.plotly_chart(pred_fig, use_container_width=True)

        st.subheader(f"🔮 Future Price Predictions ({future_days} Days)")
        hist_vol = data["Close"].pct_change().std()
        confidence = forecast_confidence(future_predictions, hist_vol)
        st.info(f"**Prediction Confidence:** {confidence:.1f}% | Based on historical volatility & model accuracy")

        window = min(90, len(data))
        uncertainty = np.array(future_predictions) * (hist_vol * 2)
        ffig = go.Figure()
        ffig.add_trace(go.Scatter(x=data.index[-window:], y=data["Close"][-window:], mode="lines",
                                  name="Historical", line=dict(color="#3b82f6", width=2)))
        ffig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode="lines+markers",
                                  name="Forecast", line=dict(color="#f59e0b", width=3, dash="dot")))
        ffig.add_trace(go.Scatter(x=future_dates, y=future_predictions + uncertainty, mode="lines",
                                  line=dict(color="rgba(245,158,11,0)"), showlegend=False, hoverinfo="skip"))
        ffig.add_trace(go.Scatter(x=future_dates, y=future_predictions - uncertainty, fill="tonexty",
                                  mode="lines", line=dict(color="rgba(245,158,11,0)"),
                                  name="Confidence Interval", fillcolor="rgba(245,158,11,0.2)"))
        ffig.update_layout(title=f"{stock} Forecast — Next {future_days} Days", xaxis_title="Date",
                           yaxis_title="Price (USD)", height=500, template=TEMPLATE, hovermode="x unified")
        st.plotly_chart(ffig, use_container_width=True)

        dcol, scol = st.columns(2)
        with dcol:
            pred_df = pd.DataFrame({
                "Date": future_dates, "Predicted_Price": future_predictions,
                "Upper_Bound": future_predictions + uncertainty,
                "Lower_Bound": future_predictions - uncertainty,
            })
            st.download_button(
                "📥 Download Future Predictions (CSV)", pred_df.to_csv(index=False),
                file_name=f"{stock}_predictions_{datetime.now():%Y%m%d}.csv", mime="text/csv",
            )
        with scol:
            st.markdown("### 📊 Prediction Summary")
            avg_future = float(np.mean(future_predictions))
            change_forecast = ((avg_future - current_price) / current_price) * 100
            st.write(f"**Average Predicted Price:** ${avg_future:.2f}")
            st.write(f"**Expected Change:** {change_forecast:+.2f}%")
            st.write(f"**Highest Prediction:** ${max(future_predictions):.2f}")
            st.write(f"**Lowest Prediction:** ${min(future_predictions):.2f}")
    except ValueError as exc:
        st.warning(f"⚠️ {exc}")


# --------------------------------------------------------------------------- #
# TAB 3 — Technical Analysis
# --------------------------------------------------------------------------- #
with tab3:
    st.header("📈 Technical Analysis")

    st.subheader("📊 Moving Averages Analysis")
    ma = go.Figure()
    ma.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#3b82f6", width=2)))
    for col, color in [("MA_50", "#10b981"), ("MA_100", "#f59e0b"), ("MA_200", "#ef4444")]:
        ma.add_trace(go.Scatter(x=data.index, y=data[col], name=col.replace("_", "-"),
                                line=dict(color=color, width=1.5)))
    ma.update_layout(title="Price with Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)",
                     height=500, template=TEMPLATE, hovermode="x unified")
    st.plotly_chart(ma, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("📊 RSI (Relative Strength Index)")
        rsi = go.Figure()
        rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="#8b5cf6", width=2)))
        rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        rsi.update_layout(yaxis_title="RSI", xaxis_title="Date", height=300, template=TEMPLATE, showlegend=False)
        st.plotly_chart(rsi, use_container_width=True)
        cur_rsi = float(data["RSI"].iloc[-1])
        if cur_rsi > 70:
            st.warning(f"⚠️ RSI at {cur_rsi:.2f} — may be **overbought**")
        elif cur_rsi < 30:
            st.success(f"✅ RSI at {cur_rsi:.2f} — may be **oversold**")
        else:
            st.info(f"ℹ️ RSI at {cur_rsi:.2f} — **neutral** zone")
    with right:
        st.subheader("📊 MACD")
        macd = go.Figure()
        macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#3b82f6", width=2)))
        macd.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal", line=dict(color="#ef4444", width=2)))
        macd.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="Histogram", marker_color="gray", opacity=0.3))
        macd.update_layout(yaxis_title="MACD", xaxis_title="Date", height=300, template=TEMPLATE)
        st.plotly_chart(macd, use_container_width=True)
        if data["MACD"].iloc[-1] > data["MACD_Signal"].iloc[-1]:
            st.success("✅ MACD indicates a **bullish** trend")
        else:
            st.warning("⚠️ MACD indicates a **bearish** trend")

    st.subheader("📊 Additional Technical Indicators")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Bollinger Band Position", f"{bollinger_position(data):.1f}%")
    i2.metric("ADX (Trend Strength)", f"{adx_strength(data):.2f}")
    avg_volume = data["Volume"].mean()
    i3.metric("Volume vs Average", f"{((data['Volume'].iloc[-1] - avg_volume) / avg_volume) * 100:+.1f}%")
    momentum = ((data["Close"].iloc[-1] - data["Close"].iloc[-20]) / data["Close"].iloc[-20]) * 100
    i4.metric("20-Day Momentum", f"{momentum:+.2f}%")


# --------------------------------------------------------------------------- #
# TAB 4 — Model Performance
# --------------------------------------------------------------------------- #
with tab4:
    st.header("🤖 Model Performance Analysis")
    try:
        errors = result.y_true - result.y_pred
        left, right = st.columns(2)
        with left:
            st.subheader("📊 Performance Metrics")
            mdf = pd.DataFrame({
                "Metric": ["RMSE", "MAE", "MAPE", "Accuracy", "Train Split", "Test Split"],
                "Value": [f"${result.rmse:.2f}", f"${result.mae:.2f}", f"{result.mape:.2f}%",
                          f"{result.accuracy:.2f}%", "80%", "20%"],
            })
            st.dataframe(mdf, use_container_width=True, hide_index=True)

            st.subheader("📊 Prediction Error Distribution")
            efig = go.Figure(go.Histogram(x=errors, nbinsx=30, marker_color="#8b5cf6"))
            efig.update_layout(title="Distribution of Prediction Errors", xaxis_title="Error (USD)",
                               yaxis_title="Frequency", height=300, template=TEMPLATE, showlegend=False)
            st.plotly_chart(efig, use_container_width=True)
        with right:
            st.subheader("🏗️ Model Architecture")
            st.code(
                "Model: LSTM Neural Network\n"
                "──────────────────────────\n"
                "LSTM(50)  + Dropout(0.2)\n"
                "LSTM(60)  + Dropout(0.3)\n"
                "LSTM(80)  + Dropout(0.4)\n"
                "LSTM(120) + Dropout(0.5)\n"
                "Dense(1)\n\n"
                "Optimizer: Adam | Loss: MSE | Lookback: 100 days",
                language="text",
            )
            st.subheader("📊 Actual vs Predicted")
            scatter = go.Figure()
            scatter.add_trace(go.Scatter(x=result.y_true, y=result.y_pred, mode="markers",
                                         marker=dict(size=7, color=np.abs(errors), colorscale="Viridis",
                                                     showscale=True, colorbar=dict(title="Error")), name="Predictions"))
            lo, hi = float(min(result.y_true.min(), result.y_pred.min())), float(max(result.y_true.max(), result.y_pred.max()))
            scatter.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect",
                                         line=dict(color="red", dash="dash", width=2)))
            scatter.update_layout(xaxis_title="Actual (USD)", yaxis_title="Predicted (USD)",
                                  height=350, template=TEMPLATE)
            st.plotly_chart(scatter, use_container_width=True)

        st.subheader("💡 Performance Insights")
        p1, p2, p3 = st.columns(3)
        avg_error_pct = (result.mae / result.y_true.mean()) * 100
        if avg_error_pct < 5:
            p1.success(f"✅ Excellent — avg error {avg_error_pct:.2f}%")
        elif avg_error_pct < 10:
            p1.info(f"ℹ️ Good — avg error {avg_error_pct:.2f}%")
        else:
            p1.warning(f"⚠️ Fair — avg error {avg_error_pct:.2f}%")
        actual_dir = np.diff(result.y_true) > 0
        pred_dir = np.diff(result.y_pred) > 0
        p2.metric("Directional Accuracy", f"{(actual_dir == pred_dir).mean() * 100:.1f}%")
        p3.metric("Maximum Error", f"${np.max(np.abs(errors)):.2f}")
    except NameError:
        st.info("Open the Predictions tab first to run the back-test.")

# --------------------------------------------------------------------------- #
# TAB 5 — About
# --------------------------------------------------------------------------- #
with tab5:
    st.header("ℹ️ About This Project")
    st.markdown(
        """
        ### 🎯 Overview
        An end-to-end deep learning application that forecasts stock prices with a stacked
        **LSTM** network, served through a decoupled **FastAPI** backend and this interactive
        **Streamlit** dashboard.

        ### 🧠 Pipeline
        1. **Data** — real-time OHLCV from Yahoo Finance (validated tickers)
        2. **Scaling** — `MinMaxScaler` fit on the training slice only (no leakage)
        3. **Sequencing** — 100-day rolling windows
        4. **Inference** — 4-layer LSTM with dropout regularization
        5. **Post-processing** — inverse-transform + confidence bands
        6. **Visualization** — interactive Plotly charts

        ### ⚠️ Disclaimer
        For **educational and portfolio purposes only**. Not financial advice.
        Past performance does not guarantee future results.
        """
    )

st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#64748b;'>"
    f"<p><strong>AI-Powered Stock Price Predictor</strong> v{__version__} | Streamlit + FastAPI + Keras</p>"
    f"<p>⚠️ <em>For educational purposes only. Not financial advice.</em></p></div>",
    unsafe_allow_html=True,
)
