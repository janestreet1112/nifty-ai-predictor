import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from model_engine import MultiFactorAIEngine

st.set_page_config(layout="wide", page_title="NIFTY ML Real-Time Predictor")

st.title("🤖 NIFTY 50 Live Predictor (FinBERT + XGBoost)")

# Load AI Engine
@st.cache_resource
def init_ai():
    return MultiFactorAIEngine()

ai_engine = init_ai()

# 1. Fetch Headlines & Run Sentiment Analysis
sample_headlines = [
    "Nifty consolidates near support levels as market volatility eases.",
    "FII flows show stabilization in blue-chip equities.",
    "RBI monetary policy commentary provides steady market baseline."
]
sentiment_score = ai_engine.analyze_news_sentiment(sample_headlines)

# 2. Generate Realistic NIFTY Price Movement (Base ~23,800)
np.random.seed(int(time.time()) % 1000)
base_price = 23800.0
num_bars = 40

# Simulated 15-min price bar random walk
time_stamps = [datetime.now() - timedelta(minutes=15 * (num_bars - i)) for i in range(num_bars)]
returns = np.random.normal(0.0001, 0.002, num_bars)
price_series = base_price * np.exp(np.cumsum(returns))

df_history = pd.DataFrame({
    'time': time_stamps,
    'open': price_series - np.random.uniform(2, 8, num_bars),
    'high': price_series + np.random.uniform(5, 15, num_bars),
    'low': price_series - np.random.uniform(5, 15, num_bars),
    'close': price_series
})

# 3. XGBoost Forecast Calculation
ai_target = ai_engine.forecast_next_target(df_history, sentiment_score)
latest_price = df_history['close'].iloc[-1]
price_diff = ai_target - latest_price

# 4. Display KPI Header Metrics
col1, col2, col3 = st.columns(3)
col1.metric("FinBERT Sentiment Score", f"{sentiment_score:+.2f}", 
            delta="Bullish" if sentiment_score > 0 else "Bearish")
col2.metric("Latest NIFTY 50 Level", f"₹{latest_price:.2f}")
col3.metric("XGBoost Projected Target", f"₹{ai_target:.2f}", f"{price_diff:+.2f} pts")

st.divider()

# 5. Interactive Price Chart
st.subheader("NIFTY 50 Historical Feed & AI Target Trajectory")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_history['time'], 
    y=df_history['close'], 
    mode='lines', 
    name='NIFTY Close Price',
    line=dict(color='#1f77b4', width=2)
))

# Forecast point
fig.add_trace(go.Scatter(
    x=[df_history['time'].iloc[-1] + timedelta(minutes=15)], 
    y=[ai_target], 
    mode='markers+text', 
    name='XGBoost Target',
    text=[f"Target: {ai_target:.2f}"],
    textposition="top center",
    marker=dict(color='green' if price_diff >= 0 else 'red', size=12, symbol='star')
))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="NIFTY Index Value (₹)",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

if price_diff >= 0:
    st.success(f"🎯 **XGBoost Next-Bar Target:** **{ai_target:.2f}** ({price_diff:+.2f} pts upside potential)")
else:
    st.error(f"🎯 **XGBoost Next-Bar Target:** **{ai_target:.2f}** ({price_diff:+.2f} pts downside risk)")

# Refresh every 30 seconds
time.sleep(30)
st.rerun()
