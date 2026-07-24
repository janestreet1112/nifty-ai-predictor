import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from model_engine import MultiFactorAIEngine
from market_feed import get_live_nifty_candles

st.set_page_config(layout="wide", page_title="NIFTY ML Real-Time Predictor")
st.title("🤖 NIFTY 50 Live Predictor (FinBERT + XGBoost)")

@st.cache_resource
def init_ai():
    return MultiFactorAIEngine()

ai_engine = init_ai()

# 1. Fetch Live Market Candles (SmartAPI or Simulated Fallback)
# 1. Fetch Live Market Candles (SmartAPI or Simulated Fallback)
df_history = None

# Support both naming formats in Streamlit Secrets
api_key = st.secrets.get("SMARTAPI_API_KEY") or st.secrets.get("SMARTAPI_KEY")
client_id = st.secrets.get("SMARTAPI_USERNAME") or st.secrets.get("CLIENT_ID")
pin = st.secrets.get("SMARTAPI_PASSWORD") or st.secrets.get("PIN")
totp_secret = st.secrets.get("SMARTAPI_TOTP_KEY") or st.secrets.get("TOTP_SECRET")

if api_key and client_id and pin and totp_secret:
    with st.spinner("Connecting to Angel One Live Market Stream..."):
        df_history = get_live_nifty_candles(api_key, client_id, pin, totp_secret)

# Fallback generator if credentials fail or are missing
if df_history is None or df_history.empty:
    st.info("ℹ️ Running on simulated feed. Verify SmartAPI secrets on Streamlit Cloud.")
    num_bars = 40
    base_price = 23630.0  # Set closer to current market level
    time_stamps = [datetime.now() - timedelta(minutes=15 * (num_bars - i)) for i in range(num_bars)]
    returns = np.random.normal(0.00005, 0.001, num_bars)
    price_series = base_price * np.exp(np.cumsum(returns))
    
    df_history = pd.DataFrame({
        'time': time_stamps,
        'open': price_series - 4,
        'high': price_series + 8,
        'low': price_series - 8,
        'close': price_series
    })

# 2. Run NLP & ML Pipeline
sample_headlines = [
    "Nifty consolidates near key support levels amid global market cues.",
    "Institutional flows indicate selective buying in large-cap stocks.",
    "Domestic macro data remains supportive for market baseline."
]
sentiment_score = ai_engine.analyze_news_sentiment(sample_headlines)
ai_target = ai_engine.forecast_next_target(df_history, sentiment_score)

latest_price = df_history['close'].iloc[-1]
price_diff = ai_target - latest_price

# 3. Render Dashboard Metrics & Plots
col1, col2, col3 = st.columns(3)
col1.metric("FinBERT Sentiment Score", f"{sentiment_score:+.2f}", 
            delta="Bullish" if sentiment_score > 0 else "Bearish")
col2.metric("Current NIFTY Level", f"₹{latest_price:.2f}")
col3.metric("XGBoost 15-Min Forecast", f"₹{ai_target:.2f}", f"{price_diff:+.2f} pts")

st.divider()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_history['time'], y=df_history['close'], mode='lines', name='NIFTY Index'))
fig.add_trace(go.Scatter(
    x=[df_history['time'].iloc[-1] + timedelta(minutes=15)], 
    y=[ai_target], 
    mode='markers+text', 
    name='XGBoost Target',
    text=[f"Target: {ai_target:.2f}"],
    textposition="top center",
    marker=dict(color='green' if price_diff >= 0 else 'red', size=12, symbol='star')
))

fig.update_layout(xaxis_title="Time", yaxis_title="NIFTY Index (₹)", template="plotly_white", height=450)
st.plotly_chart(fig, use_container_width=True)
