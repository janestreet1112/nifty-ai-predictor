import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from model_engine import MultiFactorAIEngine
from market_feed import get_live_nifty_candles

st.set_page_config(layout="wide", page_title="NIFTY ML TradingView Dashboard", page_icon="📈")

# Apply TradingView Dark Theme CSS Override
st.markdown("""
    <style>
    .stApp { background-color: #131722; color: #d1d4dc; }
    div[data-testid="metric-container"] {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 NIFTY 50 Live Predictor (TradingView Style)")

@st.cache_resource
def init_ai():
    return MultiFactorAIEngine()

ai_engine = init_ai()

# 1. Load Live Market Feed
api_key = st.secrets.get("SMARTAPI_API_KEY")
client_id = st.secrets.get("SMARTAPI_USERNAME")
pin = st.secrets.get("SMARTAPI_PASSWORD")
totp_secret = st.secrets.get("SMARTAPI_TOTP_KEY")

df_history = None
if api_key and client_id and pin and totp_secret:
    with st.spinner("Syncing Live Angel One Stream..."):
        df_history = get_live_nifty_candles(api_key, client_id, pin, totp_secret)

if df_history is None or df_history.empty:
    st.info("ℹ️ Running on simulated feed. Check SmartAPI secrets on Streamlit Cloud.")
    num_bars = 40
    base_price = 23870.0
    time_stamps = [datetime.now() - timedelta(minutes=15 * (num_bars - i)) for i in range(num_bars)]
    returns = np.random.normal(0.00005, 0.001, num_bars)
    price_series = base_price * np.exp(np.cumsum(returns))
    
    df_history = pd.DataFrame({
        'time': time_stamps,
        'open': price_series - 4,
        'high': price_series + 8,
        'low': price_series - 8,
        'close': price_series,
        'volume': np.random.randint(1000, 5000, num_bars)
    })

# 2. AI Forecast Logic
sample_headlines = [
    "Nifty consolidates near key resistance levels amid strong institutional buying.",
    "RBI monetary policy stance remains stable.",
    "Global market cues remain balanced."
]
sentiment_score = ai_engine.analyze_news_sentiment(sample_headlines)
ai_target = ai_engine.forecast_next_target(df_history, sentiment_score)

latest_price = df_history['close'].iloc[-1]
price_diff = ai_target - latest_price

# 3. Streamlit Metrics Header
col1, col2, col3 = st.columns(3)
col1.metric("FinBERT Sentiment", f"{sentiment_score:+.2f}", 
            delta="Bullish" if sentiment_score > 0 else "Bearish")
col2.metric("NIFTY Live Price", f"₹{latest_price:.2f}")
col3.metric("XGBoost 15M Target", f"₹{ai_target:.2f}", f"{price_diff:+.2f} pts")

st.divider()

# 4. TradingView Style Candlestick Chart Engine
fig = go.Figure()

# Candlestick Series (TradingView colors: Green #089981, Red #f23645)
fig.add_trace(go.Candlestick(
    x=df_history['time'].dt.strftime('%m-%d %H:%M'),
    open=df_history['open'],
    high=df_history['high'],
    low=df_history['low'],
    close=df_history['close'],
    name='NIFTY 50',
    increasing_line_color='#089981', 
    decreasing_line_color='#f23645',
    increasing_fillcolor='#089981',
    decreasing_fillcolor='#f23645'
))

# Projected Target Horizontal Ray
fig.add_hline(
    y=ai_target, 
    line_dash="dash", 
    line_color="#2962ff" if price_diff >= 0 else "#f23645",
    annotation_text=f"  AI Target: ₹{ai_target:.2f}", 
    annotation_position="top right",
    annotation_font_color="white",
    annotation_font_size=12
)

# Dark TradingView Styling Layout
fig.update_layout(
    title="NIFTY 50 Intraday Candlesticks (15-Min Bar)",
    template="plotly_dark",
    paper_bgcolor="#131722",
    plot_bgcolor="#131722",
    height=520,
    xaxis_rangeslider_visible=False, # Removes ugly bottom slider
    xaxis=dict(
        showgrid=True, 
        gridcolor="#2a2e39", 
        color="#b2b5be",
        type='category' # Removes weekend/non-trading time gaps!
    ),
    yaxis=dict(
        showgrid=True, 
        gridcolor="#2a2e39", 
        color="#b2b5be",
        side="right" # Moves price scale to right (TradingView standard)
    )
)

st.plotly_chart(fig, use_container_width=True)

time.sleep(30)
st.rerun()
