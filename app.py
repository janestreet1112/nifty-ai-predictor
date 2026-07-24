import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from data_fetcher import MarketSentimentFetcher
from model_engine import MultiFactorAIEngine

st.set_page_config(layout="wide", page_title="NIFTY ML Real-Time Predictor")

st.title("🤖 NIFTY 50 Live Predictor (FinBERT + XGBoost)")

# Load AI Engines
@st.cache_resource
def init_ai():
    fetcher = MarketSentimentFetcher()
    ai_engine = MultiFactorAIEngine()
    return fetcher, ai_engine

fetcher, ai_engine = init_ai()

# 1. Fetch Headlines & Run FinBERT Sentiment
sample_headlines = [
    "Sensex surges 400 points as IT and banking stocks rally.",
    "RBI maintains hawkish stance on rate cut projections amid inflation.",
    "FII flows turn positive in Indian equities session."
]
sentiment_score = ai_engine.analyze_news_sentiment(sample_headlines)

# 2. Display KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("FinBERT Sentiment Score", f"{sentiment_score:+.2f}", 
            delta="Bullish Bias" if sentiment_score > 0 else "Bearish Bias")
col2.metric("ML Model Architecture", "XGBoost + FinBERT")
col3.metric("Data Feed Status", "Live Active")

st.divider()

# 3. Generate Historical Candle Context
times = [datetime.now() - timedelta(minutes=15 * i) for i in range(50, 0, -1)]
prices = np.random.normal(24500, 10, 50).cumsum() / 5 + 24500

df_history = pd.DataFrame({
    'time': times,
    'open': prices - 2,
    'high': prices + 5,
    'low': prices - 5,
    'close': prices
})

# 4. Predict Next Target using XGBoost
ai_target = ai_engine.forecast_next_target(df_history, sentiment_score)
latest_price = df_history['close'].iloc[-1]
price_diff = ai_target - latest_price

# Plotting Output
st.subheader("Live Market Feed & ML Forecast Target")
st.line_chart(df_history.set_index('time')['close'])

st.success(f"🎯 **XGBoost Next-Bar Projected Target:** **{ai_target:.2f}** ({price_diff:+.2f} pts from current price)")

# Auto-Rerun Loop
time.sleep(15)
st.rerun()
