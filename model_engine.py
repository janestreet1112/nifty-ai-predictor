import numpy as np
import pandas as pd
import xgboost as xgb
from transformers import pipeline
import streamlit as st

class MultiFactorAIEngine:
    def __init__(self):
        # Model is cached across runs
        pass

    @st.cache_resource
    def load_sentiment_pipeline(_self):
        try:
            return pipeline("text-classification", model="Vansh180/FinBERT-India-v1")
        except Exception as e:
            print("FinBERT load fallback:", e)
            return None

    def analyze_news_sentiment(self, headlines):
        pipe = self.load_sentiment_pipeline()
        if not headlines or not pipe:
            return 0.0

        scores = []
        for text in headlines[:3]:
            result = pipe(text)[0]
            label = result['label'].lower()
            confidence = result['score']

            if 'positive' in label:
                scores.append(confidence)
            elif 'negative' in label:
                scores.append(-confidence)
            else:
                scores.append(0.0)

        return round(float(np.mean(scores)), 3) if scores else 0.0

    def prepare_features(self, df, sentiment_score):
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=5).std()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['lag_1'] = df['close'].shift(1)
        df['lag_2'] = df['close'].shift(2)
        df['sentiment'] = sentiment_score
        return df.dropna()

    def forecast_next_target(self, df_history, sentiment_score):
        featured_df = self.prepare_features(df_history, sentiment_score)
        
        if len(featured_df) < 15:
            return float(df_history['close'].iloc[-1])

        feature_cols = ['open', 'high', 'low', 'close', 'returns', 'volatility', 'lag_1', 'sentiment']
        X = featured_df[feature_cols]
        y = featured_df['close'].shift(-1).dropna()
        X = X.iloc[:-1]

        model = xgb.XGBRegressor(
            n_estimators=30,  # Lower estimators to save CPU
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X, y)

        latest_row = featured_df[feature_cols].iloc[[-1]]
        prediction = model.predict(latest_row)[0]

        return float(prediction)
