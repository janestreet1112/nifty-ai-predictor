import numpy as np
import pandas as pd
import xgboost as xgb
from transformers import pipeline

class MultiFactorAIEngine:
    def __init__(self):
        # Load pre-trained FinBERT model for financial domain sentiment classification
        try:
            self.sentiment_pipe = pipeline("text-classification", model="Vansh180/FinBERT-India-v1")
        except Exception as e:
            print("FinBERT load fallback:", e)
            self.sentiment_pipe = None

    def analyze_news_sentiment(self, headlines):
        """Processes headlines through FinBERT to yield a normalized sentiment score [-1 to 1]"""
        if not headlines or not self.sentiment_pipe:
            return 0.0

        scores = []
        for text in headlines[:5]:  # Analyze top 5 recent headlines
            result = self.sentiment_pipe(text)[0]
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
        """Engineers technical indicators & lag features for XGBoost"""
        df = df.copy()
        
        # Calculate technical indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=5).std()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        
        # Lagged features
        df['lag_1'] = df['close'].shift(1)
        df['lag_2'] = df['close'].shift(2)
        
        # Append dynamic NLP sentiment feature
        df['sentiment'] = sentiment_score
        
        return df.dropna()

    def forecast_next_target(self, df_history, sentiment_score):
        """Trains a lightweight XGBoost model on historical window & predicts next bar target"""
        featured_df = self.prepare_features(df_history, sentiment_score)
        
        if len(featured_df) < 15:
            # Fallback if historical dataset is too small
            return float(df_history['close'].iloc[-1])

        # Define features (X) and target variable (y)
        feature_cols = ['open', 'high', 'low', 'close', 'returns', 'volatility', 'lag_1', 'sentiment']
        X = featured_df[feature_cols]
        y = featured_df['close'].shift(-1).dropna()
        X = X.iloc[:-1]  # Align dimensions with y

        # Fit XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X, y)

        # Predict next step target based on latest market row
        latest_row = featured_df[feature_cols].iloc[[-1]]
        prediction = model.predict(latest_row)[0]

        return float(prediction)
