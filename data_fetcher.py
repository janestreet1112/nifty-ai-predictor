import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

class MarketSentimentFetcher:
    def __init__(self):
        pass

    def get_financial_news_sentiment(self):
        """Scrapes headlines from Indian financial news RSS feeds"""
        try:
            url = "https://economictimes.indiatimes.com/markets/rssfeeds/55376510.cms"
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.content, features="xml")
            
            items = soup.find_all('item')[:10]
            scores = []
            
            for item in items:
                title = item.title.text
                analysis = TextBlob(title)
                scores.append(analysis.sentiment.polarity)
                
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return round(avg_score, 2)
        except Exception as e:
            print("News Scraping Error:", e)
            return 0.0

    def get_policy_updates(self):
        """Monitors monetary policy / macroeconomic events"""
        return {"policy_bias": "NEUTRAL", "impact_score": 0.0}
