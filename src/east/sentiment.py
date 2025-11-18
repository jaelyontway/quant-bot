"""Sentiment analysis"""
import pandas as pd
from typing import Dict, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader_analyzer: Optional[SentimentIntensityAnalyzer] = None

def get_event_sentiment(event, articles: pd.DataFrame, method: str = 'local') -> Dict:
    event_articles = articles.iloc[event.article_indices]
    combined_text = " ".join(event_articles['title'] + ". " + event_articles['text'])
    return local_vader_sentiment(combined_text)

def local_vader_sentiment(text: str) -> Dict:
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    scores = _vader_analyzer.polarity_scores(text)
    sentiment_score = scores['compound']
    effect = 'bullish' if sentiment_score > 0.05 else ('bearish' if sentiment_score < -0.05 else 'neutral')
    return {'sentiment_score': sentiment_score, 'short_term_effect': effect, 'method_used': 'vader'}

