"""Trading signals"""
from typing import Dict, Literal, Optional

Signal = Literal['BUY', 'SHORT', 'HOLD']

DEFAULT_THRESHOLDS = {'buy_sentiment': 0.6, 'short_sentiment': -0.6, 'min_coverage': 3}

def generate_signal(sentiment_score: float, coverage_count: int, thresholds: Optional[Dict[str, float]] = None) -> Signal:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    else:
        thresholds = {**DEFAULT_THRESHOLDS, **thresholds}
    
    if coverage_count < thresholds['min_coverage']:
        return 'HOLD'
    if sentiment_score > thresholds['buy_sentiment']:
        return 'BUY'
    elif sentiment_score < thresholds['short_sentiment']:
        return 'SHORT'
    return 'HOLD'

def calculate_margins(sentiment_score: float, signal: Signal, current_price: float, base_margin_pct: float = 0.02) -> Dict[str, float]:
    """
    Calculate upper and lower trading margins based on sentiment, signal, and current price.
    
    Args:
        sentiment_score: Sentiment score [-1, 1]
        signal: Trading signal
        current_price: Current stock price
        base_margin_pct: Base margin percentage (default: 2%)
    
    Returns:
        Dict with 'upper_margin' and 'lower_margin' as absolute prices
    """
    sentiment_strength = abs(sentiment_score)
    
    if signal == 'BUY':
        # Bullish: wider upper margin, tighter lower margin
        upper_pct = base_margin_pct * (1 + sentiment_strength)
        lower_pct = base_margin_pct * (1 - sentiment_strength * 0.5)
    elif signal == 'SHORT':
        # Bearish: tighter upper margin, wider lower margin
        upper_pct = base_margin_pct * (1 - sentiment_strength * 0.5)
        lower_pct = base_margin_pct * (1 + sentiment_strength)
    else:
        # HOLD: symmetric margins
        upper_pct = base_margin_pct
        lower_pct = base_margin_pct
    
    # Convert percentages to absolute prices
    upper_margin = current_price * (1 + upper_pct)
    lower_margin = current_price * (1 - lower_pct)
    
    return {
        'upper_margin': upper_margin,
        'lower_margin': lower_margin,
        'upper_margin_pct': upper_pct,
        'lower_margin_pct': lower_pct
    }

