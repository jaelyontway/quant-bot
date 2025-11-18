"""
Margin prediction for buy/sell thresholds.

This module predicts upper and lower price margins based on news sentiment
to determine optimal buy/sell points for a given trading day.

Can use either:
1. Rule-based prediction (default, no training needed)
2. ML-based prediction (if trained model is available)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to load trained ML model if available
_TRAINED_MODEL = None
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "margin_predictor.pkl"

def _load_trained_model():
    """Load trained ML model if available."""
    global _TRAINED_MODEL
    if _TRAINED_MODEL is None and _MODEL_PATH.exists():
        try:
            from .train_margins import MarginPredictor
            _TRAINED_MODEL = MarginPredictor.load_model(str(_MODEL_PATH))
            logger.info(f"✅ Loaded trained ML model from {_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load trained model: {e}")
    return _TRAINED_MODEL


def predict_margins(
    news_articles: List[str],
    current_price: float,
    date: datetime,
    sentiment_scores: Optional[List[float]] = None,
    historical_volatility: Optional[float] = None,
    method: str = 'auto',
    use_ml: bool = True
) -> Dict:
    """
    Predict upper and lower price margins for buy/sell decisions.
    
    Given news from a specific day and the current stock price, this function
    predicts two margin lines:
    - Upper margin: Price level to SELL at (take profit)
    - Lower margin: Price level to BUY at (buy the dip) or stop-loss
    
    Args:
        news_articles: List of news article texts from the target date
        current_price: Current stock price on the target date
        date: The date we're making predictions for
        sentiment_scores: Optional pre-computed sentiment scores (if None, will compute)
        historical_volatility: Optional historical volatility (default: 2% daily)
        method: Prediction method ('sentiment_based', 'llm', or 'hybrid')
        
    Returns:
        Dictionary with:
        - upper_margin: float - Price level to sell at
        - lower_margin: float - Price level to buy at
        - confidence: float - Confidence score (0-1)
        - reasoning: str - Explanation of the prediction
        - sentiment_aggregate: float - Overall sentiment score
        - expected_move_pct: float - Expected price movement percentage
        
    Example:
        >>> news = ["NVIDIA announces new chip", "Strong earnings beat"]
        >>> result = predict_margins(news, current_price=140.50, date=datetime.now())
        >>> print(f"Upper: ${result['upper_margin']:.2f}, Lower: ${result['lower_margin']:.2f}")
        Upper: $145.20, Lower: $137.80
    """
    logger.info(f"Predicting margins for {date.date()} at price ${current_price:.2f}")

    # Compute sentiment if not provided
    if sentiment_scores is None:
        from .sentiment import local_vader_sentiment
        sentiment_scores = []
        for article in news_articles:
            sent = local_vader_sentiment(article)
            sentiment_scores.append(sent['sentiment_score'])

    # Aggregate sentiment
    if sentiment_scores:
        sentiment_aggregate = np.mean(sentiment_scores)
        sentiment_std = np.std(sentiment_scores)
    else:
        sentiment_aggregate = 0.0
        sentiment_std = 0.0

    # Default volatility if not provided (2% daily move)
    if historical_volatility is None:
        historical_volatility = 0.02

    # Try to use trained ML model if available and requested
    if use_ml:
        trained_model = _load_trained_model()
        if trained_model is not None:
            logger.info("Using trained ML model for prediction")
            try:
                ml_result = trained_model.predict(
                    sentiment_score=sentiment_aggregate,
                    current_price=current_price,
                    date=date,
                    news_count=len(news_articles),
                    volatility=historical_volatility
                )
                # Add additional fields
                ml_result['sentiment_aggregate'] = sentiment_aggregate
                ml_result['date'] = date
                ml_result['reasoning'] = (
                    f"ML model prediction (trained on {ml_result['n_training_samples']} samples). "
                    f"Sentiment: {sentiment_aggregate:+.2f}. "
                    f"Expected upper move: {ml_result['upper_pct']:+.2f}%, "
                    f"lower move: {ml_result['lower_pct']:+.2f}%. "
                    f"Model confidence (R²): {ml_result['confidence']:.2f}"
                )
                ml_result['expected_move_pct'] = ml_result['upper_pct']
                ml_result['method'] = 'ml_trained'
                return ml_result
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}, falling back to rule-based")

    # Fall back to rule-based prediction
    logger.info("Using rule-based prediction")
    
    # Calculate expected price movement based on sentiment
    # Sentiment ranges from -1 to +1
    # We scale this to expected percentage move
    base_move_pct = sentiment_aggregate * historical_volatility * 3  # Amplify by 3x
    
    # Add uncertainty based on sentiment disagreement
    uncertainty = sentiment_std * historical_volatility
    
    # Calculate margins
    if sentiment_aggregate > 0.1:  # Bullish
        # Expect price to go up
        upper_margin = current_price * (1 + abs(base_move_pct) + uncertainty)
        lower_margin = current_price * (1 - historical_volatility)  # Stop loss
        reasoning = (
            f"Bullish sentiment ({sentiment_aggregate:.2f}) suggests upward movement. "
            f"Upper margin set at +{abs(base_move_pct)*100:.1f}% for profit taking. "
            f"Lower margin set at -{historical_volatility*100:.1f}% as stop-loss."
        )
        
    elif sentiment_aggregate < -0.1:  # Bearish
        # Expect price to go down
        upper_margin = current_price * (1 + historical_volatility)  # Stop loss for shorts
        lower_margin = current_price * (1 + base_move_pct - uncertainty)  # Buy the dip
        reasoning = (
            f"Bearish sentiment ({sentiment_aggregate:.2f}) suggests downward movement. "
            f"Lower margin set at {base_move_pct*100:.1f}% for buying opportunity. "
            f"Upper margin set at +{historical_volatility*100:.1f}% as stop-loss."
        )
        
    else:  # Neutral
        # Symmetric margins around current price
        margin_pct = historical_volatility * 1.5
        upper_margin = current_price * (1 + margin_pct)
        lower_margin = current_price * (1 - margin_pct)
        reasoning = (
            f"Neutral sentiment ({sentiment_aggregate:.2f}) suggests range-bound trading. "
            f"Margins set at ±{margin_pct*100:.1f}% around current price."
        )
    
    # Calculate confidence based on sentiment agreement
    confidence = 1.0 - min(sentiment_std, 0.5) * 2  # Higher std = lower confidence
    
    result = {
        'upper_margin': float(upper_margin),
        'lower_margin': float(lower_margin),
        'confidence': float(confidence),
        'reasoning': reasoning,
        'sentiment_aggregate': float(sentiment_aggregate),
        'expected_move_pct': float(base_move_pct * 100),  # Convert to percentage
        'num_articles': len(news_articles),
        'date': date,
        'current_price': current_price
    }
    
    logger.info(f"Predicted margins: Upper=${upper_margin:.2f}, Lower=${lower_margin:.2f}")
    logger.info(f"Reasoning: {reasoning}")
    
    return result


def predict_margins_with_llm(
    news_articles: List[str],
    current_price: float,
    date: datetime,
    api_key: str
) -> Dict:
    """
    Use LLM (GPT-4) to predict margins with reasoning.
    
    This sends news articles to GPT-4 and asks it to predict
    upper/lower margins with detailed reasoning.
    
    Args:
        news_articles: List of news texts
        current_price: Current stock price
        date: Target date
        api_key: OpenAI API key
        
    Returns:
        Same format as predict_margins()
    """
    import json
    import openai
    
    logger.info("Using LLM for margin prediction")
    
    # Combine news articles
    combined_news = "\n\n".join([f"Article {i+1}: {article[:500]}" 
                                  for i, article in enumerate(news_articles[:5])])
    
    prompt = f"""You are a quantitative trading analyst. Given news articles from {date.date()} and the current stock price of ${current_price:.2f}, predict optimal trading margins.

News articles:
{combined_news}

Based on this news, predict:
1. UPPER MARGIN: Price level where you would SELL/take profit
2. LOWER MARGIN: Price level where you would BUY or set stop-loss

Respond with ONLY valid JSON:
{{
  "upper_margin": <float>,
  "lower_margin": <float>,
  "confidence": <float 0-1>,
  "reasoning": "<detailed explanation>",
  "sentiment": "<bullish/bearish/neutral>",
  "expected_move_pct": <float percentage>
}}

Consider:
- Sentiment of the news (positive/negative)
- Magnitude of the news (major announcement vs minor update)
- Historical volatility (assume ~2% daily for tech stocks)
- Risk/reward ratio
"""
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a quantitative trading analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        # Add metadata
        result['date'] = date
        result['current_price'] = current_price
        result['num_articles'] = len(news_articles)
        result['method'] = 'llm'
        
        return result
        
    except Exception as e:
        logger.error(f"LLM prediction failed: {e}")
        logger.info("Falling back to sentiment-based prediction")
        return predict_margins(news_articles, current_price, date)

