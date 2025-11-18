"""
Sentiment analysis utilities.

This module provides sentiment extraction using:
1. Local VADER sentiment analyzer (default, no API required)
2. Optional OpenAI GPT API for advanced sentiment (requires API key)
"""

import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Initialize VADER analyzer (lazy-loaded)
_vader_analyzer: Optional[SentimentIntensityAnalyzer] = None


def get_event_sentiment(
    event,  # Event object from clustering module
    articles: pd.DataFrame,
    method: str = 'local',
    api_key: Optional[str] = None
) -> Dict:
    """
    Extract sentiment for a news event.

    Args:
        event: Event object with article_indices
        articles: DataFrame with news articles
        method: 'local' for VADER or 'api' for OpenAI (default: 'local')
        api_key: OpenAI API key (required if method='api')

    Returns:
        Dictionary with keys:
        - sentiment_score: float in range [-1, 1]
        - key_actors: list of mentioned entities (empty for VADER)
        - short_term_effect: 'bullish' | 'bearish' | 'neutral'
        - method_used: 'vader' | 'openai'

    Raises:
        ValueError: If method is invalid or API key missing
    """
    # Get articles for this event
    event_articles = articles.iloc[event.article_indices]

    # Combine texts
    combined_text = " ".join(
        event_articles['title'] + ". " + event_articles['text']
    )

    if method == 'local':
        return local_vader_sentiment(combined_text)
    elif method == 'api':
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("No API key provided, falling back to VADER")
            return local_vader_sentiment(combined_text)
        return llm_sentiment_api(combined_text, api_key)
    else:
        raise ValueError(f"Invalid sentiment method: {method}. Use 'local' or 'api'")


def local_vader_sentiment(text: str) -> Dict:
    """
    Compute sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER is a lexicon and rule-based sentiment analysis tool specifically
    attuned to sentiments expressed in social media and news.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with sentiment_score, key_actors, short_term_effect, method_used
    """
    global _vader_analyzer

    # Lazy-load VADER
    if _vader_analyzer is None:
        logger.info("Initializing VADER sentiment analyzer")
        _vader_analyzer = SentimentIntensityAnalyzer()

    # Get sentiment scores
    scores = _vader_analyzer.polarity_scores(text)

    # VADER returns: neg, neu, pos, compound
    # compound is normalized score in [-1, 1]
    sentiment_score = scores['compound']

    # Determine short-term effect
    if sentiment_score > 0.05:
        effect = 'bullish'
    elif sentiment_score < -0.05:
        effect = 'bearish'
    else:
        effect = 'neutral'

    logger.debug(f"VADER sentiment: {sentiment_score:.3f} ({effect})")

    return {
        'sentiment_score': sentiment_score,
        'key_actors': [],  # VADER doesn't extract entities
        'short_term_effect': effect,
        'method_used': 'vader'
    }


def llm_sentiment_api(text: str, api_key: str) -> Dict:
    """
    Compute sentiment using OpenAI GPT API.

    This function sends a structured prompt to GPT-4 requesting JSON output
    with sentiment analysis. Falls back to VADER if API call fails.

    Args:
        text: Input text to analyze
        api_key: OpenAI API key

    Returns:
        Dictionary with sentiment_score, key_actors, short_term_effect, method_used
    """
    try:
        import openai

        # Initialize client
        client = openai.OpenAI(api_key=api_key)

        # Construct prompt
        prompt = _build_sentiment_prompt(text)

        logger.info("Calling OpenAI API for sentiment analysis")

        # Call API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial news sentiment analyzer. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic output
            max_tokens=500
        )

        # Parse response
        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        # Validate and normalize
        sentiment_score = float(result.get('sentiment_score', 0))
        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]

        key_actors = result.get('key_actors', [])
        if not isinstance(key_actors, list):
            key_actors = []

        effect = result.get('short_term_effect', 'neutral')
        if effect not in ['bullish', 'bearish', 'neutral']:
            effect = 'neutral'

        logger.info(f"OpenAI sentiment: {sentiment_score:.3f} ({effect})")

        return {
            'sentiment_score': sentiment_score,
            'key_actors': key_actors,
            'short_term_effect': effect,
            'method_used': 'openai'
        }

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        logger.info("Falling back to VADER sentiment")
        return local_vader_sentiment(text)


def _build_sentiment_prompt(text: str, max_chars: int = 4000) -> str:
    """
    Build the sentiment analysis prompt for LLM.

    Args:
        text: Input text (will be truncated if too long)
        max_chars: Maximum characters to include (default: 4000)

    Returns:
        Formatted prompt string
    """
    # Truncate text if needed
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    prompt = f"""You are a financial news classifier. Given the combined texts and titles of multiple articles about the same event, produce JSON with the following structure:

{{
  "sentiment_score": <float in range [-1, 1], where -1 is very negative, 0 is neutral, 1 is very positive>,
  "summary": "<one-sentence summary of the event>",
  "key_actors": ["<company or person 1>", "<company or person 2>", ...],
  "short_term_effect": "<one of: bullish, bearish, neutral>"
}}

Consider the following for sentiment_score:
- Positive news (product launches, partnerships, earnings beats): 0.5 to 1.0
- Negative news (earnings misses, layoffs, scandals): -1.0 to -0.5
- Neutral or mixed news: -0.5 to 0.5

For short_term_effect, consider the likely 3-day stock price impact:
- bullish: likely to drive stock price up
- bearish: likely to drive stock price down
- neutral: unclear or minimal impact

Text to analyze:
{text}

Respond with ONLY the JSON object, no additional text."""

    return prompt

