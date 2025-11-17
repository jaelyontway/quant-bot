#!/usr/bin/env python3
"""
Integration module connecting data_fetching.py with margin_predictor.py

This module takes the news and price data fetched by data_fetching.py
and uses it to predict trading margins.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .margin_predictor import predict_margins

logger = logging.getLogger(__name__)


def predict_margins_from_fetched_data(
    news_csv: str | Path,
    prices_csv: str | Path,
    target_date: str | datetime = None
) -> Dict:
    """
    Predict trading margins using data from data_fetching.py output.
    
    This function:
    1. Loads news CSV from data_fetching.py
    2. Loads price CSV from data_fetching.py
    3. Combines news articles into text
    4. Gets the opening price for the target date
    5. Calls margin predictor
    6. Returns margins for trading simulation
    
    Args:
        news_csv: Path to news CSV (e.g., "NVDA_2024-11-10/news_2024-11-10.csv")
        prices_csv: Path to prices CSV (e.g., "NVDA_2024-11-10/nvda_prices_around_news_2024-11-10.csv")
        target_date: Optional target date (will use first news date if not provided)
        
    Returns:
        Dictionary with:
        - upper_margin: float
        - lower_margin: float
        - reasoning: str
        - confidence: float
        - current_price: float
        - date: datetime
        - news_count: int
        
    Example:
        >>> result = predict_margins_from_fetched_data(
        ...     "NVDA_2024-11-10/news_2024-11-10.csv",
        ...     "NVDA_2024-11-10/nvda_prices_around_news_2024-11-10.csv"
        ... )
        >>> print(f"Upper: ${result['upper_margin']:.2f}")
        >>> print(f"Lower: ${result['lower_margin']:.2f}")
    """
    logger.info(f"Loading data from {news_csv} and {prices_csv}")
    
    # Load news data
    news_df = pd.read_csv(news_csv)
    if news_df.empty:
        raise ValueError(f"No news data found in {news_csv}")
    
    # Parse timestamps
    if 'published_at' in news_df.columns:
        news_df['published_at'] = pd.to_datetime(news_df['published_at'])
    
    # Combine news articles into text
    news_articles = []
    for _, row in news_df.iterrows():
        # Combine title and description
        title = row.get('title', '')
        desc = row.get('description', '')
        combined = f"{title}. {desc}".strip()
        if combined:
            news_articles.append(combined)
    
    logger.info(f"Loaded {len(news_articles)} news articles")
    
    # Load price data
    prices_df = pd.read_csv(prices_csv)
    if prices_df.empty:
        raise ValueError(f"No price data found in {prices_csv}")
    
    # Parse timestamps
    if 'timestamp (UTC)' in prices_df.columns:
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp (UTC)'])
    elif 'timestamp (America/New_York)' in prices_df.columns:
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp (America/New_York)'])
    else:
        # Try to find any timestamp column
        timestamp_cols = [col for col in prices_df.columns if 'timestamp' in col.lower()]
        if timestamp_cols:
            prices_df['timestamp'] = pd.to_datetime(prices_df[timestamp_cols[0]])
        else:
            raise ValueError("No timestamp column found in prices CSV")
    
    # Get the opening price (first price in the dataset)
    current_price = prices_df['close'].iloc[0]
    
    # Determine target date
    if target_date is None:
        # Use the first news article date
        if 'published_at' in news_df.columns:
            target_date = news_df['published_at'].iloc[0]
        else:
            target_date = datetime.now()
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    logger.info(f"Predicting margins for {target_date.date()} at price ${current_price:.2f}")
    
    # Call margin predictor
    result = predict_margins(
        news_articles=news_articles,
        current_price=current_price,
        date=target_date
    )
    
    # Add metadata
    result['news_count'] = len(news_articles)
    result['price_data_points'] = len(prices_df)
    
    return result


def batch_predict_margins(
    data_dir: str | Path,
    output_csv: str | Path = None
) -> pd.DataFrame:
    """
    Predict margins for multiple dates in a directory.
    
    Looks for subdirectories like "NVDA_2024-11-10" and processes each one.
    
    Args:
        data_dir: Directory containing date subdirectories
        output_csv: Optional path to save results CSV
        
    Returns:
        DataFrame with predictions for each date
    """
    data_dir = Path(data_dir)
    results = []
    
    # Find all date directories
    date_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('NVDA_')]
    
    logger.info(f"Found {len(date_dirs)} date directories to process")
    
    for date_dir in sorted(date_dirs):
        # Extract date from directory name (e.g., "NVDA_2024-11-10" -> "2024-11-10")
        date_str = date_dir.name.split('_', 1)[1] if '_' in date_dir.name else None
        
        # Find news and price CSVs
        news_csv = date_dir / f"news_{date_str}.csv"
        prices_csv = date_dir / f"nvda_prices_around_news_{date_str}.csv"
        
        if not news_csv.exists() or not prices_csv.exists():
            logger.warning(f"Skipping {date_dir.name}: missing CSV files")
            continue
        
        try:
            result = predict_margins_from_fetched_data(news_csv, prices_csv, date_str)
            results.append({
                'date': result['date'],
                'current_price': result['current_price'],
                'upper_margin': result['upper_margin'],
                'lower_margin': result['lower_margin'],
                'confidence': result['confidence'],
                'sentiment': result['sentiment_aggregate'],
                'expected_move_pct': result['expected_move_pct'],
                'news_count': result['news_count'],
                'reasoning': result['reasoning']
            })
            logger.info(f"✓ Processed {date_str}: Upper=${result['upper_margin']:.2f}, Lower=${result['lower_margin']:.2f}")
        except Exception as e:
            logger.error(f"✗ Failed to process {date_dir.name}: {e}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save if output path provided
    if output_csv and not results_df.empty:
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(results_df)} predictions to {output_csv}")
    
    return results_df

