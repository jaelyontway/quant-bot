#!/usr/bin/env python3
"""
Collect historical training data for the ML model.

This script:
1. Fetches historical news and prices for multiple dates
2. Calculates actual price movements (what happened)
3. Creates a training dataset
4. Saves it for model training
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent / "src"))

from east.data_fetching import fetch_news_for_date, fetch_prices_from_day, CONFIG

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def collect_historical_data(
    start_date: str,
    end_date: str,
    keywords: list,
    output_file: str = "training_data.csv"
):
    """
    Collect historical data for training.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        keywords: Keywords to search for
        output_file: Where to save the training data
    """
    
    logger.info(f"Collecting training data from {start_date} to {end_date}")
    logger.info(f"Keywords: {keywords}")
    
    # Generate date range (only weekdays)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    dates = pd.date_range(start, end, freq='B')  # B = business days
    logger.info(f"Processing {len(dates)} business days")
    
    training_rows = []
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"\n📅 Processing {date_str}...")
        
        try:
            # Fetch news for this date
            news_df, domain_counts = fetch_news_for_date(
                date_str, 
                keywords, 
                "America/New_York"
            )
            
            if news_df.empty:
                logger.warning(f"   No news found for {date_str}")
                continue
            
            # Fetch prices (3 days forward)
            prices_df, _ = fetch_prices_from_day(date_str, interval_minutes=60)
            
            if prices_df.empty:
                logger.warning(f"   No price data for {date_str}")
                continue
            
            # Combine news articles
            news_texts = []
            for _, row in news_df.iterrows():
                combined = f"{row.get('title', '')}. {row.get('description', '')}".strip()
                if combined:
                    news_texts.append(combined)
            
            if not news_texts:
                continue
            
            combined_news = " ".join(news_texts)
            
            # Get price at news time (opening price)
            price_at_news = prices_df['close'].iloc[0]
            
            # Calculate max/min prices in next 3 days
            # (prices_df already contains 3 days of data from fetch_prices_from_day)
            max_price = prices_df['close'].max()
            min_price = prices_df['close'].min()
            
            # Calculate volatility
            returns = prices_df['close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.02
            
            training_rows.append({
                'date': date_str,
                'news_text': combined_news,
                'news_count': len(news_texts),
                'price_at_news': price_at_news,
                'max_price_next_3d': max_price,
                'min_price_next_3d': min_price,
                'volatility': volatility,
                'actual_upper_pct': ((max_price - price_at_news) / price_at_news) * 100,
                'actual_lower_pct': ((min_price - price_at_news) / price_at_news) * 100
            })
            
            logger.info(f"   ✓ {len(news_texts)} articles, "
                       f"Price: ${price_at_news:.2f}, "
                       f"Range: {training_rows[-1]['actual_lower_pct']:+.2f}% to "
                       f"{training_rows[-1]['actual_upper_pct']:+.2f}%")
            
        except Exception as e:
            logger.error(f"   ✗ Error processing {date_str}: {e}")
            continue
    
    # Create DataFrame
    training_df = pd.DataFrame(training_rows)
    
    if training_df.empty:
        logger.error("No training data collected!")
        return None
    
    # Save to CSV
    training_df.to_csv(output_file, index=False)
    logger.info(f"\n✅ Saved {len(training_df)} training samples to {output_file}")
    
    # Print statistics
    logger.info("\n📊 Training Data Statistics:")
    logger.info(f"   Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    logger.info(f"   Total samples: {len(training_df)}")
    logger.info(f"   Avg news per day: {training_df['news_count'].mean():.1f}")
    logger.info(f"   Avg upper move: {training_df['actual_upper_pct'].mean():+.2f}%")
    logger.info(f"   Avg lower move: {training_df['actual_lower_pct'].mean():+.2f}%")
    logger.info(f"   Avg volatility: {training_df['volatility'].mean():.4f}")
    
    return training_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect historical training data")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--keywords", nargs="+", default=["NVIDIA", "AI"], 
                       help="Keywords to search for")
    parser.add_argument("--output", default="training_data.csv",
                       help="Output CSV file")
    
    args = parser.parse_args()
    
    # Check API keys
    if CONFIG.news_api_key == "YOUR_NEWS_API_KEY":
        logger.error("❌ NEWS_API_KEY not set! Set environment variable or update config.")
        sys.exit(1)
    
    if CONFIG.stock_api_key == "YOUR_STOCK_API_KEY":
        logger.error("❌ STOCK_API_KEY not set! Set environment variable or update config.")
        sys.exit(1)
    
    collect_historical_data(
        start_date=args.start_date,
        end_date=args.end_date,
        keywords=args.keywords,
        output_file=args.output
    )

