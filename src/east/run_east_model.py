#!/usr/bin/env python3
"""
Run EAST Model

Takes in news CSV and price CSV, runs the model, outputs to output.txt.

Usage:
    python run_east_model.py <news_csv> <price_csv>

Example:
    python run_east_model.py quant-bot/feedcsv.csv quant-bot/feedcsv2.csv
"""

import sys
import logging
from pathlib import Path

# Add parent directory (quant-bot/src) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from east.model import run_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run_east_model.py <news_csv> <price_csv>")
        print("\nExample:")
        print("  python run_east_model.py quant-bot/feedcsv.csv quant-bot/feedcsv2.csv")
        sys.exit(1)

    news_csv, price_csv = sys.argv[1], sys.argv[2]

    if not Path(news_csv).exists():
        print(f"Error: News CSV not found: {news_csv}")
        sys.exit(1)

    if not Path(price_csv).exists():
        print(f"Error: Price CSV not found: {price_csv}")
        sys.exit(1)

    logger.info("="*60)
    logger.info("EAST Model - Running")
    logger.info("="*60)
    logger.info(f"News CSV: {news_csv}")
    logger.info(f"Price CSV: {price_csv}")

    # Run model
    signal, margins = run_model(news_csv, price_csv)

    # Save to output.txt in the same directory as this script
    output_path = Path(__file__).parent / "output.txt"

    with open(output_path, 'w') as f:
        f.write(f"UPPER_MARGIN: {margins['upper_margin']:.2f}\n")
        f.write(f"LOWER_MARGIN: {margins['lower_margin']:.2f}\n")
        f.write(f"TARGET_PRICE: {margins['current_price']:.2f}\n")

    # Print results
    logger.info("="*60)
    logger.info(f"SIGNAL: {signal}")
    logger.info(f"CURRENT_PRICE: ${margins['current_price']:.2f}")
    logger.info(f"UPPER_MARGIN: ${margins['upper_margin']:.2f} ({margins['upper_margin_pct']:.2%})")
    logger.info(f"LOWER_MARGIN: ${margins['lower_margin']:.2f} ({margins['lower_margin_pct']:.2%})")
    logger.info(f"SENTIMENT: {margins['sentiment_score']:.3f}")
    logger.info(f"COVERAGE: {margins['coverage']} articles")
    logger.info(f"Output saved to: {output_path}")
    logger.info("="*60)

