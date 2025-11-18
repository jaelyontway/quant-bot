#!/usr/bin/env python3
"""
Train the Machine Learning model for margin prediction.

This script:
1. Loads historical training data
2. Trains a Random Forest model
3. Evaluates performance
4. Saves the trained model
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent / "src"))

from east.train_margins import MarginPredictor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_model(
    training_data_csv: str,
    model_output_path: str = "models/margin_predictor.pkl",
    test_size: float = 0.2
):
    """
    Train the ML model.
    
    Args:
        training_data_csv: Path to training data CSV
        model_output_path: Where to save the trained model
        test_size: Fraction of data for testing (default 0.2)
    """
    
    logger.info("="*70)
    logger.info("🤖 TRAINING MACHINE LEARNING MODEL")
    logger.info("="*70)
    
    # Load training data
    logger.info(f"\n📥 Loading training data from {training_data_csv}")
    training_df = pd.read_csv(training_data_csv)
    
    logger.info(f"   Loaded {len(training_df)} samples")
    logger.info(f"   Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    
    # Initialize model
    logger.info("\n🔧 Initializing Random Forest model...")
    predictor = MarginPredictor(n_estimators=100, random_state=42)
    
    # Train
    logger.info(f"\n🎓 Training model (test_size={test_size})...")
    stats = predictor.train(training_df, test_size=test_size)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("📊 TRAINING RESULTS")
    logger.info("="*70)
    
    logger.info(f"\n📈 Dataset:")
    logger.info(f"   Total samples: {stats['n_samples']}")
    logger.info(f"   Training samples: {stats['n_train']}")
    logger.info(f"   Test samples: {stats['n_test']}")
    
    logger.info(f"\n🎯 Model Performance:")
    logger.info(f"   Upper Margin:")
    logger.info(f"      Train R²: {stats['upper_r2_train']:.3f}")
    logger.info(f"      Test R²:  {stats['upper_r2_test']:.3f}")
    logger.info(f"      Test MAE: {stats['upper_mae_test']*100:.2f}%")
    
    logger.info(f"   Lower Margin:")
    logger.info(f"      Train R²: {stats['lower_r2_train']:.3f}")
    logger.info(f"      Test R²:  {stats['lower_r2_test']:.3f}")
    logger.info(f"      Test MAE: {stats['lower_mae_test']*100:.2f}%")
    
    logger.info(f"\n🔍 Feature Importance (Upper Margin):")
    for feature, importance in sorted(
        stats['upper_feature_importance'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        logger.info(f"      {feature}: {importance:.3f}")
    
    logger.info(f"\n📊 Data Statistics:")
    logger.info(f"   Avg sentiment: {stats['avg_sentiment']:+.3f}")
    logger.info(f"   Avg upper move: {stats['avg_upper_move_pct']:+.2f}%")
    logger.info(f"   Avg lower move: {stats['avg_lower_move_pct']:+.2f}%")
    
    # Interpret results
    logger.info("\n" + "="*70)
    logger.info("💡 INTERPRETATION")
    logger.info("="*70)
    
    test_r2_avg = (stats['upper_r2_test'] + stats['lower_r2_test']) / 2
    
    if test_r2_avg > 0.7:
        logger.info("   ✅ EXCELLENT: Model explains >70% of variance")
    elif test_r2_avg > 0.5:
        logger.info("   ✓ GOOD: Model explains >50% of variance")
    elif test_r2_avg > 0.3:
        logger.info("   ⚠️  MODERATE: Model explains >30% of variance")
    else:
        logger.info("   ⚠️  WEAK: Model explains <30% of variance")
        logger.info("   Consider collecting more training data")
    
    # Save model
    logger.info(f"\n💾 Saving model to {model_output_path}")
    predictor.save_model(model_output_path)
    
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nModel saved to: {model_output_path}")
    logger.info(f"Test R² Score: {test_r2_avg:.3f}")
    logger.info("\nYou can now use this trained model for predictions!")
    
    return predictor, stats


def test_prediction(predictor: MarginPredictor):
    """Test the trained model with example predictions."""
    
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING PREDICTIONS")
    logger.info("="*70)
    
    test_cases = [
        {"sentiment": 0.8, "price": 140.50, "label": "Very Bullish"},
        {"sentiment": 0.3, "price": 140.50, "label": "Slightly Bullish"},
        {"sentiment": 0.0, "price": 140.50, "label": "Neutral"},
        {"sentiment": -0.3, "price": 140.50, "label": "Slightly Bearish"},
        {"sentiment": -0.8, "price": 140.50, "label": "Very Bearish"},
    ]
    
    for case in test_cases:
        result = predictor.predict(
            sentiment_score=case["sentiment"],
            current_price=case["price"]
        )
        
        logger.info(f"\n{case['label']} (sentiment={case['sentiment']:+.1f}):")
        logger.info(f"   Upper: ${result['upper_margin']:.2f} ({result['upper_pct']:+.2f}%)")
        logger.info(f"   Lower: ${result['lower_margin']:.2f} ({result['lower_pct']:+.2f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ML model for margin prediction")
    parser.add_argument("--training-data", default="training_data.csv",
                       help="Path to training data CSV")
    parser.add_argument("--output", default="models/margin_predictor.pkl",
                       help="Where to save the trained model")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Fraction of data for testing (default 0.2)")
    parser.add_argument("--test-predictions", action="store_true",
                       help="Run test predictions after training")
    
    args = parser.parse_args()
    
    # Check if training data exists
    if not Path(args.training_data).exists():
        logger.error(f"❌ Training data not found: {args.training_data}")
        logger.error("   Run collect_training_data.py first!")
        sys.exit(1)
    
    # Train model
    predictor, stats = train_model(
        training_data_csv=args.training_data,
        model_output_path=args.output,
        test_size=args.test_size
    )
    
    # Test predictions
    if args.test_predictions:
        test_prediction(predictor)

