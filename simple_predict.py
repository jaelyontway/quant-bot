#!/usr/bin/env python3
"""
Simple interface: Pass in text, get margin predictions.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from east.sentiment import local_vader_sentiment
from east.train_margins import MarginPredictor


def predict_from_text(
    news_text: str,
    current_price: float,
    model_path: str = "models/margin_predictor.pkl"
):
    """
    Pass in text, get margin predictions.
    
    Args:
        news_text: Amalgamation of news text (can be multiple articles combined)
        current_price: Current stock price
        model_path: Path to trained model (optional)
    
    Returns:
        Dictionary with upper_margin, lower_margin, reasoning
    """
    
    print(f"\n📰 Analyzing news text ({len(news_text)} characters)...")
    
    # Step 1: Get sentiment from text
    sentiment_result = local_vader_sentiment(news_text)
    sentiment_score = sentiment_result['sentiment_score']
    effect = sentiment_result['short_term_effect']

    print(f"   Sentiment: {sentiment_score:+.3f} ({effect})")
    
    # Step 2: Try to load trained ML model
    model = None
    if Path(model_path).exists():
        try:
            model = MarginPredictor.load_model(model_path)
            print(f"   ✅ Using trained ML model")
        except Exception as e:
            print(f"   ⚠️  Could not load model: {e}")
    
    # Step 3: Make prediction
    if model is not None:
        # Use ML model
        result = model.predict(
            sentiment_score=sentiment_score,
            current_price=current_price
        )
        result['method'] = 'ml_trained'
    else:
        # Fallback to rule-based
        print(f"   ⚠️  No trained model found, using rule-based prediction")
        
        # Simple rule-based calculation
        volatility = 0.02  # 2% default
        expected_move = abs(sentiment_score) * volatility * 3
        
        if sentiment_score > 0:
            upper_pct = expected_move * 1.5
            lower_pct = -expected_move * 0.5
        else:
            upper_pct = expected_move * 0.5
            lower_pct = -expected_move * 1.5
        
        result = {
            'upper_margin': current_price * (1 + upper_pct),
            'lower_margin': current_price * (1 + lower_pct),
            'upper_pct': upper_pct * 100,
            'lower_pct': lower_pct * 100,
            'confidence': 0.5,
            'method': 'rule_based'
        }
    
    # Add reasoning
    result['sentiment_score'] = sentiment_score
    result['reasoning'] = (
        f"Sentiment: {sentiment_score:+.2f}. "
        f"Predicted upper margin: {result['upper_pct']:+.2f}%, "
        f"lower margin: {result['lower_pct']:+.2f}%. "
        f"Method: {result['method']}"
    )
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict margins from text")
    parser.add_argument("--text", required=True, help="News text to analyze")
    parser.add_argument("--price", type=float, required=True, help="Current stock price")
    parser.add_argument("--model", default="models/margin_predictor.pkl", 
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    # Make prediction
    result = predict_from_text(args.text, args.price, args.model)
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\n💰 Current Price: ${args.price:.2f}")
    print(f"\n📈 Upper Margin (SELL): ${result['upper_margin']:.2f} ({result['upper_pct']:+.2f}%)")
    print(f"📉 Lower Margin (BUY):  ${result['lower_margin']:.2f} ({result['lower_pct']:+.2f}%)")
    print(f"\n🎯 Confidence: {result['confidence']:.2f}")
    print(f"🔧 Method: {result['method']}")
    print(f"\n💡 Reasoning: {result['reasoning']}")
    print("\n" + "="*70)

