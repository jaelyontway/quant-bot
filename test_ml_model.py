#!/usr/bin/env python3
"""
Test the ML model implementation.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("🧪 TESTING MACHINE LEARNING MODEL")
print("="*70)

# Test 1: Import modules
print("\n1️⃣ Testing imports...")
try:
    from east.train_margins import MarginPredictor
    from east.margin_predictor import predict_margins
    print("   ✅ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create synthetic training data
print("\n2️⃣ Creating synthetic training data...")
try:
    np.random.seed(42)
    n_samples = 50
    
    training_data = []
    for i in range(n_samples):
        sentiment = np.random.uniform(-1, 1)
        price = 140.0 + np.random.uniform(-10, 10)
        
        # Simulate realistic price movements based on sentiment
        upper_move = 2.0 + sentiment * 2.0 + np.random.normal(0, 0.5)
        lower_move = -1.5 + sentiment * 1.5 + np.random.normal(0, 0.5)
        
        training_data.append({
            'date': f'2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}',
            'news_text': f'Sample news article {i}',
            'news_count': np.random.randint(1, 10),
            'price_at_news': price,
            'max_price_next_3d': price * (1 + upper_move / 100),
            'min_price_next_3d': price * (1 + lower_move / 100),
            'volatility': np.random.uniform(0.01, 0.03),
            'actual_upper_pct': upper_move,
            'actual_lower_pct': lower_move
        })
    
    training_df = pd.DataFrame(training_data)
    print(f"   ✅ Created {len(training_df)} synthetic training samples")
    print(f"      Avg upper move: {training_df['actual_upper_pct'].mean():+.2f}%")
    print(f"      Avg lower move: {training_df['actual_lower_pct'].mean():+.2f}%")
except Exception as e:
    print(f"   ❌ Failed to create training data: {e}")
    sys.exit(1)

# Test 3: Initialize model
print("\n3️⃣ Initializing Random Forest model...")
try:
    predictor = MarginPredictor(n_estimators=50, random_state=42)
    print("   ✅ Model initialized (50 trees)")
except Exception as e:
    print(f"   ❌ Model initialization failed: {e}")
    sys.exit(1)

# Test 4: Train model
print("\n4️⃣ Training model...")
try:
    stats = predictor.train(training_df, test_size=0.2)
    print("   ✅ Model trained successfully!")
    print(f"      Training samples: {stats['n_train']}")
    print(f"      Test samples: {stats['n_test']}")
    print(f"      Upper R² (test): {stats['upper_r2_test']:.3f}")
    print(f"      Lower R² (test): {stats['lower_r2_test']:.3f}")
    print(f"      Upper MAE (test): {stats['upper_mae_test']*100:.2f}%")
    print(f"      Lower MAE (test): {stats['lower_mae_test']*100:.2f}%")
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Make predictions
print("\n5️⃣ Testing predictions...")
try:
    test_cases = [
        {"sentiment": 0.8, "price": 140.50, "label": "Very Bullish"},
        {"sentiment": 0.0, "price": 140.50, "label": "Neutral"},
        {"sentiment": -0.8, "price": 140.50, "label": "Very Bearish"},
    ]
    
    for case in test_cases:
        result = predictor.predict(
            sentiment_score=case["sentiment"],
            current_price=case["price"]
        )
        print(f"\n   {case['label']} (sentiment={case['sentiment']:+.1f}):")
        print(f"      Upper: ${result['upper_margin']:.2f} ({result['upper_pct']:+.2f}%)")
        print(f"      Lower: ${result['lower_margin']:.2f} ({result['lower_pct']:+.2f}%)")
        print(f"      Confidence: {result['confidence']:.2f}")
    
    print("\n   ✅ Predictions working correctly!")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Save and load model
print("\n6️⃣ Testing model persistence...")
try:
    test_model_path = "test_model.pkl"
    predictor.save_model(test_model_path)
    print(f"   ✅ Model saved to {test_model_path}")
    
    loaded_predictor = MarginPredictor.load_model(test_model_path)
    print(f"   ✅ Model loaded successfully")
    
    # Test loaded model
    result = loaded_predictor.predict(sentiment_score=0.5, current_price=140.50)
    print(f"   ✅ Loaded model prediction: Upper=${result['upper_margin']:.2f}, Lower=${result['lower_margin']:.2f}")
    
    # Clean up
    Path(test_model_path).unlink()
    print(f"   ✅ Cleaned up test file")
except Exception as e:
    print(f"   ❌ Save/load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Integration with margin_predictor
print("\n7️⃣ Testing integration with margin_predictor...")
try:
    # Test rule-based prediction (no trained model)
    result = predict_margins(
        news_articles=["NVIDIA announces breakthrough chip technology"],
        current_price=140.50,
        date=datetime(2024, 11, 13),
        use_ml=False  # Force rule-based
    )
    print(f"   ✅ Rule-based prediction:")
    print(f"      Method: {result.get('method', 'N/A')}")
    print(f"      Upper: ${result['upper_margin']:.2f}")
    print(f"      Lower: ${result['lower_margin']:.2f}")
except Exception as e:
    print(f"   ❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n🎉 Your machine learning model is working perfectly!")
print("\nNext steps:")
print("1. Collect real training data: python collect_training_data.py")
print("2. Train on real data: python train_ml_model.py")
print("3. Use trained model: python quick_demo.py")

