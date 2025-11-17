#!/usr/bin/env python3
"""
Create dummy training data for testing the ML model.
Simulates realistic news-price correlations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

# Sample news templates with known sentiment
BULLISH_NEWS = [
    "NVIDIA announces breakthrough AI chip with {pct}% performance improvement",
    "NVIDIA beats earnings expectations, revenue up {pct}%",
    "Analysts upgrade NVIDIA to strong buy, price target raised",
    "NVIDIA secures major contract with tech giants",
    "NVIDIA unveils revolutionary GPU architecture",
    "Strong demand for NVIDIA AI chips drives record sales",
    "NVIDIA partners with major cloud providers for AI expansion",
    "NVIDIA reports exceptional quarterly results",
    "Tech sector rallies as NVIDIA leads innovation",
    "NVIDIA stock surges on positive analyst reports",
]

BEARISH_NEWS = [
    "NVIDIA faces supply chain disruptions, delays expected",
    "Government shutdown threatens tech sector funding",
    "NVIDIA misses earnings expectations, stock falls",
    "Analysts downgrade NVIDIA citing market concerns",
    "NVIDIA warns of slower growth in coming quarters",
    "Competition intensifies in AI chip market",
    "NVIDIA faces regulatory scrutiny over market dominance",
    "Tech sector selloff impacts NVIDIA stock",
    "NVIDIA production delays raise investor concerns",
    "Market volatility hits NVIDIA hard",
]

NEUTRAL_NEWS = [
    "NVIDIA maintains steady performance in Q{q}",
    "NVIDIA announces routine product update",
    "NVIDIA CEO speaks at industry conference",
    "NVIDIA releases quarterly financial report",
    "NVIDIA stock trades sideways amid mixed signals",
    "Analysts maintain hold rating on NVIDIA",
    "NVIDIA participates in tech industry summit",
    "NVIDIA updates corporate governance policies",
    "NVIDIA announces minor organizational changes",
    "NVIDIA stock shows typical market correlation",
]


def generate_dummy_training_data(n_samples=200, output_file="dummy_training_data.csv"):
    """
    Generate dummy training data with realistic news-price correlations.
    
    Args:
        n_samples: Number of training samples to generate
        output_file: Output CSV filename
    
    Returns:
        DataFrame with training data
    """
    
    print(f"🎲 Generating {n_samples} dummy training samples...")
    
    training_data = []
    base_price = 140.0
    start_date = datetime(2024, 1, 1)
    
    for i in range(n_samples):
        # Random date
        date = start_date + timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        
        # Random sentiment category (40% bullish, 40% bearish, 20% neutral)
        rand = np.random.random()
        if rand < 0.4:
            # Bullish news
            template = np.random.choice(BULLISH_NEWS)
            pct = np.random.randint(10, 50)
            news_text = template.format(pct=pct, q=np.random.randint(1, 5))
            
            # Bullish news → positive price movement
            base_upper = np.random.uniform(3.0, 8.0)  # 3-8% up
            base_lower = np.random.uniform(-1.0, 1.0)  # -1 to +1% down
            
        elif rand < 0.8:
            # Bearish news
            template = np.random.choice(BEARISH_NEWS)
            news_text = template.format(q=np.random.randint(1, 5))
            
            # Bearish news → negative price movement
            base_upper = np.random.uniform(0.5, 2.0)  # 0.5-2% up
            base_lower = np.random.uniform(-8.0, -3.0)  # -8 to -3% down
            
        else:
            # Neutral news
            template = np.random.choice(NEUTRAL_NEWS)
            news_text = template.format(q=np.random.randint(1, 5))
            
            # Neutral news → small movements
            base_upper = np.random.uniform(1.0, 3.0)  # 1-3% up
            base_lower = np.random.uniform(-3.0, -1.0)  # -3 to -1% down
        
        # Add some noise to make it realistic
        noise_upper = np.random.normal(0, 0.5)
        noise_lower = np.random.normal(0, 0.5)
        
        actual_upper_pct = base_upper + noise_upper
        actual_lower_pct = base_lower + noise_lower
        
        # Random price variation
        price_variation = np.random.uniform(-5, 5)
        price_at_news = base_price + price_variation
        
        # Calculate actual prices
        max_price = price_at_news * (1 + actual_upper_pct / 100)
        min_price = price_at_news * (1 + actual_lower_pct / 100)
        
        # Random volatility
        volatility = np.random.uniform(0.015, 0.035)
        
        # Random news count
        news_count = np.random.randint(1, 8)
        
        training_data.append({
            'date': date_str,
            'news_text': news_text,
            'news_count': news_count,
            'price_at_news': price_at_news,
            'max_price_next_3d': max_price,
            'min_price_next_3d': min_price,
            'volatility': volatility,
            'actual_upper_pct': actual_upper_pct,
            'actual_lower_pct': actual_lower_pct
        })
    
    # Create DataFrame
    df = pd.DataFrame(training_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created {len(df)} samples")
    print(f"📊 Statistics:")
    print(f"   Average upper move: {df['actual_upper_pct'].mean():+.2f}%")
    print(f"   Average lower move: {df['actual_lower_pct'].mean():+.2f}%")
    print(f"   Price range: ${df['price_at_news'].min():.2f} - ${df['price_at_news'].max():.2f}")
    print(f"   Volatility range: {df['volatility'].min():.3f} - {df['volatility'].max():.3f}")
    print(f"\n💾 Saved to: {output_file}")
    
    # Show sample
    print(f"\n📰 Sample news:")
    for i in range(3):
        row = df.iloc[i]
        print(f"   {i+1}. {row['news_text'][:60]}...")
        print(f"      Upper: {row['actual_upper_pct']:+.2f}%, Lower: {row['actual_lower_pct']:+.2f}%")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dummy training data")
    parser.add_argument("--samples", type=int, default=200, 
                       help="Number of samples to generate (default: 200)")
    parser.add_argument("--output", default="dummy_training_data.csv",
                       help="Output CSV file (default: dummy_training_data.csv)")
    
    args = parser.parse_args()
    
    # Generate data
    df = generate_dummy_training_data(args.samples, args.output)
    
    print("\n" + "="*70)
    print("✅ DUMMY DATA CREATED!")
    print("="*70)
    print("\nNext steps:")
    print(f"1. Train model: python train_ml_model.py --training-data {args.output}")
    print(f"2. Test predictions: python simple_predict.py --text 'Your text' --price 140.50")

