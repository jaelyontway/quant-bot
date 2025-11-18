"""
Training module for margin prediction - MACHINE LEARNING MODEL.

This module learns optimal margin parameters from historical data
by analyzing past news events and their impact on stock prices.

Uses supervised learning (Random Forest Regression) to predict
optimal buy/sell margins based on sentiment and other features.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class MarginPredictor:
    """
    Machine Learning model for predicting trading margins.

    Uses Random Forest Regression to learn from historical data:
    - News sentiment scores
    - Historical volatility
    - Day of week patterns
    - News volume

    Predicts optimal upper/lower margins based on these features.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the ML model.

        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
        """
        self.upper_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5
        )
        self.lower_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5
        )
        self.is_trained = False
        self.training_stats = {}
        self.feature_names = []
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from raw data.

        Features:
        - sentiment_score: Base sentiment
        - sentiment_squared: Non-linear sentiment effect
        - sentiment_abs: Absolute sentiment (strength regardless of direction)
        - day_of_week: Weekday patterns (0=Monday, 4=Friday)
        - news_count: Number of news articles (if available)
        - volatility: Historical volatility (if available)
        """
        features = pd.DataFrame()

        # Base sentiment features
        features['sentiment_score'] = data['sentiment_score']
        features['sentiment_squared'] = data['sentiment_score'] ** 2
        features['sentiment_abs'] = data['sentiment_score'].abs()

        # Time-based features
        if 'date' in data.columns:
            features['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
        else:
            features['day_of_week'] = 2  # Default to Wednesday

        # News volume (if available)
        if 'news_count' in data.columns:
            features['news_count'] = data['news_count']
        else:
            features['news_count'] = 1

        # Volatility (if available)
        if 'volatility' in data.columns:
            features['volatility'] = data['volatility']
        else:
            features['volatility'] = 0.02  # Default 2% daily volatility

        return features

    def train(self, training_data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the ML model on historical data with train/test split.

        Expected DataFrame columns:
        - date: datetime
        - news_text: str (combined news for that day)
        - price_at_news: float (price when news came out)
        - max_price_next_3d: float (highest price in next 3 days)
        - min_price_next_3d: float (lowest price in next 3 days)
        - sentiment_score: float (optional, will compute if missing)
        - news_count: int (optional)
        - volatility: float (optional)

        Args:
            training_data: DataFrame with historical news and price data
            test_size: Fraction of data to use for testing (default 0.2)

        Returns:
            Dictionary with training and test statistics
        """
        logger.info(f"Training ML model on {len(training_data)} samples")

        # Compute sentiment if not provided
        if 'sentiment_score' not in training_data.columns:
            from .sentiment import local_vader_sentiment
            logger.info("Computing sentiment scores...")
            training_data['sentiment_score'] = training_data['news_text'].apply(
                lambda x: local_vader_sentiment(x)['sentiment_score']
            )

        # Calculate actual margins (what happened)
        training_data['actual_upper_margin_pct'] = (
            (training_data['max_price_next_3d'] - training_data['price_at_news'])
            / training_data['price_at_news']
        )
        training_data['actual_lower_margin_pct'] = (
            (training_data['min_price_next_3d'] - training_data['price_at_news'])
            / training_data['price_at_news']
        )

        # Engineer features
        X = self._engineer_features(training_data)
        self.feature_names = list(X.columns)

        y_upper = training_data['actual_upper_margin_pct'].values
        y_lower = training_data['actual_lower_margin_pct'].values

        # Train/test split
        X_train, X_test, y_upper_train, y_upper_test = train_test_split(
            X, y_upper, test_size=test_size, random_state=42
        )
        _, _, y_lower_train, y_lower_test = train_test_split(
            X, y_lower, test_size=test_size, random_state=42
        )

        # Train models
        logger.info("Training Random Forest models...")
        self.upper_model.fit(X_train, y_upper_train)
        self.lower_model.fit(X_train, y_lower_train)
        self.is_trained = True

        # Evaluate on test set
        upper_pred_test = self.upper_model.predict(X_test)
        lower_pred_test = self.lower_model.predict(X_test)

        upper_pred_train = self.upper_model.predict(X_train)
        lower_pred_train = self.lower_model.predict(X_train)

        # Calculate statistics
        self.training_stats = {
            'n_samples': len(training_data),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'features': self.feature_names,

            # Training metrics
            'upper_r2_train': r2_score(y_upper_train, upper_pred_train),
            'lower_r2_train': r2_score(y_lower_train, lower_pred_train),
            'upper_mae_train': mean_absolute_error(y_upper_train, upper_pred_train),
            'lower_mae_train': mean_absolute_error(y_lower_train, lower_pred_train),

            # Test metrics (generalization)
            'upper_r2_test': r2_score(y_upper_test, upper_pred_test),
            'lower_r2_test': r2_score(y_lower_test, lower_pred_test),
            'upper_mae_test': mean_absolute_error(y_upper_test, upper_pred_test),
            'lower_mae_test': mean_absolute_error(y_lower_test, lower_pred_test),

            # Feature importance
            'upper_feature_importance': dict(zip(
                self.feature_names,
                self.upper_model.feature_importances_
            )),
            'lower_feature_importance': dict(zip(
                self.feature_names,
                self.lower_model.feature_importances_
            )),

            # Data statistics
            'avg_sentiment': float(np.mean(training_data['sentiment_score'])),
            'avg_upper_move_pct': float(np.mean(y_upper) * 100),
            'avg_lower_move_pct': float(np.mean(y_lower) * 100),
        }

        logger.info(f"✅ Training complete!")
        logger.info(f"   Upper margin - Train R²: {self.training_stats['upper_r2_train']:.3f}, "
                   f"Test R²: {self.training_stats['upper_r2_test']:.3f}")
        logger.info(f"   Lower margin - Train R²: {self.training_stats['lower_r2_train']:.3f}, "
                   f"Test R²: {self.training_stats['lower_r2_test']:.3f}")

        return self.training_stats
    
    def predict(
        self,
        sentiment_score: float,
        current_price: float,
        date: Optional[datetime] = None,
        news_count: int = 1,
        volatility: float = 0.02
    ) -> Dict:
        """
        Predict margins using the trained ML model.

        Args:
            sentiment_score: Sentiment score (-1 to 1)
            current_price: Current stock price
            date: Date for day-of-week feature (optional)
            news_count: Number of news articles (default 1)
            volatility: Historical volatility (default 0.02)

        Returns:
            Dictionary with upper_margin, lower_margin, confidence, etc.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")

        # Prepare features
        feature_data = pd.DataFrame([{
            'sentiment_score': sentiment_score,
            'date': date or datetime.now(),
            'news_count': news_count,
            'volatility': volatility
        }])

        X = self._engineer_features(feature_data)

        # Predict percentage moves
        upper_pct = self.upper_model.predict(X)[0]
        lower_pct = self.lower_model.predict(X)[0]

        # Convert to absolute prices
        upper_margin = current_price * (1 + upper_pct)
        lower_margin = current_price * (1 + lower_pct)

        # Ensure margins make sense (upper > current > lower)
        if upper_margin < current_price:
            upper_margin = current_price * 1.02  # At least 2% above
        if lower_margin > current_price:
            lower_margin = current_price * 0.98  # At least 2% below

        # Calculate confidence based on model performance
        confidence = (self.training_stats.get('upper_r2_test', 0.5) +
                     self.training_stats.get('lower_r2_test', 0.5)) / 2

        return {
            'upper_margin': float(upper_margin),
            'lower_margin': float(lower_margin),
            'upper_pct': float(upper_pct * 100),
            'lower_pct': float(lower_pct * 100),
            'sentiment_score': sentiment_score,
            'current_price': current_price,
            'confidence': float(confidence),
            'model': 'random_forest_trained',
            'n_training_samples': self.training_stats.get('n_samples', 0)
        }
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model (e.g., 'models/margin_predictor.pkl')
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'upper_model': self.upper_model,
            'lower_model': self.lower_model,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'is_trained': self.is_trained
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'MarginPredictor':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded MarginPredictor instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        predictor = cls()
        predictor.upper_model = model_data['upper_model']
        predictor.lower_model = model_data['lower_model']
        predictor.feature_names = model_data['feature_names']
        predictor.training_stats = model_data['training_stats']
        predictor.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"   Trained on {predictor.training_stats['n_samples']} samples")
        logger.info(f"   Test R²: Upper={predictor.training_stats['upper_r2_test']:.3f}, "
                   f"Lower={predictor.training_stats['lower_r2_test']:.3f}")

        return predictor


def create_training_data_from_csv(
    news_csv: str,
    prices_csv: str,
    look_forward_days: int = 3
) -> pd.DataFrame:
    """
    Create training dataset from news and price CSVs.
    
    Args:
        news_csv: Path to news CSV with columns: date, news_text
        prices_csv: Path to prices CSV with columns: Date, Close
        look_forward_days: Days to look forward for max/min prices
        
    Returns:
        DataFrame ready for training
    """
    logger.info(f"Creating training data from {news_csv} and {prices_csv}")
    
    # Load data
    news_df = pd.read_csv(news_csv, parse_dates=['date'])
    prices_df = pd.read_csv(prices_csv, parse_dates=['Date'])
    prices_df = prices_df.set_index('Date')
    
    training_rows = []
    
    for _, news_row in news_df.iterrows():
        news_date = news_row['date']
        news_text = news_row['news_text']
        
        # Get price at news time
        try:
            price_at_news = prices_df.loc[news_date, 'Close']
        except KeyError:
            continue  # Skip if no price data
        
        # Get max/min prices in next N days
        end_date = news_date + timedelta(days=look_forward_days)
        future_prices = prices_df.loc[news_date:end_date, 'Close']
        
        if len(future_prices) < 2:
            continue  # Not enough data
        
        max_price = future_prices.max()
        min_price = future_prices.min()
        
        training_rows.append({
            'date': news_date,
            'news_text': news_text,
            'price_at_news': price_at_news,
            'max_price_next_3d': max_price,
            'min_price_next_3d': min_price
        })
    
    training_df = pd.DataFrame(training_rows)
    logger.info(f"Created {len(training_df)} training samples")
    
    return training_df

