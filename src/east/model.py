"""Main EAST model"""
import pandas as pd
from typing import Dict, Tuple
from . import preprocess, embeddings, clustering, sentiment, signals

def load_news_csv(csv_path: str) -> pd.DataFrame:
    """Load news CSV with flexible column handling."""
    df = pd.read_csv(csv_path)
    
    # Handle different column name formats
    rename_map = {}
    if 'published_at' in df.columns:
        rename_map['published_at'] = 'published_utc'
    elif 'published_date_utc' in df.columns:
        rename_map['published_date_utc'] = 'published_utc'
    
    if 'source_domain' in df.columns:
        rename_map['source_domain'] = 'source'
    
    if 'description' in df.columns:
        rename_map['description'] = 'text'
    elif 'content' in df.columns:
        rename_map['content'] = 'text'
    elif 'summary' in df.columns:
        rename_map['summary'] = 'text'
    
    df = df.rename(columns=rename_map)
    
    # Ensure required columns exist
    if 'published_utc' in df.columns:
        df['published_utc'] = pd.to_datetime(df['published_utc'])
    
    if 'text' not in df.columns:
        df['text'] = df['title']
    else:
        df['text'] = df['text'].fillna(df['title'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)
    
    return df

def load_price_csv(csv_path: str) -> pd.DataFrame:
    """Load price CSV."""
    df = pd.read_csv(csv_path)
    return df

def run_model(news_csv_path: str, price_csv_path: str) -> Tuple[str, Dict]:
    """
    Run EAST model with news and price data.
    
    Args:
        news_csv_path: Path to news CSV
        price_csv_path: Path to price CSV
    
    Returns:
        Tuple of (signal, margins_dict)
    """
    # Load data
    news_df = load_news_csv(news_csv_path)
    price_df = load_price_csv(price_csv_path)
    
    # Get current price from first row's Open column
    current_price = float(price_df.iloc[0]['open'])
    
    # Preprocess text
    news_df = preprocess.normalize_text(news_df, text_col='text')
    news_df = preprocess.combine_title_text(news_df)
    
    # Generate embeddings
    texts = news_df['combined_text'].tolist()
    embs = embeddings.embed_texts(texts, show_progress=False)
    
    # Cluster into events
    timestamps = news_df['published_utc'].tolist()
    titles = news_df['title'].tolist()
    events = clustering.cluster_embeddings(embs, texts, timestamps, titles)
    
    # Analyze sentiment and generate signals
    event_results = []
    for event in events:
        sent = sentiment.get_event_sentiment(event=event, articles=news_df, method='local')
        sig = signals.generate_signal(sent['sentiment_score'], event.coverage_count)
        event_results.append({
            'event': event,
            'sentiment': sent,
            'signal': sig,
            'coverage': event.coverage_count,
            'sentiment_score': sent['sentiment_score']
        })
    
    # Select primary signal
    action_events = [e for e in event_results if e['signal'] != 'HOLD']
    if action_events:
        action_events.sort(key=lambda e: (e['coverage'], abs(e['sentiment_score'])), reverse=True)
        primary = action_events[0]
    else:
        event_results.sort(key=lambda e: e['coverage'], reverse=True)
        primary = event_results[0]
    
    # Calculate margins using current price
    margins = signals.calculate_margins(
        sentiment_score=primary['sentiment_score'],
        signal=primary['signal'],
        current_price=current_price
    )
    
    # Add metadata
    margins['current_price'] = current_price
    margins['sentiment_score'] = primary['sentiment_score']
    margins['coverage'] = primary['coverage']
    
    return primary['signal'], margins

def save_output(signal: str, margins: Dict, output_path: str):
    """Save results to text file."""
    with open(output_path, 'w') as f:
        f.write(f"SIGNAL: {signal}\n")
        f.write(f"CURRENT_PRICE: {margins['current_price']:.2f}\n")
        f.write(f"UPPER_MARGIN: {margins['upper_margin']:.2f}\n")
        f.write(f"LOWER_MARGIN: {margins['lower_margin']:.2f}\n")
        f.write(f"UPPER_MARGIN_PCT: {margins['upper_margin_pct']:.4f}\n")
        f.write(f"LOWER_MARGIN_PCT: {margins['lower_margin_pct']:.4f}\n")
        f.write(f"SENTIMENT: {margins['sentiment_score']:.4f}\n")
        f.write(f"COVERAGE: {margins['coverage']}\n")

