"""Text preprocessing"""
import re
import pandas as pd

def normalize_text(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    df = df.copy()
    df[f"{text_col}_normalized"] = df[text_col].apply(_normalize_single_text)
    return df

def _normalize_single_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s.,!?;:\-\'\"()]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def combine_title_text(df: pd.DataFrame, title_col: str = 'title', text_col: str = 'text') -> pd.DataFrame:
    df = df.copy()
    df['combined_text'] = df[title_col] + ". " + df[text_col]
    return df

