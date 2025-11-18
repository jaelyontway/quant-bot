#!/usr/bin/env python3
"""
News Fetcher Script
Fetches news articles from Google News RSS based on keywords and date range.
Stores results in both JSON and CSV formats.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import pandas as pd
from urllib.parse import quote
import time
from typing import List, Dict
import re
from newspaper import Article


class NewsFetcher:
    def __init__(self, keywords: List[str], start_date: str, end_date: str):
        """
        Initialize the news fetcher.

        Args:
            keywords: List of keywords to search for
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
        """
        self.keywords = keywords
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.start_date_date = self.start_date.date()
        self.end_date_date = self.end_date.date()
        self.articles = []
        self.utc_tz = ZoneInfo("UTC")
        self.ny_tz = ZoneInfo("America/New_York")

    def build_google_news_url(self) -> str:
        """Build Google News RSS URL with keywords."""
        # Combine keywords into a search query
        query = ' '.join(self.keywords)
        encoded_query = quote(query)

        # Add date range to query (Google News syntax)
        # Format: after:YYYY-MM-DD before:YYYY-MM-DD
        date_filter = f" after:{self.start_date.strftime('%Y-%m-%d')} before:{self.end_date.strftime('%Y-%m-%d')}"
        encoded_query += quote(date_filter)

        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        return url

    def fetch_rss_feed(self) -> feedparser.FeedParserDict:
        """Fetch and parse the RSS feed from Google News."""
        url = self.build_google_news_url()
        print(f"Fetching news from: {url}")

        try:
            feed = feedparser.parse(url)
            print(f"Found {len(feed.entries)} articles in RSS feed")
            return feed
        except Exception as e:
            print(f"Error fetching RSS feed: {e}")
            return None

    def extract_article_content(self, url: str) -> str:
        """
        Extract full article content from URL using newspaper3k library.
        Falls back to BeautifulSoup if newspaper3k fails.
        """
        # Method 1: Try newspaper3k (best for news articles)
        try:
            article = Article(url)
            article.download()
            article.parse()

            if article.text and len(article.text) > 100:
                print(f"    ✓ Extracted {len(article.text)} chars with newspaper3k")
                return article.text
            else:
                print(f"    ⚠ newspaper3k returned short content, trying fallback...")
        except Exception as e:
            print(f"    ⚠ newspaper3k failed ({str(e)[:50]}), trying fallback...")

        # Method 2: Fallback to BeautifulSoup
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()

            # Try to find article content
            article_content = None
            selectors = [
                'article',
                '[class*="article-body"]',
                '[class*="article-content"]',
                '[class*="story-body"]',
                '[class*="post-content"]',
                'main'
            ]

            for selector in selectors:
                if selector == 'article' or selector == 'main':
                    article_content = soup.find(selector)
                else:
                    article_content = soup.select_one(selector)
                if article_content:
                    break

            if article_content:
                text = article_content.get_text(separator='\n', strip=True)
                text = re.sub(r'\n\s*\n', '\n\n', text)
                if len(text) > 100:
                    print(f"    ✓ Extracted {len(text)} chars with BeautifulSoup")
                    return text

            # Last resort: get all paragraphs
            paragraphs = soup.find_all('p')
            text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])

            if text and len(text) > 100:
                print(f"    ✓ Extracted {len(text)} chars from paragraphs")
                return text
            else:
                return "Content extraction failed - article may be behind paywall or blocked"

        except Exception as e:
            print(f"    ✗ All methods failed: {str(e)[:80]}")
            return f"Error: {str(e)}"

    def parse_article_date(self, date_string: str) -> datetime:
        """Parse article published date."""
        try:
            # RSS feeds typically use RFC 2822 format
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_string)
            return dt
        except:
            return None

    def normalize_pub_date(self, pub_date: datetime):
        """Ensure published date has timezone info and build UTC/NY conversions."""
        if not pub_date:
            return None, None, None

        aware_dt = pub_date if pub_date.tzinfo else pub_date.replace(tzinfo=self.utc_tz)
        utc_dt = aware_dt.astimezone(self.utc_tz)
        ny_dt = aware_dt.astimezone(self.ny_tz)
        return aware_dt, utc_dt, ny_dt

    def build_date_fields(self, utc_dt: datetime, ny_dt: datetime) -> Dict[str, str]:
        """Create CSV-friendly date fields mirroring the reference training data."""
        unknown = "Unknown"
        if not utc_dt or not ny_dt:
            return {
                'actual date and time': unknown,
                'published_date_utc': unknown,
                'published_date_ny': unknown,
                'published_time_ny': unknown,
                'published_date_et': unknown
            }

        actual_time = f"Updated {ny_dt.strftime('%I:%M %p')} ET {ny_dt.strftime('%m/%d/%Y')}"
        return {
            'actual date and time': actual_time,
            'published_date_utc': utc_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'published_date_ny': ny_dt.strftime('%Y-%m-%d'),
            'published_time_ny': ny_dt.strftime('%H:%M:%S'),
            'published_date_et': ny_dt.strftime('%m/%d/%Y %H:%M')
        }

    def fetch_articles(self, extract_full_content: bool = True):
        """
        Fetch articles from Google News RSS.

        Args:
            extract_full_content: Whether to extract full article content (slower but more complete)
        """
        feed = self.fetch_rss_feed()

        if not feed or not feed.entries:
            print("No articles found")
            return

        for idx, entry in enumerate(feed.entries):
            print(f"\nProcessing article {idx + 1}/{len(feed.entries)}: {entry.title}")

            # Parse published date
            pub_date = self.parse_article_date(entry.published) if hasattr(entry, 'published') else None
            _, pub_date_utc, pub_date_ny = self.normalize_pub_date(pub_date)

            # Filter by date range if date is available
            if pub_date_ny:
                if not (self.start_date_date <= pub_date_ny.date() <= self.end_date_date):
                    print(f"  Skipping - outside date range ({pub_date.strftime('%Y-%m-%d')})")
                    continue

            # Extract article information
            date_fields = self.build_date_fields(pub_date_utc, pub_date_ny)
            article = {
                'title': entry.title,
                'source': entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Unknown',
                'url': entry.link,
                'summary': entry.summary if hasattr(entry, 'summary') else '',
                'content': ''
            }
            article.update(date_fields)

            # Extract full content if requested
            if extract_full_content:
                print(f"  Extracting full content...")
                article['content'] = self.extract_article_content(entry.link)
                time.sleep(1)  # Be polite to servers
            else:
                article['content'] = article['summary']

            self.articles.append(article)
            print(f"  Added article: {article['title'][:60]}...")

        print(f"\n✓ Successfully fetched {len(self.articles)} articles")

    def save_to_json(self, filename: str):
        """Save articles to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to {filename}")

    def save_to_csv(self, filename: str):
        """Save articles to CSV file."""
        if not self.articles:
            print("No articles to save")
            return

        df = pd.DataFrame(self.articles)
        column_order = [
            'title',
            'source',
            'url',
            'actual date and time',
            'published_date_utc',
            'published_date_ny',
            'published_time_ny',
            'published_date_et',
            'summary',
            'content'
        ]
        df = df[column_order]
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Saved to {filename}")

    def print_summary(self):
        """Print summary of fetched articles."""
        print(f"\n{'='*80}")
        print(f"SUMMARY: Fetched {len(self.articles)} articles")
        print(f"{'='*80}")

        for idx, article in enumerate(self.articles, 1):
            print(f"\n{idx}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Date: {article.get('published_date_et', 'Unknown')}")
            print(f"   URL: {article['url']}")
            print(f"   Content length: {len(article['content'])} characters")


def main():
    """Main function to run the news fetcher."""

    # Configuration
    KEYWORDS = ["government shutdown", "nvidia stock", "US"]
    START_DATE = "2025-10-01"  # Format: YYYY-MM-DD
    END_DATE = "2025-10-15"    # Format: YYYY-MM-DD
    OUTPUT_DIR = "news_data"

    # Create output directory if it doesn't exist
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    json_filename = f"{OUTPUT_DIR}/news_articles_{timestamp}.json"
    csv_filename = f"{OUTPUT_DIR}/news_articles_{timestamp}.csv"

    print("="*80)
    print("NEWS FETCHER")
    print("="*80)
    print(f"Keywords: {', '.join(KEYWORDS)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Source: Google News RSS")
    print("="*80)

    # Initialize fetcher
    fetcher = NewsFetcher(
        keywords=KEYWORDS,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Fetch articles
    # Set extract_full_content=False for faster fetching (will only get summaries)
    fetcher.fetch_articles(extract_full_content=True)

    # Save results
    if fetcher.articles:
        fetcher.save_to_json(json_filename)
        fetcher.save_to_csv(csv_filename)
        fetcher.print_summary()
    else:
        print("\n⚠ No articles found matching the criteria")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
