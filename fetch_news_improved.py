#!/usr/bin/env python3
"""
Improved News Fetcher Script
Attempts to extract more accurate publication times from article pages.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import pandas as pd
from urllib.parse import quote
import time
from typing import List, Dict
import re
from newspaper import Article


class ImprovedNewsFetcher:
    def __init__(self, keywords: List[str], start_date: str, end_date: str):
        self.keywords = keywords
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.articles = []

    def build_google_news_url(self) -> str:
        """Build Google News RSS URL with keywords."""
        query = ' '.join(self.keywords)
        encoded_query = quote(query)
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

    def extract_publish_date_from_page(self, soup: BeautifulSoup, url: str) -> datetime:
        """
        Try to extract the actual publication date/time from the article page.
        Tries multiple methods to find the most accurate timestamp.
        """
        # Method 1: Look for <time> tags (most reliable)
        time_tags = soup.find_all('time')
        for tag in time_tags:
            if tag.get('datetime'):
                try:
                    dt_str = tag.get('datetime')
                    # Parse ISO format datetime
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    return dt.replace(tzinfo=None)
                except:
                    pass

        # Method 2: Look for meta tags
        meta_patterns = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'article:published_time'}),
            ('meta', {'property': 'og:published_time'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'publishdate'}),
            ('meta', {'itemprop': 'datePublished'}),
        ]

        for tag_name, attrs in meta_patterns:
            tag = soup.find(tag_name, attrs)
            if tag and tag.get('content'):
                try:
                    dt_str = tag.get('content')
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    return dt.replace(tzinfo=None)
                except:
                    pass

        # Method 3: Use newspaper3k's published date if available
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.publish_date:
                return article.publish_date.replace(tzinfo=None) if article.publish_date.tzinfo else article.publish_date
        except:
            pass

        return None

    def extract_article_content(self, url: str) -> tuple:
        """
        Extract full article content and publication date from URL.
        Returns: (content_text, actual_publish_datetime)
        """
        content = ""
        actual_date = None

        # Try newspaper3k first
        try:
            article = Article(url)
            article.download()
            article.parse()

            if article.text and len(article.text) > 100:
                content = article.text
                actual_date = article.publish_date
                if actual_date and actual_date.tzinfo:
                    actual_date = actual_date.replace(tzinfo=None)
                print(f"    ✓ Extracted {len(content)} chars with newspaper3k")
                if actual_date:
                    print(f"    ✓ Found publish date: {actual_date}")
                return content, actual_date
        except Exception as e:
            print(f"    ⚠ newspaper3k failed, trying fallback...")

        # Fallback to BeautifulSoup
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to extract actual publication date
            actual_date = self.extract_publish_date_from_page(soup, url)
            if actual_date:
                print(f"    ✓ Found publish date from page: {actual_date}")

            # Extract content
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()

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
                if selector in ['article', 'main']:
                    article_content = soup.find(selector)
                else:
                    article_content = soup.select_one(selector)
                if article_content:
                    break

            if article_content:
                text = article_content.get_text(separator='\n', strip=True)
                text = re.sub(r'\n\s*\n', '\n\n', text)
                if len(text) > 100:
                    content = text
                    print(f"    ✓ Extracted {len(content)} chars with BeautifulSoup")
                    return content, actual_date

            # Last resort: paragraphs
            paragraphs = soup.find_all('p')
            text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            if text and len(text) > 100:
                content = text
                print(f"    ✓ Extracted {len(content)} chars from paragraphs")
                return content, actual_date

            return "Content extraction failed", actual_date

        except Exception as e:
            print(f"    ✗ All methods failed: {str(e)[:80]}")
            return f"Error: {str(e)}", None

    def parse_article_date(self, date_string: str) -> datetime:
        """Parse article published date from RSS feed."""
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_string)
            return dt.replace(tzinfo=None) if dt else None
        except:
            return None

    def fetch_articles(self, extract_full_content: bool = True):
        """Fetch articles from Google News RSS."""
        feed = self.fetch_rss_feed()

        if not feed or not feed.entries:
            print("No articles found")
            return

        for idx, entry in enumerate(feed.entries):
            print(f"\nProcessing article {idx + 1}/{len(feed.entries)}: {entry.title}")

            # Parse RSS feed date
            rss_date = self.parse_article_date(entry.published) if hasattr(entry, 'published') else None

            # Extract article information
            article = {
                'title': entry.title,
                'source': entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Unknown',
                'url': entry.link,
                'rss_published_date': rss_date.strftime('%Y-%m-%d %H:%M:%S') if rss_date else 'Unknown',
                'summary': entry.summary if hasattr(entry, 'summary') else '',
                'content': ''
            }

            # Extract full content and try to get actual publish date
            if extract_full_content:
                print(f"  Extracting full content and publish date...")
                content, actual_date = self.extract_article_content(entry.link)
                article['content'] = content

                # Use actual date if found, otherwise fall back to RSS date
                if actual_date:
                    article['published_date'] = actual_date.strftime('%Y-%m-%d %H:%M:%S')
                    article['date_source'] = 'article_page'
                elif rss_date:
                    article['published_date'] = article['rss_published_date']
                    article['date_source'] = 'rss_feed'
                else:
                    article['published_date'] = 'Unknown'
                    article['date_source'] = 'unknown'

                time.sleep(1)  # Be polite to servers
            else:
                article['content'] = article['summary']
                article['published_date'] = article['rss_published_date']
                article['date_source'] = 'rss_feed'

            self.articles.append(article)
            print(f"  ✓ Added article (date from: {article['date_source']})")

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
        column_order = ['title', 'source', 'published_date', 'date_source', 'rss_published_date', 'url', 'summary', 'content']
        # Only include columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Saved to {filename}")

    def print_summary(self):
        """Print summary of fetched articles."""
        print(f"\n{'='*80}")
        print(f"SUMMARY: Fetched {len(self.articles)} articles")
        print(f"{'='*80}")

        date_sources = {}
        for article in self.articles:
            source = article.get('date_source', 'unknown')
            date_sources[source] = date_sources.get(source, 0) + 1

        print(f"\nDate source statistics:")
        for source, count in date_sources.items():
            print(f"  {source}: {count} articles")

        for idx, article in enumerate(self.articles, 1):
            print(f"\n{idx}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Date: {article['published_date']} (from: {article.get('date_source', 'unknown')})")
            print(f"   URL: {article['url']}")
            print(f"   Content length: {len(article['content'])} characters")


def main():
    """Main function to run the improved news fetcher."""

    # Configuration
    KEYWORDS = ["government shutdown", "nvidia stock", "US"]
    START_DATE = "2025-10-01"
    END_DATE = "2025-10-15"
    OUTPUT_DIR = "news_data"

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    json_filename = f"{OUTPUT_DIR}/news_improved_{timestamp}.json"
    csv_filename = f"{OUTPUT_DIR}/news_improved_{timestamp}.csv"

    print("="*80)
    print("IMPROVED NEWS FETCHER (with accurate publication times)")
    print("="*80)
    print(f"Keywords: {', '.join(KEYWORDS)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Source: Google News RSS + Article Pages")
    print("="*80)

    fetcher = ImprovedNewsFetcher(
        keywords=KEYWORDS,
        start_date=START_DATE,
        end_date=END_DATE
    )

    fetcher.fetch_articles(extract_full_content=True)

    if fetcher.articles:
        fetcher.save_to_json(json_filename)
        fetcher.save_to_csv(csv_filename)
        fetcher.print_summary()
    else:
        print("\n⚠ No articles found matching the criteria")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
