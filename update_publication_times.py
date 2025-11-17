#!/usr/bin/env python3
"""
Update Publication Times Script
Updates the publication times in clean_content.csv by fetching actual times from article pages.
"""

import csv
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from newspaper import Article
from typing import Optional


def get_real_url(google_news_url: str) -> str:
    """
    Follow Google News redirect to get the actual article URL.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.head(google_news_url, headers=headers, allow_redirects=True, timeout=10)
        real_url = response.url
        if real_url != google_news_url:
            print(f"    ↳ Redirected to: {real_url[:80]}...")
        return real_url
    except Exception as e:
        print(f"    ⚠ Could not follow redirect: {str(e)[:50]}")
        return google_news_url


def extract_publish_date_from_page(url: str) -> Optional[datetime]:
    """
    Try to extract the actual publication date/time from the article page.
    Returns datetime object if found, None otherwise.
    """
    print(f"  Fetching: {url[:80]}...")

    # First, follow Google News redirect to get real URL
    real_url = get_real_url(url)

    # Method 1: Try newspaper3k first (fastest and most reliable)
    try:
        article = Article(real_url)
        article.download()
        article.parse()

        if article.publish_date:
            pub_date = article.publish_date
            if pub_date.tzinfo:
                pub_date = pub_date.replace(tzinfo=None)
            print(f"    ✓ Found date via newspaper3k: {pub_date}")
            return pub_date
    except Exception as e:
        print(f"    ⚠ newspaper3k failed: {str(e)[:50]}")

    # Method 2: Try BeautifulSoup with multiple strategies
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(real_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Strategy 1: Look for <time> tags with datetime attribute
        time_tags = soup.find_all('time')
        for tag in time_tags:
            if tag.get('datetime'):
                try:
                    dt_str = tag.get('datetime')
                    # Parse ISO format datetime
                    if 'T' in dt_str:
                        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        dt = dt.replace(tzinfo=None)
                        print(f"    ✓ Found date via <time> tag: {dt}")
                        return dt
                except Exception as e:
                    continue

        # Strategy 2: Look for meta tags with publication date
        meta_patterns = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'article:published_time'}),
            ('meta', {'property': 'og:published_time'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'publishdate'}),
            ('meta', {'itemprop': 'datePublished'}),
            ('meta', {'property': 'article:published'}),
        ]

        for tag_name, attrs in meta_patterns:
            tag = soup.find(tag_name, attrs)
            if tag and tag.get('content'):
                try:
                    dt_str = tag.get('content')
                    # Try to parse ISO format
                    if 'T' in dt_str:
                        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        dt = dt.replace(tzinfo=None)
                        print(f"    ✓ Found date via meta tag {attrs}: {dt}")
                        return dt
                except Exception as e:
                    continue

        print(f"    ✗ Could not extract date from page")
        return None

    except Exception as e:
        print(f"    ✗ Error fetching page: {str(e)[:50]}")
        return None


def update_csv_with_accurate_times(input_csv: str, output_csv: str):
    """
    Read CSV, fetch actual publication times, and save updated CSV.
    """
    print("="*80)
    print("UPDATING PUBLICATION TIMES FROM ARTICLE PAGES")
    print("="*80)

    # Read existing CSV
    articles = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        articles = list(reader)

    print(f"\nLoaded {len(articles)} articles from {input_csv}")
    print("\nFetching actual publication times from article pages...\n")

    updated_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, article in enumerate(articles, 1):
        print(f"\n[{idx}/{len(articles)}] {article['title'][:60]}...")

        old_date = article['published_date']
        url = article['url']

        # Skip if URL is invalid or missing
        if not url or url == 'Unknown':
            print(f"    ⚠ No valid URL, keeping original date: {old_date}")
            skipped_count += 1
            continue

        # Try to fetch actual publication date
        try:
            actual_date = extract_publish_date_from_page(url)

            if actual_date:
                new_date_str = actual_date.strftime('%Y-%m-%d %H:%M:%S')
                article['published_date'] = new_date_str

                if old_date != new_date_str:
                    print(f"    ✓ UPDATED: {old_date} → {new_date_str}")
                    updated_count += 1
                else:
                    print(f"    ✓ Same as before: {new_date_str}")
            else:
                print(f"    ⚠ Could not find date, keeping original: {old_date}")
                failed_count += 1

            # Be polite to servers - wait between requests
            time.sleep(2)

        except Exception as e:
            print(f"    ✗ Error processing article: {str(e)[:60]}")
            print(f"    ⚠ Keeping original date: {old_date}")
            failed_count += 1
            continue

    # Save updated CSV
    print(f"\n{'='*80}")
    print("SAVING UPDATED CSV")
    print(f"{'='*80}")

    fieldnames = ['title', 'source', 'url', 'published_date', 'summary', 'content']

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)

    print(f"\n✓ Saved updated CSV to: {output_csv}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total articles: {len(articles)}")
    print(f"✓ Updated with new times: {updated_count}")
    print(f"⚠ Could not find new time: {failed_count}")
    print(f"⚠ Skipped (no URL): {skipped_count}")
    print(f"{'='*80}")


def main():
    input_file = 'news_data/clean_content.csv'
    output_file = 'news_data/clean_content.csv'  # Overwrite the original

    # Ask for confirmation before overwriting
    print(f"\nThis will update publication times in: {input_file}")
    print(f"The updated file will be saved to: {output_file}")

    update_csv_with_accurate_times(input_file, output_file)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
