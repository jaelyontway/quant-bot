#!/usr/bin/env python3
"""
Smart Time Update Script
Estimates more accurate publication times by:
1. Extracting dates from article titles
2. Using typical financial news publication times based on article type
3. Analyzing article content for time clues
"""

import csv
import re
from datetime import datetime, time as datetime_time


def extract_date_from_title(title: str) -> tuple:
    """
    Extract date from article title.
    Returns (date_string, has_explicit_date)
    """
    # Pattern 1: "Oct. 1, 2025" or "October 1, 2025"
    pattern1 = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})'
    match = re.search(pattern1, title, re.IGNORECASE)
    if match:
        month_abbr = match.group(1)
        day = match.group(2)
        year = match.group(3)
        # Convert month abbreviation to number
        months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                  'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month = months.get(month_abbr[:3].capitalize(), 1)
        return f"{year}-{month:02d}-{int(day):02d}", True

    # Pattern 2: ISO date "2025-10-01"
    pattern2 = r'(\d{4})-(\d{2})-(\d{2})'
    match = re.search(pattern2, title)
    if match:
        return match.group(0), True

    return None, False


def estimate_publication_time(title: str, source: str, base_date: str) -> str:
    """
    Estimate publication time based on article type and source.
    Financial news typically publishes:
    - Market opening articles: 7:00-9:00 AM ET
    - Market closing articles: 4:00-6:00 PM ET
    - After-hours articles: 6:00-10:00 PM ET
    """
    title_lower = title.lower()

    # Check for time indicators in title
    if 'futures' in title_lower or 'stock market opens' in title_lower:
        time_str = '07:30:00'  # Pre-market futures
    elif 'close' in title_lower or 'closes' in title_lower or 'closed' in title_lower:
        time_str = '16:30:00'  # Market close time (4:30 PM ET)
    elif 'after hours' in title_lower or 'after-hours' in title_lower:
        time_str = '18:00:00'  # After hours
    elif 'today' in title_lower and 'market' in title_lower:
        # "Stock market today" articles usually publish throughout the day
        time_str = '14:00:00'  # Mid-afternoon
    elif 'news' in title_lower and any(word in title_lower for word in ['oct', 'sep', 'nov', 'dec']):
        # Daily market news summaries
        time_str = '17:00:00'  # End of day summary
    else:
        # Default: financial news sites publish most during market hours
        # Use 10:00 AM as a reasonable default
        time_str = '10:00:00'

    return f"{base_date} {time_str}"


def update_times_intelligently(input_csv: str, output_csv: str):
    """
    Update publication times using intelligent estimation.
    """
    print("="*80)
    print("SMART TIME UPDATE - Using Title Analysis & Financial News Patterns")
    print("="*80)

    # Read existing CSV
    articles = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        articles = list(reader)

    print(f"\nLoaded {len(articles)} articles from {input_csv}\n")

    updated_count = 0
    title_date_count = 0

    for idx, article in enumerate(articles, 1):
        title = article['title']
        source = article['source']
        old_date = article['published_date']

        print(f"[{idx}/{len(articles)}] {title[:70]}...")
        print(f"  Current: {old_date}")

        # Try to extract date from title first
        extracted_date, has_explicit_date = extract_date_from_title(title)

        if extracted_date:
            # Use extracted date from title
            base_date = extracted_date
            title_date_count += 1
            print(f"  ✓ Found date in title: {extracted_date}")
        else:
            # Use the date part from current published_date
            if old_date != 'Unknown' and ' ' in old_date:
                base_date = old_date.split(' ')[0]
            else:
                print(f"  ⚠ No date found, keeping: {old_date}")
                continue

        # Estimate publication time based on article type
        new_date = estimate_publication_time(title, source, base_date)

        if new_date != old_date:
            article['published_date'] = new_date
            print(f"  → Updated to: {new_date}")
            updated_count += 1
        else:
            print(f"  → No change: {new_date}")

    # Save updated CSV
    print(f"\n{'='*80}")
    print("SAVING UPDATED CSV")
    print(f"{'='*80}")

    fieldnames = ['title', 'source', 'url', 'published_date', 'summary', 'content']

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)

    print(f"\n✓ Saved to: {output_csv}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total articles: {len(articles)}")
    print(f"✓ Dates extracted from titles: {title_date_count}")
    print(f"✓ Times updated with smart estimates: {updated_count}")
    print(f"\nTime estimation rules used:")
    print(f"  - Futures/pre-market: 07:30 AM")
    print(f"  - Market close articles: 04:30 PM")
    print(f"  - After hours: 06:00 PM")
    print(f"  - Daily summaries: 05:00 PM")
    print(f"  - General news: 10:00 AM")
    print(f"{'='*80}")


def main():
    input_file = 'news_data/clean_content.csv'
    output_file = 'news_data/clean_content.csv'

    print(f"\nUpdating times in: {input_file}\n")
    update_times_intelligently(input_file, output_file)
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
