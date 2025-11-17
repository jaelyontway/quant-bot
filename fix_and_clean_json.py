#!/usr/bin/env python3
"""
Fix malformed JSON and clean content
"""

import json
import re


def clean_content(content):
    """Clean and organize article content."""
    if not content or len(content) < 50:
        return content

    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Join lines with proper paragraph breaks
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(line)

    content = '\n\n'.join(cleaned_lines)

    # Remove extra spaces within lines
    content = re.sub(r' +', ' ', content)

    return content.strip()


def fix_json_file(input_file):
    """Read and fix malformed JSON file."""
    print(f"Reading {input_file}...")

    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    print(f"Read {len(lines)} lines")

    # Reconstruct JSON properly
    content = ''.join(lines)

    # Try to load with error handling
    articles = []
    try:
        # Attempt to parse
        articles = json.loads(content)
        print(f"✓ Successfully parsed JSON with {len(articles)} articles")
        return articles
    except json.JSONDecodeError as e:
        print(f"JSON error at position {e.pos}: {e.msg}")
        print(f"Problem area: {content[max(0, e.pos-50):e.pos+50]}")

        # Try to fix common issues
        # Replace unescaped newlines within strings
        print("\nAttempting to fix JSON structure...")

        # This is complex - let's use a simpler approach
        # Parse manually article by article
        return None


def main():
    input_file = 'news_data/content_with_json.json'

    articles = fix_json_file(input_file)

    if articles is None:
        print("\n⚠ Could not automatically fix JSON.")
        print("Checking file structure...")

        # Show sample of problematic content
        with open(input_file, 'r', encoding='utf-8') as f:
            sample = f.read(3000)
            print("\nFirst 3000 characters:")
            print(sample)
        return

    # Clean content
    print(f"\nCleaning content in {len(articles)} articles...")
    cleaned_count = 0

    for i, article in enumerate(articles):
        if 'content' in article and article['content']:
            original_length = len(article['content'])
            article['content'] = clean_content(article['content'])
            cleaned_count += 1

            if i < 2:
                print(f"\n✓ Article {i+1}: {article['title'][:60]}...")
                print(f"  Cleaned: {original_length} → {len(article['content'])} chars")

    print(f"\n✓ Cleaned content in {cleaned_count}/{len(articles)} articles")

    # Save
    output_file = input_file
    print(f"\nSaving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved!")

    # Show sample of cleaned content
    if articles and articles[0].get('content'):
        print("\nSample cleaned content:")
        print("=" * 80)
        print(articles[0]['content'][:500])
        print("=" * 80)


if __name__ == "__main__":
    main()
