#!/usr/bin/env python3
"""
Clean and organize content in content_with_json.json
"""

import json
import re


def clean_content(content):
    """
    Clean and organize article content for better readability.
    """
    if not content or len(content) < 50:
        return content

    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in content.split('\n')]

    # Join lines and ensure proper paragraph breaks
    content = '\n'.join(lines)

    # Ensure double newlines between paragraphs
    content = re.sub(r'\n\n+', '\n\n', content)

    # Remove extra spaces
    content = re.sub(r' +', ' ', content)

    return content.strip()


def main():
    input_file = 'news_data/content_with_json.json'
    output_file = 'news_data/content_with_json_cleaned.json'

    print(f"Reading {input_file}...")

    # Read the file as text first to handle malformed JSON
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to parse as JSON
        # Replace literal newlines in content with \n
        data = json.loads(content)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Attempting to fix malformed JSON...")

        # Read line by line and reconstruct
        import ast
        with open(input_file, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # Try using ast.literal_eval or manual parsing
        # For now, let's try a different approach
        try:
            # Load with a more permissive decoder
            data = json.loads(file_content, strict=False)
        except:
            print("Could not parse JSON. Using manual reconstruction...")
            # Manual reconstruction would go here
            return

    print(f"Found {len(data)} articles")

    # Clean content for each article
    cleaned_count = 0
    for i, article in enumerate(data):
        if 'content' in article and article['content']:
            original_length = len(article['content'])
            article['content'] = clean_content(article['content'])
            cleaned_count += 1

            if i < 3:  # Show progress for first 3 articles
                print(f"\nArticle {i+1}: {article['title'][:60]}...")
                print(f"  Original length: {original_length} chars")
                print(f"  Cleaned length: {len(article['content'])} chars")

    print(f"\n✓ Cleaned content in {cleaned_count} articles")

    # Save cleaned data
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved cleaned JSON to {output_file}")

    # Also update the original file
    print(f"\nUpdating original file {input_file}...")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Updated {input_file}")


if __name__ == "__main__":
    main()
