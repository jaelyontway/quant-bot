import json
import csv
import re

# Read and fix the JSON file
with open('news_data/content_with_json.json', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix triple quotes (Python style) to single quotes (JSON style)
content = content.replace('"""', '"')

# Remove invalid control characters (keep only newlines and tabs for now)
content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', content)

# Parse the fixed JSON
try:
    data = json.loads(content)
    print(f"Successfully loaded {len(data)} articles from JSON")
except json.JSONDecodeError as e:
    print(f"Error: {e}")
    exit(1)

# Define CSV columns
fieldnames = ['title', 'source', 'url', 'published_date', 'summary', 'content']

# Write to CSV
with open('news_data/content_with_json.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write each article
    for article in data:
        row = {}
        for field in fieldnames:
            value = article.get(field, '')
            # Clean the value: remove newlines and extra whitespace
            if isinstance(value, str):
                value = value.replace('\n', ' ').replace('\r', ' ')
                value = re.sub(r'\s+', ' ', value).strip()
            row[field] = value
        writer.writerow(row)

print(f"Successfully converted {len(data)} articles to CSV")
print(f"Output: news_data/content_with_json.csv")
