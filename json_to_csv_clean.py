import json
import csv
import re

# Read the JSON file with error handling
try:
    with open('news_data/content_with_json.json', 'r', encoding='utf-8') as f:
        content = f.read()
        # Remove invalid control characters
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)
        data = json.loads(content)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print("Attempting to fix common JSON issues...")
    # Try to fix common issues
    content = content.replace('\n', ' ').replace('\r', ' ')
    content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
    data = json.loads(content)

# Define the CSV columns
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
            # Clean the value: remove newlines and control characters
            if isinstance(value, str):
                value = value.replace('\n', ' ').replace('\r', ' ')
                value = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', value)
                value = re.sub(r'\s+', ' ', value).strip()
            row[field] = value
        writer.writerow(row)

print(f"Successfully converted {len(data)} articles to CSV")
print(f"Output file: news_data/content_with_json.csv")
