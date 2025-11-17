import json
import csv

# Read the JSON file
with open('news_data/content_with_json.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define the CSV columns based on the JSON structure
fieldnames = ['title', 'source', 'url', 'published_date', 'summary', 'content']

# Write to CSV
with open('news_data/content_with_json.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write each article
    for article in data:
        # Create a row with all fields, using empty string for missing fields
        row = {}
        for field in fieldnames:
            value = article.get(field, '')
            # Remove newlines from the content to keep CSV clean
            if isinstance(value, str):
                value = value.replace('\n', ' ').replace('\r', ' ')
            row[field] = value
        writer.writerow(row)

print(f"Successfully converted {len(data)} articles to CSV")
print(f"Output file: news_data/content_with_json.csv")
