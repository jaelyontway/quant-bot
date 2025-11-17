import csv
import re
import ast

# Read the file
with open('news_data/content_with_json.json', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix triple quotes
content = content.replace('"""', '"')

# Try to manually parse since the JSON has actual newlines in strings
# which is technically invalid JSON but we can work around it
try:
    # Use ast.literal_eval as a fallback for Python-style data
    data = ast.literal_eval(content)
    print(f"Loaded {len(data)} articles using ast.literal_eval")
except:
    print("ast.literal_eval failed, trying manual JSON fix...")

    # More aggressive fix: escape unescaped newlines within string values
    # This is a complex regex that tries to find newlines within quoted strings
    import json

    # Remove control characters except newlines and tabs
    fixed_content = ''
    in_string = False
    escape_next = False

    for i, char in enumerate(content):
        if escape_next:
            fixed_content += char
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            fixed_content += char
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            fixed_content += char
            continue

        # If we're inside a string and hit a newline, escape it
        if in_string and char == '\n':
            fixed_content += '\\n'
        elif in_string and char == '\r':
            fixed_content += '\\r'
        elif in_string and char == '\t':
            fixed_content += '\\t'
        else:
            fixed_content += char

    try:
        data = json.loads(fixed_content)
        print(f"Loaded {len(data)} articles after manual fix")
    except json.JSONDecodeError as e:
        print(f"Still failed: {e}")
        # Last resort: save the fixed content for inspection
        with open('news_data/fixed_temp.json', 'w') as f:
            f.write(fixed_content)
        print("Saved attempted fix to news_data/fixed_temp.json for inspection")
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
