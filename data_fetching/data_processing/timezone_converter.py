"""
Timezone Converter Script
Converts timestamps from various timezones to UTC and NY timezone for news data.
"""

import pandas as pd
from datetime import datetime
import pytz
import re
from dateutil import parser
import os


def parse_datetime_with_timezone(date_string):
    """
    Parse a datetime string that may contain timezone information.

    Args:
        date_string: String containing date, time, and possibly timezone info

    Returns:
        tuple: (datetime_object, timezone_string) or (None, None) if parsing fails
    """
    if pd.isna(date_string) or not date_string:
        return None, None

    # Common timezone abbreviations mapping
    timezone_map = {
        'ET': 'America/New_York',
        'EST': 'America/New_York',
        'EDT': 'America/New_York',
        'CT': 'America/Chicago',
        'CST': 'America/Chicago',
        'CDT': 'America/Chicago',
        'MT': 'America/Denver',
        'MST': 'America/Denver',
        'MDT': 'America/Denver',
        'PT': 'America/Los_Angeles',
        'PST': 'America/Los_Angeles',
        'PDT': 'America/Los_Angeles',
        'UTC': 'UTC',
        'GMT': 'GMT'
    }

    # Try to extract timezone from the string
    tz_pattern = r'\b(ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|UTC|GMT)\b'
    tz_match = re.search(tz_pattern, date_string)

    timezone_str = None
    if tz_match:
        tz_abbr = tz_match.group(1)
        timezone_str = timezone_map.get(tz_abbr, 'UTC')
    else:
        # Default to UTC if no timezone found
        timezone_str = 'UTC'

    # Remove common prefixes and clean the string for parsing
    clean_string = re.sub(r'^(Updated|Published|Posted)\s+', '', date_string, flags=re.IGNORECASE)

    # Remove timezone abbreviations for easier parsing
    clean_string = re.sub(r'\b(ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|UTC|GMT)\b', '', clean_string)

    try:
        # Try to parse the datetime
        dt = parser.parse(clean_string, fuzzy=True)
        return dt, timezone_str
    except Exception as e:
        print(f"Error parsing date string '{date_string}': {e}")
        return None, None


def convert_to_utc(dt, source_timezone):
    """
    Convert a naive datetime to UTC.

    Args:
        dt: datetime object (naive)
        source_timezone: Source timezone string (e.g., 'America/New_York')

    Returns:
        datetime: UTC datetime object
    """
    if dt is None:
        return None

    try:
        # Localize the naive datetime to source timezone
        source_tz = pytz.timezone(source_timezone)
        localized_dt = source_tz.localize(dt)

        # Convert to UTC
        utc_dt = localized_dt.astimezone(pytz.UTC)
        return utc_dt
    except Exception as e:
        print(f"Error converting to UTC: {e}")
        return None


def convert_to_ny_timezone(utc_dt):
    """
    Convert UTC datetime to NY timezone.

    Args:
        utc_dt: UTC datetime object

    Returns:
        datetime: NY timezone datetime object
    """
    if utc_dt is None:
        return None

    try:
        ny_tz = pytz.timezone('America/New_York')
        ny_dt = utc_dt.astimezone(ny_tz)
        return ny_dt
    except Exception as e:
        print(f"Error converting to NY timezone: {e}")
        return None


def process_news_csv(input_file, output_file=None):
    """
    Process news CSV file to add UTC and NY timezone columns.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional, defaults to overwriting input)
    """
    print(f"Reading CSV file: {input_file}")

    # Read the CSV file
    df = pd.read_csv(input_file)

    print(f"Processing {len(df)} rows...")

    # Expected column name for the original datetime
    datetime_col = 'actual date and time'

    if datetime_col not in df.columns:
        print(f"Warning: Column '{datetime_col}' not found in CSV!")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Initialize new columns
    utc_datetimes = []
    ny_dates = []
    ny_times = []
    ny_datetimes = []

    # Process each row
    for idx, row in df.iterrows():
        original_dt_str = row[datetime_col]

        # Parse the datetime and timezone
        dt, tz_str = parse_datetime_with_timezone(original_dt_str)

        if dt is None:
            utc_datetimes.append(None)
            ny_dates.append(None)
            ny_times.append(None)
            ny_datetimes.append(None)
            continue

        # Convert to UTC
        utc_dt = convert_to_utc(dt, tz_str)

        # Convert to NY timezone
        ny_dt = convert_to_ny_timezone(utc_dt)

        # Format the outputs
        if utc_dt:
            utc_datetimes.append(utc_dt.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            utc_datetimes.append(None)

        if ny_dt:
            ny_dates.append(ny_dt.strftime('%Y-%m-%d'))
            ny_times.append(ny_dt.strftime('%H:%M:%S'))
            ny_datetimes.append(ny_dt.strftime('%m/%d/%Y %H:%M'))
        else:
            ny_dates.append(None)
            ny_times.append(None)
            ny_datetimes.append(None)

    # Update the dataframe
    df['published_date_utc'] = utc_datetimes
    df['published_date_ny'] = ny_dates
    df['published_time_ny'] = ny_times
    df['published_date_et'] = ny_datetimes

    # Save to output file
    if output_file is None:
        output_file = input_file

    df.to_csv(output_file, index=False)
    print(f"Processed CSV saved to: {output_file}")
    print(f"Successfully converted {len([x for x in utc_datetimes if x is not None])} timestamps")


def main():
    """Main function to run the timezone converter."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert timezones in news CSV files')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)', default=None)

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return

    # Process the CSV file
    process_news_csv(args.input_file, args.output)


if __name__ == '__main__':
    main()
