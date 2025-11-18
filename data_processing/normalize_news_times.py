#!/usr/bin/env python3
"""Normalize news timestamps to UTC and New York time.

Reads a CSV that matches the training-data layout
(title, source, url, actual date and time, published_date_utc, ...),
parses the human-readable timestamp (e.g. "Updated 03:00 AM ET 10/13/2025"),
and writes normalized UTC / NY date fields back to the file.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
TIME_PATTERN = re.compile(
    r"(?P<time>\d{1,2}:\d{2}\s?(?:AM|PM)?)\s*(?P<tz>UTC|ET|EDT|EST)?\s*(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE,
)


@dataclass
class ParsedTimestamp:
    original: str
    localized: datetime


def _clean_input(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML anchors, etc.
    text = text.replace("Updated", "").strip()
    return re.sub(r"\s+", " ", text)


def _parse_actual_datetime(raw: str) -> Optional[ParsedTimestamp]:
    if not raw or str(raw).strip().lower() == "unknown":
        return None

    text = _clean_input(str(raw))
    match = TIME_PATTERN.search(text)
    if not match:
        return None

    time_part = match.group("time")
    tz_part = (match.group("tz") or "ET").upper()
    date_part = match.group("date").replace("-", "/")
    try:
        dt = datetime.strptime(f"{date_part} {time_part.upper()}", "%m/%d/%Y %I:%M %p")
    except ValueError:
        return None

    if tz_part == "UTC":
        tz = UTC_TZ
    else:  # treat ET/EST/EDT as America/New_York and let ZoneInfo handle DST
        tz = NY_TZ
    dt = dt.replace(tzinfo=tz)
    return ParsedTimestamp(original=text, localized=dt)


def _format_outputs(parsed: ParsedTimestamp):
    """Return (utc_str, ny_date_str, ny_time_str, et_display_str)."""
    utc_dt = parsed.localized.astimezone(UTC_TZ)
    ny_dt = utc_dt.astimezone(NY_TZ)
    return (
        utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
        ny_dt.strftime("%Y-%m-%d"),
        ny_dt.strftime("%H:%M:%S"),
        ny_dt.strftime("%m/%d/%Y %H:%M"),
    )


def normalize_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    df = pd.read_csv(input_path)
    if "actual date and time" not in df.columns:
        raise ValueError("Input file is missing 'actual date and time' column.")

    updated_rows = 0
    for idx, value in df["actual date and time"].items():
        parsed = _parse_actual_datetime(value)
        if not parsed:
            continue

        utc_str, ny_date, ny_time, et_display = _format_outputs(parsed)
        df.at[idx, "published_date_utc"] = utc_str
        df.at[idx, "published_date_ny"] = ny_date
        df.at[idx, "published_time_ny"] = ny_time
        df.at[idx, "published_date_et"] = et_display
        updated_rows += 1

    destination = output_path or input_path
    df.to_csv(destination, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Updated {updated_rows} rows in {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize human-readable news timestamps into UTC and New York time columns."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the news CSV (e.g., data/demo_data/NVDA_2025-10-13/news_2025-10-13.csv)",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. If omitted, the file is updated in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    normalize_file(input_path, output_path)


if __name__ == "__main__":
    main()
