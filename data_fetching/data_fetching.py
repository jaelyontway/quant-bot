#!/usr/bin/env python3
"""Utility script to link AI-related news to NVDA intraday price reactions."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
import sys

import pandas as pd
import requests
from zoneinfo import ZoneInfo

# Give the script access to repo-level modules when executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from fetch_news import NewsFetcher
except Exception as exc:  # pragma: no cover - optional dependency
    NewsFetcher = None
    _NEWS_FETCH_IMPORT_ERROR = exc
else:  # pragma: no cover - simple assignment
    _NEWS_FETCH_IMPORT_ERROR = None

# ----- Configuration -------------------------------------------------------
# Default values + config loading/merging live here so library imports and
# CLI executions rely on the same centralized settings and API credentials.

DEFAULT_NEWS_DOMAINS = [
    # "wsj.com",
    # "ft.com",
    # "reuters.com",
    # "bloomberg.com",
    # "nytimes.com",
    # "cnbc.com",
    # "marketwatch.com",
    # "barrons.com",
    # "economist.com",
    # "forbes.com",
    # "businessinsider.com",
    # "goldmansachs.com",
]

DEFAULT_STOCK_API_URL = "https://api.polygon.io/v2/aggs/ticker/NVDA/range/60/minute"
DEFAULT_NEWS_TIMEZONE = "America/New_York"
DEFAULT_MARKET_TIMEZONE = "America/New_York"
DEFAULT_PRICE_HOURS_AFTER = 72
DEFAULT_INTERVAL_MINUTES = 60


# ----- Data containers -----------------------------------------------------
# Dataclasses ensure news and price rows have a predictable schema everywhere.

@dataclass
class AppConfig:
    """Configuration container loaded from YAML/env/defaults."""

    ticker: str
    stock_api_url: str
    stock_api_key: str
    news_domains: list[str]
    default_news_timezone: str
    market_timezone: str
    default_price_hours_after: int
    default_interval_minutes: int
    default_news_date: str | None
    default_news_end_date: str | None
    default_price_date: str | None
    default_keywords: list[str] | None
    default_price_trading_days: int


def _build_default_config() -> AppConfig:
    """Return default config using environment variables when available."""

    return AppConfig(
        ticker="NVDA",
        stock_api_url=DEFAULT_STOCK_API_URL,
        stock_api_key=os.getenv("STOCK_API_KEY", "YOUR_STOCK_API_KEY"),
        news_domains=list(DEFAULT_NEWS_DOMAINS),
        default_news_timezone=DEFAULT_NEWS_TIMEZONE,
        market_timezone=DEFAULT_MARKET_TIMEZONE,
        default_price_hours_after=DEFAULT_PRICE_HOURS_AFTER,
        default_interval_minutes=DEFAULT_INTERVAL_MINUTES,
        default_news_date=None,
        default_news_end_date=None,
        default_price_date=None,
        default_keywords=None,
        default_price_trading_days=3,
    )


# Load default configuration immediately; CLI args can override it later.
CONFIG = _build_default_config()


def load_config_from_file(path: str | None) -> AppConfig:
    """Load configuration overrides from a YAML file."""

    base = _build_default_config()

    if not path or not os.path.exists(path):
        return base

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "PyYAML is required to load config files. Install it via `pip install pyyaml`."
        ) from exc

    with open(path, "r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}

    return _merge_config(base, overrides)


def _merge_config(base: AppConfig, overrides: dict) -> AppConfig:
    """Merge user overrides onto the base configuration."""

    cfg = asdict(base)

    def _maybe_copy(target_key: str, source_key: str | None = None):
        key = source_key or target_key
        if key in overrides:
            cfg[target_key] = overrides[key]

    _maybe_copy("ticker")
    _maybe_copy("stock_api_url")
    _maybe_copy("stock_api_key")
    _maybe_copy("news_domains")
    _maybe_copy("default_news_timezone")
    _maybe_copy("market_timezone")
    _maybe_copy("default_interval_minutes")

    apis = overrides.get("apis", {})
    stocks_api = apis.get("stocks", {})
    cfg["stock_api_url"] = stocks_api.get("url", cfg["stock_api_url"])
    cfg["stock_api_key"] = stocks_api.get("key", cfg["stock_api_key"])

    defaults_section = overrides.get("defaults", {})
    cfg["default_news_timezone"] = defaults_section.get(
        "news_timezone", cfg["default_news_timezone"]
    )
    cfg["default_price_hours_after"] = defaults_section.get(
        "price_hours_after", defaults_section.get("hours_after", cfg["default_price_hours_after"])
    )
    cfg["default_interval_minutes"] = defaults_section.get(
        "interval_minutes", cfg["default_interval_minutes"]
    )
    cfg["default_news_date"] = defaults_section.get(
        "news_date", defaults_section.get("date", cfg["default_news_date"])
    )
    cfg["default_news_end_date"] = defaults_section.get("news_end_date", cfg["default_news_end_date"])
    cfg["default_price_date"] = defaults_section.get(
        "price_date", defaults_section.get("date", cfg["default_price_date"])
    )
    if "keywords" in defaults_section:
        cfg["default_keywords"] = defaults_section["keywords"]
    cfg["default_price_trading_days"] = defaults_section.get(
        "price_trading_days",
        defaults_section.get("trading_days", cfg["default_price_trading_days"]),
    )

    news_domains = overrides.get("news_domains")
    if news_domains:
        cfg["news_domains"] = news_domains

    return AppConfig(**cfg)


@dataclass
class Article:
    """Container for standardized news article fields."""

    published_at: str
    source_domain: str
    title: str
    description: str
    url: str
    keyword: str


@dataclass
class PriceBar:
    """Container for standardized intraday price bars."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


# ----- News fetching helpers -----------------------------------------------
# These helpers encapsulate grabbing Google News articles, normalizing them,
# filtering them by keyword/domain, and returning pandas-friendly structures.

def fetch_news_for_date(
    target_date: str,
    keywords: Sequence[str],
    timezone: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fetch AI-related news for a given date using the Google News scraper."""

    tz_name = timezone or CONFIG.default_news_timezone
    tz = ZoneInfo(tz_name)
    domain_counts: dict[str, int] = {domain: 0 for domain in CONFIG.news_domains}

    try:
        payload = _fetch_articles_from_google_news(keywords, target_date)
    except Exception as exc:
        logging.error("Failed to fetch Google News results for %s: %s", target_date, exc)
        empty_columns = [
            "title",
            "source",
            "url",
            "actual date and time",
            "published_date_utc",
            "published_date_ny",
            "published_time_ny",
            "published_date_et",
            "summary",
            "content",
            "news_date",
        ]
        empty_df = pd.DataFrame(columns=empty_columns)
        return empty_df, domain_counts

    rows: list[dict] = []
    for raw_article in payload or []:
        url = raw_article.get("url", "")
        if not url:
            continue
        if not _article_contains_keywords(raw_article, keywords):
            continue

        domain = _extract_domain(url) or (raw_article.get("source") or "")
        if domain and domain not in domain_counts:
            domain_counts[domain] = 0

        description = (
            raw_article.get("summary")
            or raw_article.get("description")
            or raw_article.get("content")
            or ""
        )

        published_utc = raw_article.get("published_date_utc")
        published_et = raw_article.get("published_date_et")
        published_ny = raw_article.get("published_date_ny")
        actual_dt = raw_article.get("actual date and time")
        published_value = published_utc or published_et or published_ny or actual_dt

        row = {
            "title": raw_article.get("title", ""),
            "source": raw_article.get("source", domain or "Unknown"),
            "url": url,
            "actual date and time": actual_dt or "Unknown",
            "published_date_utc": published_utc or "",
            "published_date_ny": published_ny or "",
            "published_time_ny": raw_article.get("published_time_ny", ""),
            "published_date_et": published_et or "",
            "summary": description,
            "content": raw_article.get("content", ""),
            "news_date": target_date,
        }
        rows.append(row)
        if domain:
            domain_counts[domain] += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df, domain_counts


def _fetch_articles_from_google_news(
    keywords: Sequence[str],
    target_date: str,
) -> list[dict]:
    """Invoke the shared RSS scraper for a single-day window."""

    if NewsFetcher is None:
        msg = (
            "fetch_news.NewsFetcher is unavailable. Ensure fetch_news.py and its "
            "dependencies (feedparser, newspaper3k, etc.) are installed."
        )
        raise RuntimeError(msg) from _NEWS_FETCH_IMPORT_ERROR

    fetcher = NewsFetcher(list(keywords), start_date=target_date, end_date=target_date)
    fetcher.fetch_articles(extract_full_content=False)
    return fetcher.articles


def _normalize_article_timestamp(value: str | None, tz: ZoneInfo) -> str:
    """Convert the NewsFetcher timestamp into an ISO-8601 string."""

    if not value or str(value).lower() == "unknown":
        return ""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return str(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts.isoformat()


def _article_contains_keywords(article: dict, keywords: Sequence[str]) -> bool:
    """Return True when the article text contains at least one keyword."""

    haystack = " ".join(
        filter(
            None,
            [
                article.get("title", ""),
                article.get("description", ""),
                article.get("summary", ""),
                article.get("content", ""),
            ],
        )
    ).lower()
    for keyword in keywords:
        if keyword.lower() in haystack:
            return True
    return not keywords


def _extract_domain(url: str) -> str:
    """Extract netloc from a URL without subpaths."""

    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "") if parsed.netloc else ""


# ----- Price fetching helpers ----------------------------------------------
# Intraday price retrieval lives here: convert anchors to market time, query
# Polygon, and post-process the bars so downstream notebooks can compare them
# against the news events gathered above.

def fetch_nvda_prices_around_event(
    event_time: pd.Timestamp,
    hours_after: int = 72,
    interval_minutes: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Fetch NVDA intraday bars from the event time for the next `hours_after` hours."""

    # --- Convert event timestamp into the market timezone and window -------

    market_tz = ZoneInfo(CONFIG.market_timezone)
    event_time_local = _ensure_timezone(event_time, market_tz)
    end_time_local = event_time_local + pd.Timedelta(hours=hours_after)

    raw_bars, query_meta = _query_stock_api(event_time_local, end_time_local, interval_minutes)
    # --- Transform Polygon results into an analyzable DataFrame ------------
    records: list[PriceBar] = []
    for bar in raw_bars:
        records.append(
            PriceBar(
                timestamp=bar.get("timestamp") or bar.get("t"),
                open=bar.get("open") or bar.get("o"),
                high=bar.get("high") or bar.get("h"),
                low=bar.get("low") or bar.get("l"),
                close=bar.get("close") or bar.get("c"),
                volume=bar.get("volume") or bar.get("v"),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return df, query_meta

    df["timestamp (UTC)"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop(columns=["timestamp"])
    df = df.sort_values("timestamp (UTC)").reset_index(drop=True)

    # Restrict to regular trading hours in market timezone
    market_tz = ZoneInfo(CONFIG.market_timezone)
    local_ts = df["timestamp (UTC)"].dt.tz_convert(market_tz)
    df["timestamp (America/New_York)"] = local_ts.dt.tz_localize(None)
    df["weekday (America/New_York)"] = df["timestamp (America/New_York)"].dt.day_name()
    start_time = time(9, 30)
    end_time = time(16, 0)
    in_regular_hours = local_ts.dt.time.between(start_time, end_time, inclusive="both")
    df = df[in_regular_hours].reset_index(drop=True)
    if df.empty:
        query_meta["note"] = "No bars within regular trading hours"
        return df, query_meta

    first_close = df["close"].iloc[0]
    df["pct_change_from_event"] = (df["close"] / first_close) - 1
    df["log_return"] = df["close"].apply(math.log).diff()

    cols = [
        "timestamp (UTC)",
        "timestamp (America/New_York)",
        "weekday (America/New_York)",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "pct_change_from_event",
        "log_return",
    ]
    df = df[cols]

    query_meta = {
        **query_meta,
        "bars_returned": len(df),
        "first_timestamp": df["timestamp (UTC)"].iloc[0].isoformat(),
        "last_timestamp": df["timestamp (UTC)"].iloc[-1].isoformat(),
    }
    return df, query_meta


def _query_stock_api(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, interval_minutes: int
) -> tuple[list[dict], dict]:
    """Fetch intraday data from Polygon."""

    if interval_minutes % 60 == 0:
        multiplier = interval_minutes // 60
        timespan = "hour"
    else:
        multiplier = interval_minutes
        timespan = "minute"

    start_date = start_ts.date().isoformat()
    end_date = end_ts.date().isoformat()

    url = (
        f"{CONFIG.stock_api_url}/"
        f"{CONFIG.ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": CONFIG.stock_api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    bars = data.get("results", []) or []
    if not bars:
        metadata = {
            "request_path": url,
            "status_code": resp.status_code,
            "result_count": 0,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "interval_minutes": interval_minutes,
        }
        return [], metadata

    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    start_utc = start_ts.tz_convert("UTC")
    end_utc = end_ts.tz_convert("UTC")
    df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]

    records = df[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    metadata = {
        "request_path": url,
        "status_code": resp.status_code,
        "result_count": len(records),
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "interval_minutes": interval_minutes,
    }
    return records, metadata


def _ensure_timezone(ts: pd.Timestamp, tz: ZoneInfo) -> pd.Timestamp:
    """Force a pandas Timestamp to have the provided timezone."""

    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def _collect_trading_dates(start_date: str, count: int) -> list[pd.Timestamp]:
    """Return a list of consecutive weekday dates starting from `start_date`."""

    current = pd.Timestamp(start_date).normalize()
    dates: list[pd.Timestamp] = []
    while len(dates) < count:
        if current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current)
        current += pd.Timedelta(days=1)
    return dates


def _expand_date_range(start_date: str, end_date: str | None) -> list[str]:
    """Return an inclusive list of YYYY-MM-DD strings between start and end."""

    if not start_date:
        raise ValueError("start date is required")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date or start_date).normalize()
    if end < start:
        raise ValueError("end date cannot be before start date")
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def _format_date_label(dates: Sequence[str]) -> str:
    """Return a readable label for a list of date strings."""

    if not dates:
        return "unspecified"
    unique = sorted(set(dates))
    if len(unique) == 1:
        return unique[0]
    return f"{unique[0]}_to_{unique[-1]}"


def fetch_prices_from_day(
    target_date: str,
    interval_minutes: int,
    *,
    trading_days: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch NVDA prices covering `trading_days` full sessions starting from `target_date`."""

    market_tz = ZoneInfo(CONFIG.market_timezone)
    target_date_str = str(target_date)
    trading_days = trading_days if trading_days is not None else CONFIG.default_price_trading_days
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer when requesting full sessions")
    trading_dates = _collect_trading_dates(target_date_str, trading_days)
    first_day = trading_dates[0]
    last_day = trading_dates[-1]
    anchor_ts = first_day.tz_localize(market_tz) + pd.Timedelta(hours=9, minutes=30)
    close_ts = last_day.tz_localize(market_tz) + pd.Timedelta(hours=16)
    hours_span = math.ceil((close_ts - anchor_ts) / pd.Timedelta(hours=1))

    prices, metadata = fetch_nvda_prices_around_event(anchor_ts, hours_span, interval_minutes)
    metadata = {
        **metadata,
        "anchor_timestamp": anchor_ts.isoformat(),
        "anchor_reason": "start_of_day_regular_hours",
        "target_date": target_date_str,
        "trading_days_requested": trading_days,
        "trading_days_actual": [d.date().isoformat() for d in trading_dates],
        "regular_hours": {"start": "09:30", "end": "16:00", "timezone": CONFIG.market_timezone},
    }
    return prices, metadata


# ----- Command-line interface ----------------------------------------------
# The CLI resolves configuration, orchestrates the news/price fetch steps,
# and writes CSV/JSON artifacts that subsequent analysis scripts consume.

def main() -> None:
    parser = argparse.ArgumentParser(description="Link AI news to NVDA intraday prices.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to a YAML config file (defaults baked in if missing)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="(Deprecated) alias for --news-date; kept for backward compatibility",
    )
    parser.add_argument(
        "--news-date",
        default=None,
        help="Start date for Google News scraping (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--news-end-date",
        default=None,
        help="Optional end date for Google News scraping (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Keywords to search for (space-separated)",
    )
    parser.add_argument(
        "--price-date",
        default=None,
        help="Start date for NVDA price fetch (YYYY-MM-DD). Defaults to news date.",
    )
    parser.add_argument(
        "--hours-after",
        type=int,
        default=None,
        help="(Deprecated) alias for --price-hours-after; kept for backward compatibility",
    )
    parser.add_argument(
        "--price-hours-after",
        type=int,
        default=None,
        help="Number of hours of NVDA prices to fetch after the price anchor time",
    )
    parser.add_argument(
        "--price-trading-days",
        type=int,
        default=None,
        help="Number of full trading sessions to fetch (set to 0 to disable this mode)",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=None,
        help="Intraday interval (in minutes) for NVDA price bars",
    )
    parser.add_argument(
        "--news-timezone",
        default=None,
        help="Timezone for interpreting the news date window",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip Google News scraping and only fetch NVDA prices",
    )
    args = parser.parse_args()

    global CONFIG
    CONFIG = load_config_from_file(args.config)

    # --- Resolve runtime options, allowing CLI args to override config -----
    news_timezone = args.news_timezone or CONFIG.default_news_timezone
    interval_minutes = args.interval_minutes or CONFIG.default_interval_minutes
    news_date_value = args.news_date or args.date or CONFIG.default_news_date
    news_end_date_value = args.news_end_date or CONFIG.default_news_end_date or news_date_value
    price_date_value = args.price_date or CONFIG.default_price_date or news_date_value
    price_hours_after = (
        args.price_hours_after
        if args.price_hours_after is not None
        else args.hours_after
        if args.hours_after is not None
        else CONFIG.default_price_hours_after
    )
    price_trading_days = (
        args.price_trading_days
        if args.price_trading_days is not None
        else CONFIG.default_price_trading_days
    )
    keywords = args.keywords or CONFIG.default_keywords

    if not args.skip_news and not news_date_value:
        raise SystemExit(
            "No news date provided. Supply --news-date or set defaults.news_date in the config file."
        )
    if not keywords:
        raise SystemExit(
            "No keywords provided. Supply --keywords or set defaults.keywords in the config file."
        )
    if not price_date_value:
        raise SystemExit(
            "No price date provided. Supply --price-date or set defaults.price_date/news_date in the config file."
        )
    if price_trading_days is None:
        price_trading_days = CONFIG.default_price_trading_days
    if price_trading_days < 0:
        raise SystemExit("--price-trading-days cannot be negative.")
    if price_trading_days == 0 and (price_hours_after is None or price_hours_after <= 0):
        raise SystemExit(
            "Provide --price-hours-after (or defaults.price_hours_after) when --price-trading-days is 0."
        )

    # --- Step 1: fetch news ------------------------------------------------
    if args.skip_news:
        news_dates = [news_date_value] if news_date_value else []
        news_label = _format_date_label(news_dates) if news_dates else (price_date_value or "unspecified")
    else:
        try:
            news_dates = _expand_date_range(str(news_date_value), str(news_end_date_value))
        except ValueError as exc:
            raise SystemExit(f"Invalid news date range: {exc}") from exc
        news_label = _format_date_label(news_dates)

    price_label = str(price_date_value)
    base_output_dir = Path("data/demo_data")
    output_dir = base_output_dir / f"{CONFIG.ticker}_{price_label or news_label}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing outputs to %s", output_dir)

    output_news_columns = [
        "title",
        "source",
        "url",
        "actual date and time",
        "published_date_utc",
        "published_date_ny",
        "published_time_ny",
        "published_date_et",
        "summary",
        "content",
    ]
    news_columns = output_news_columns + ["news_date"]
    news_csv = output_dir / f"news_{news_label}.csv"
    if args.skip_news:
        logging.info("Skipping news fetch (skip-news flag set)")
        news_df = pd.DataFrame(columns=news_columns)
        domain_counts = {domain: 0 for domain in CONFIG.news_domains}
        news_df.to_csv(news_csv, index=False, columns=output_news_columns)
        print(f"Skipped news fetch; wrote empty placeholder to {news_csv}")
    else:
        domain_counts = {domain: 0 for domain in CONFIG.news_domains}
        frames: list[pd.DataFrame] = []
        for single_date in news_dates:
            daily_df, daily_counts = fetch_news_for_date(single_date, keywords, news_timezone)
            if not daily_df.empty:
                daily_df["news_date"] = single_date
                frames.append(daily_df)
            for domain, count in daily_counts.items():
                if domain not in domain_counts:
                    domain_counts[domain] = 0
                domain_counts[domain] += count

        if frames:
            news_df = pd.concat(frames, ignore_index=True)
        else:
            news_df = pd.DataFrame(columns=news_columns)
        news_df.to_csv(news_csv, index=False, columns=output_news_columns)
        print(f"Saved {len(news_df)} news rows to {news_csv}")

        matched_domains = sorted([domain for domain, count in domain_counts.items() if count])
        missing_domains = sorted([domain for domain, count in domain_counts.items() if not count])
        print(
            f"Domains with matches: {', '.join(matched_domains) if matched_domains else 'none'}"
        )
        print(
            f"Domains without matches: {', '.join(missing_domains) if missing_domains else 'none'}"
        )

    # --- Step 2: fetch NVDA prices based on requested window ---------------
    use_trading_day_mode = price_trading_days and price_trading_days > 0
    if use_trading_day_mode:
        combined_prices, price_metadata = fetch_prices_from_day(
            target_date=price_label,
            interval_minutes=interval_minutes,
            trading_days=price_trading_days,
        )
    else:
        market_tz = ZoneInfo(CONFIG.market_timezone)
        anchor_ts = pd.Timestamp(f"{price_label} 09:30:00").tz_localize(market_tz)
        combined_prices, price_metadata = fetch_nvda_prices_around_event(
            anchor_ts,
            hours_after=price_hours_after,
            interval_minutes=interval_minutes,
        )
        price_metadata = {
            **price_metadata,
            "anchor_timestamp": anchor_ts.isoformat(),
            "anchor_reason": "explicit_price_date_hours_after",
            "hours_requested": price_hours_after,
        }

    prices_csv = output_dir / f"nvda_prices_{price_label}.csv"
    combined_prices.to_csv(prices_csv, index=False)
    if not combined_prices.empty:
        print(f"Saved {len(combined_prices)} intraday price rows to {prices_csv}")
    else:
        print("No price data fetched.")

    if price_metadata:
        metadata_path = output_dir / f"nvda_price_metadata_{price_label}.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(price_metadata, handle, indent=2)
        print(f"Price fetch metadata saved to {metadata_path} for verification.")


if __name__ == "__main__":
    main()
