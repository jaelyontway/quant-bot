#!/usr/bin/env python3
"""Utility script to link AI-related news to NVDA intraday price reactions."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import time, timedelta
from typing import Sequence
from urllib.parse import urlparse

import pandas as pd
import requests
from pathlib import Path
from zoneinfo import ZoneInfo

# ----- Configuration -------------------------------------------------------

DEFAULT_NEWS_DOMAINS = [
    "wsj.com",
    "ft.com",
    "reuters.com",
    "bloomberg.com",
    "nytimes.com",
    "cnbc.com",
    "marketwatch.com",
    "barrons.com",
    "economist.com",
    "forbes.com",
    "businessinsider.com",
    "goldmansachs.com",
]

DEFAULT_NEWS_API_URL = "https://newsapi.org/v2/everything"
DEFAULT_STOCK_API_URL = "https://api.polygon.io/v2/aggs/ticker/NVDA/range/60/minute"
DEFAULT_NEWS_TIMEZONE = "America/New_York"
DEFAULT_MARKET_TIMEZONE = "America/New_York"
DEFAULT_HOURS_AFTER = 72
DEFAULT_INTERVAL_MINUTES = 60


@dataclass
class AppConfig:
    """Configuration container loaded from YAML/env/defaults."""

    ticker: str
    news_api_url: str
    stock_api_url: str
    news_api_key: str
    stock_api_key: str
    news_domains: list[str]
    default_news_timezone: str
    market_timezone: str
    default_hours_after: int
    default_interval_minutes: int
    default_date: str | None
    default_keywords: list[str] | None
    trading_days: int


def _build_default_config() -> AppConfig:
    """Return default config using environment variables when available."""

    return AppConfig(
        ticker="NVDA",
        news_api_url=DEFAULT_NEWS_API_URL,
        stock_api_url=DEFAULT_STOCK_API_URL,
        news_api_key=os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY"),
        stock_api_key=os.getenv("STOCK_API_KEY", "YOUR_STOCK_API_KEY"),
        news_domains=list(DEFAULT_NEWS_DOMAINS),
        default_news_timezone=DEFAULT_NEWS_TIMEZONE,
        market_timezone=DEFAULT_MARKET_TIMEZONE,
        default_hours_after=DEFAULT_HOURS_AFTER,
        default_interval_minutes=DEFAULT_INTERVAL_MINUTES,
        default_date=None,
        default_keywords=None,
        trading_days=3,
    )


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
    _maybe_copy("news_api_url")
    _maybe_copy("stock_api_url")
    _maybe_copy("news_api_key")
    _maybe_copy("stock_api_key")
    _maybe_copy("news_domains")
    _maybe_copy("default_news_timezone")
    _maybe_copy("market_timezone")
    _maybe_copy("default_hours_after")
    _maybe_copy("default_interval_minutes")

    apis = overrides.get("apis", {})
    news_api = apis.get("news", {})
    stocks_api = apis.get("stocks", {})
    cfg["news_api_url"] = news_api.get("url", cfg["news_api_url"])
    cfg["news_api_key"] = news_api.get("key", cfg["news_api_key"])
    cfg["stock_api_url"] = stocks_api.get("url", cfg["stock_api_url"])
    cfg["stock_api_key"] = stocks_api.get("key", cfg["stock_api_key"])

    defaults_section = overrides.get("defaults", {})
    cfg["default_news_timezone"] = defaults_section.get(
        "news_timezone", cfg["default_news_timezone"]
    )
    cfg["default_hours_after"] = defaults_section.get("hours_after", cfg["default_hours_after"])
    cfg["default_interval_minutes"] = defaults_section.get(
        "interval_minutes", cfg["default_interval_minutes"]
    )
    cfg["default_date"] = defaults_section.get("date", cfg["default_date"])
    if "keywords" in defaults_section:
        cfg["default_keywords"] = defaults_section["keywords"]
    cfg["trading_days"] = defaults_section.get("trading_days", cfg["trading_days"])

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


def fetch_news_for_date(
    target_date: str,
    keywords: Sequence[str],
    timezone: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fetch AI-related news for a given date across the fixed domain list."""

    # --- Define the date window and pull raw articles ----------------------

    tz_name = timezone or CONFIG.default_news_timezone
    tz = ZoneInfo(tz_name)
    start_ts = pd.Timestamp(f"{target_date} 00:00:00", tz=tz)
    end_ts = pd.Timestamp(f"{target_date} 23:59:59", tz=tz)

    keyword_label = " AND ".join(keywords)
    articles: list[Article] = []
    domain_counts: dict[str, int] = {domain: 0 for domain in CONFIG.news_domains}

    # --- Query each domain individually to track coverage ------------------
    for domain in CONFIG.news_domains:
        logging.info("Fetching news for %s with keywords %s", domain, keyword_label)
        try:
            payload = _query_news_api(keywords, start_ts, end_ts, domain_filter=domain)
        except requests.HTTPError as exc:
            logging.warning("Failed to fetch %s: %s", domain, exc)
            continue

        if not payload:
            logging.info("No articles returned for %s", domain)
            continue

        for raw_article in payload:
            url = raw_article.get("url", "")
            if not url:
                continue
            if not _article_contains_keywords(raw_article, keywords):
                continue

            article = Article(
                published_at=raw_article.get("publishedAt") or raw_article.get("published_at", ""),
                source_domain=domain,
                title=raw_article.get("title", ""),
                description=raw_article.get("description") or raw_article.get("summary", ""),
                url=url,
                keyword=keyword_label,
            )
            articles.append(article)
            domain_counts[domain] += 1

    df = pd.DataFrame([a.__dict__ for a in articles])
    if not df.empty:
        df = df.sort_values("published_at")
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df, domain_counts


def _query_news_api(
    keywords: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    domain_filter: str | None = None,
) -> list[dict]:
    """Call the configured news API for a given keyword combination and window."""

    quoted = [kw.strip() for kw in keywords if kw.strip()]
    if not quoted:
        return []
    # NewsAPI supports boolean operators using uppercase AND.
    query = " AND ".join(quoted)

    params = {
        "q": query,
        "from": start_ts.isoformat(),
        "to": end_ts.isoformat(),
        "language": "en",
        "pageSize": 100,
        "apiKey": CONFIG.news_api_key,
    }
    params["domains"] = domain_filter or ",".join(CONFIG.news_domains)
    response = requests.get(CONFIG.news_api_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("articles", [])


def _article_contains_keywords(article: dict, keywords: Sequence[str]) -> bool:
    """Ensure the article text contains every user keyword."""

    haystack = " ".join(
        filter(
            None,
            [
                article.get("title", ""),
                article.get("description", ""),
                article.get("content", ""),
            ],
        )
    ).lower()
    for keyword in keywords:
        if keyword.lower() not in haystack:
            return False
    return True


def _extract_domain(url: str) -> str:
    """Extract netloc from a URL without subpaths."""

    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "") if parsed.netloc else ""


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

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Restrict to regular trading hours in market timezone
    market_tz = ZoneInfo(CONFIG.market_timezone)
    local_ts = df["timestamp"].dt.tz_convert(market_tz)
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

    query_meta = {
        **query_meta,
        "bars_returned": len(df),
        "first_timestamp": df["timestamp"].iloc[0].isoformat(),
        "last_timestamp": df["timestamp"].iloc[-1].isoformat(),
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


def fetch_prices_from_day(
    target_date: str,
    interval_minutes: int,
) -> tuple[pd.DataFrame, dict]:
    """Fetch NVDA prices covering `trading_days` full sessions starting from `target_date`."""

    market_tz = ZoneInfo(CONFIG.market_timezone)
    target_date_str = str(target_date)
    trading_days = CONFIG.trading_days
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Link AI news to NVDA intraday prices.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to a YAML config file (defaults baked in if missing)",
    )
    parser.add_argument("--date", default=None, help="Target date in YYYY-MM-DD format")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Keywords to search for (space-separated)",
    )
    parser.add_argument(
        "--hours-after",
        type=int,
        default=None,
        help="Number of hours of NVDA prices to fetch after each news event",
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
    args = parser.parse_args()

    global CONFIG
    CONFIG = load_config_from_file(args.config)

    # --- Resolve runtime options, allowing CLI args to override config -----
    news_timezone = args.news_timezone or CONFIG.default_news_timezone
    hours_after = args.hours_after or CONFIG.default_hours_after
    interval_minutes = args.interval_minutes or CONFIG.default_interval_minutes
    target_date_value = args.date or CONFIG.default_date
    keywords = args.keywords or CONFIG.default_keywords

    if not target_date_value:
        raise SystemExit(
            "No date provided. Supply --date or set defaults.date in the config file."
        )
    if not keywords:
        raise SystemExit(
            "No keywords provided. Supply --keywords or set defaults.keywords in the config file."
        )

    # --- Step 1: fetch news ------------------------------------------------
    target_date = str(target_date_value)
    output_dir = Path(f"{CONFIG.ticker}_{target_date}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing outputs to %s", output_dir)

    news_df, domain_counts = fetch_news_for_date(target_date, keywords, news_timezone)

    news_csv = output_dir / f"news_{target_date}.csv"
    news_df.to_csv(news_csv, index=False)
    print(f"Saved {len(news_df)} news rows to {news_csv}")

    matched_domains = sorted([domain for domain, count in domain_counts.items() if count])
    missing_domains = sorted([domain for domain, count in domain_counts.items() if not count])
    print(
        f"Domains with matches: {', '.join(matched_domains) if matched_domains else 'none'}"
    )
    print(
        f"Domains without matches: {', '.join(missing_domains) if missing_domains else 'none'}"
    )

    # --- Step 2: fetch NVDA prices from start of day -----------------------
    combined_prices, price_metadata = fetch_prices_from_day(
        target_date=target_date,
        interval_minutes=interval_minutes,
    )

    prices_csv = output_dir / f"nvda_prices_around_news_{target_date}.csv"
    combined_prices.to_csv(prices_csv, index=False)
    if not combined_prices.empty:
        print(f"Saved {len(combined_prices)} intraday price rows to {prices_csv}")
    else:
        print("No price data fetched.")

    if price_metadata:
        metadata_path = output_dir / f"nvda_price_metadata_{target_date}.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(price_metadata, handle, indent=2)
        print(f"Price fetch metadata saved to {metadata_path} for verification.")


if __name__ == "__main__":
    main()
