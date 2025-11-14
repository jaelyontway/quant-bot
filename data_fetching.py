#!/usr/bin/env python3
"""Utility script to link AI-related news to NVDA intraday price reactions."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from typing import Sequence
from urllib.parse import urlparse

import pandas as pd
import requests
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
) -> pd.DataFrame:
    """Fetch AI-related news for a given date across the fixed domain list."""

    tz_name = timezone or CONFIG.default_news_timezone
    tz = ZoneInfo(tz_name)
    start_ts = pd.Timestamp(f"{target_date} 00:00:00", tz=tz)
    end_ts = pd.Timestamp(f"{target_date} 23:59:59", tz=tz)

    articles: list[Article] = []

    for keyword in keywords:
        payload = _query_news_api(keyword, start_ts, end_ts)
        for raw_article in payload:
            url = raw_article.get("url", "")
            domain = _extract_domain(url)
            if domain not in CONFIG.news_domains:
                # Some APIs ignore domain filters, so double-check client-side.
                continue

            article = Article(
                published_at=raw_article.get("publishedAt") or raw_article.get("published_at", ""),
                source_domain=domain,
                title=raw_article.get("title", ""),
                description=raw_article.get("description") or raw_article.get("summary", ""),
                url=url,
                keyword=keyword,
            )
            articles.append(article)

    df = pd.DataFrame([a.__dict__ for a in articles])
    if not df.empty:
        df = df.sort_values("published_at").reset_index(drop=True)
    return df

def _query_news_api(keyword: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[dict]:
    """Call the configured news API for a given keyword and window."""

    if CONFIG.news_api_url == "offline":
        return _mock_news_response(keyword, start_ts)

    params = {
        "q": keyword,
        "from": start_ts.isoformat(),
        "to": end_ts.isoformat(),
        "language": "en",
        "pageSize": 100,
        "domains": ",".join(CONFIG.news_domains),
        "apiKey": CONFIG.news_api_key,
    }
    response = requests.get(CONFIG.news_api_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("articles", [])


def _mock_news_response(keyword: str, start_ts: pd.Timestamp) -> list[dict]:
    base_time = start_ts + pd.Timedelta(hours=10)
    domain = CONFIG.news_domains[0] if CONFIG.news_domains else "example.com"
    return [
        {
            "publishedAt": (base_time + pd.Timedelta(minutes=i * 30)).isoformat(),
            "title": f"{keyword.title()} headline #{i+1}",
            "description": f"Synthetic description for {keyword} article {i+1}.",
            "url": f"https://{domain}/{keyword}/{i}",
        }
        for i in range(3)
    ]

def _extract_domain(url: str) -> str:
    """Extract netloc from a URL without subpaths."""

    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "") if parsed.netloc else ""


def fetch_nvda_prices_around_event(
    event_time: pd.Timestamp,
    hours_after: int = 72,
    interval_minutes: int = 60,
) -> pd.DataFrame:
    """Fetch NVDA intraday bars from the event time for the next `hours_after` hours."""

    market_tz = ZoneInfo(CONFIG.market_timezone)
    event_time_local = _ensure_timezone(event_time, market_tz)
    end_time_local = event_time_local + pd.Timedelta(hours=hours_after)

    raw_bars = _query_stock_api(event_time_local, end_time_local, interval_minutes)
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
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    first_close = df["close"].iloc[0]
    df["pct_change_from_event"] = (df["close"] / first_close) - 1
    df["log_return"] = df["close"].apply(math.log).diff()
    return df


def _query_stock_api(start_ts: pd.Timestamp, end_ts: pd.Timestamp, interval_minutes: int) -> list[dict]:
    """Fetch NVDA intraday data, falling back to mock data if offline."""

    if CONFIG.stock_api_url == "offline":
        return _mock_price_response(start_ts, interval_minutes, end_ts)

    # Yahoo Finance fetch path
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yfinance is required for Yahoo Finance data. Install via `pip install yfinance`."
        ) from exc

    start_utc = start_ts.tz_convert("UTC")
    end_utc = end_ts.tz_convert("UTC")

    ticker = yf.Ticker(CONFIG.ticker)
    interval = f"{interval_minutes}m"
    history = ticker.history(
        interval=interval,
        start=start_utc.to_pydatetime(),
        end=end_utc.to_pydatetime(),
        auto_adjust=False,
    )

    if history.empty:
        return []

    df = history.reset_index().rename(
        columns={
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.to_dict(orient="records")


def _mock_price_response(start_ts: pd.Timestamp, interval_minutes: int, end_ts: pd.Timestamp) -> list[dict]:
    timestamps = pd.date_range(start_ts, end_ts, freq=f"{interval_minutes}min", inclusive="left")
    baseline = 450.0
    data = []
    for idx, ts in enumerate(timestamps):
        price = baseline + 0.5 * idx
        data.append(
            {
                "timestamp": ts.isoformat(),
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.1,
                "volume": 1000 + idx * 5,
            }
        )
    return data


def _ensure_timezone(ts: pd.Timestamp, tz: ZoneInfo) -> pd.Timestamp:
    """Force a pandas Timestamp to have the provided timezone."""

    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def derive_news_events(news_df: pd.DataFrame, news_timezone: str) -> pd.DataFrame:
    """Collapse many articles per keyword into a single first-seen event."""

    if news_df.empty:
        return pd.DataFrame(columns=["keyword", "event_timestamp", "source_domain", "title", "description", "url"])

    tz = ZoneInfo(news_timezone)
    df = news_df.copy()
    df["published_at_ts"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at_ts"])
    df["published_at_ts"] = df["published_at_ts"].apply(
        lambda ts: ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    )

    df = df.sort_values("published_at_ts")
    earliest = df.drop_duplicates(subset=["keyword"], keep="first").copy()
    earliest = earliest.rename(columns={"published_at_ts": "event_timestamp"})
    event_cols = ["keyword", "event_timestamp", "source_domain", "title", "description", "url"]
    return earliest[event_cols].reset_index(drop=True)


def link_news_and_prices(
    events_df: pd.DataFrame,
    hours_after: int,
    interval_minutes: int,
) -> pd.DataFrame:
    """Fetch NVDA prices for each aggregated news event."""

    if events_df.empty:
        return pd.DataFrame()

    price_frames: list[pd.DataFrame] = []

    for event_id, row in events_df.reset_index(drop=True).iterrows():
        event_ts = row["event_timestamp"]
        prices = fetch_nvda_prices_around_event(event_ts, hours_after, interval_minutes)
        if prices.empty:
            continue
        prices["event_id"] = event_id
        prices["event_published_at"] = event_ts.isoformat()
        prices["event_title"] = row["title"]
        prices["event_source_domain"] = row["source_domain"]
        prices["event_keyword"] = row["keyword"]
        price_frames.append(prices)

    if not price_frames:
        return pd.DataFrame()

    combined = pd.concat(price_frames, ignore_index=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Link AI news to NVDA intraday prices.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to a YAML config file (defaults baked in if missing)",
    )
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format")
    parser.add_argument(
        "--keywords",
        nargs="+",
        required=True,
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

    news_timezone = args.news_timezone or CONFIG.default_news_timezone
    hours_after = args.hours_after or CONFIG.default_hours_after
    interval_minutes = args.interval_minutes or CONFIG.default_interval_minutes

    news_df = fetch_news_for_date(args.date, args.keywords, news_timezone)

    news_csv = f"news_{args.date}.csv"
    news_df.to_csv(news_csv, index=False)
    print(f"Saved {len(news_df)} news rows to {news_csv}")

    events_df = derive_news_events(news_df, news_timezone)
    events_csv = f"news_events_{args.date}.csv"
    events_df.to_csv(events_csv, index=False)
    print(f"Identified {len(events_df)} distinct events -> {events_csv}")

    combined_prices = link_news_and_prices(
        events_df=events_df,
        hours_after=hours_after,
        interval_minutes=interval_minutes,
    )

    prices_csv = f"nvda_prices_around_news_{args.date}.csv"
    combined_prices.to_csv(prices_csv, index=False)
    print(
        f"Saved {len(combined_prices)} intraday price rows to {prices_csv}" if not combined_prices.empty else "No price data fetched."
    )


if __name__ == "__main__":
    main()
