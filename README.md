# NVDA News & Price Linker

Utility scripts that fetch AI‑relevant news for NVIDIA (NVDA) from NewsAPI and link the earliest headlines to intraday Polygon price data. Designed for workflows where an LLM performs downstream event clustering and analysis.

## Requirements

- Python 3.11+ (repository already uses a `.venv`)
- Dependencies listed in `data_fetching.py` (`pandas`, `requests`, `zoneinfo`, etc.)
- Valid API keys:
  - `NEWS_API_KEY` (NewsAPI `everything` endpoint)
  - `STOCK_API_KEY` (Polygon aggregates endpoint)
  - Keys can be stored in environment variables or inside `nvidia-config.yaml`.

## Configuration (`nvidia-config.yaml`)

```yaml
apis:
  news:
    url: https://newsapi.org/v2/everything
    key: YOUR_NEWS_API_KEY
  stocks:
    url: https://api.polygon.io/v2/aggs/ticker
    key: YOUR_STOCK_API_KEY

ticker: NVDA
market_timezone: America/New_York
news_domains:
  - wsj.com
  - ft.com
  # ...

defaults:
  news_timezone: America/New_York
  interval_minutes: 60
  date: 2025-11-10
  trading_days: 3
  keywords:
    - AI
    - nvidia
```

- `news_domains`: domains NewsAPI should search. Articles outside this list are filtered out client‑side.
- `defaults.date`: run date in `YYYY-MM-DD`.
- `defaults.trading_days`: number of **full trading sessions** (09:30–16:00 ET) to capture for NVDA prices.
- `defaults.keywords`: every keyword must appear (title/description/content) for an article to be kept.

## Running the script

```bash
source .venv/bin/activate               # optional
python data_fetching.py --config nvidia-config.yaml
```

- Logging outputs appear in the terminal (e.g., which domains succeed or fail, output paths).
- Override defaults with CLI flags if needed, for example:

```bash
python data_fetching.py \
  --config nvidia-config.yaml \
  --date 2025-11-13 \
  --keywords "AI" "Nvidia" "earnings" \
  --interval-minutes 30
```

## Output

Every run writes files inside a folder named `{TICKER}_{DATE}` (e.g., `NVDA_2025-11-13/`):

- `news_<date>.csv` – Raw NewsAPI articles (one row per article) that matched all keywords.
- `nvda_prices_around_news_<date>.csv` – NVDA hourly bars for the requested number of trading days, filtered to regular hours (09:30–16:00 ET). Columns include OHLC, volume, pct change, and log return.
- `nvda_price_metadata_<date>.json` – Metadata describing the Polygon request (URL, timestamps, trading days covered). Use it to verify the price data source or replay the query.

## Plotting

`plot_nvda_prices.py` can visualize the price CSV:

```bash
python plot_nvda_prices.py NVDA_2025-11-13/nvda_prices_around_news_2025-11-13.csv --output nvda_plot.png
```

## Notes

- NewsAPI plans may not include every domain listed; missing domains are reported in the terminal logs.
- If you need per-domain coverage regardless of keyword matches, relax the keywords or query each domain with different criteria before LLM post-processing.
- Regular-hours filtering excludes pre-market/after-hours data; adjust `fetch_nvda_prices_around_event` if extended hours are required.

