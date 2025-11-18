## Data Fetching Tool

This directory contains everything needed to scrape AI-related news via Google News RSS and pull NVDA intraday bars from Polygon.

### Contents
- `data_fetching.py` – main CLI for downloading news + prices and saving CSV/JSON artifacts
- `fetch_news.py` – shared Google News RSS scraper used by the CLI
- `config.yaml` – sample configuration (ticker, keywords, API keys, timezones, etc.)
- `requirements.txt` – minimal dependency list for the tool

### Setup
```bash
cd data_fetching_tool
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Fill in your Polygon API key inside `config.yaml` (under `apis.stocks.key`). Adjust keywords plus the `defaults.*` values to control news and price windows separately.

### Usage
```bash
python data_fetching.py \
  --config config.yaml \
  --news-date 2025-10-10 \
  --news-end-date 2025-10-11 \
  --price-date 2025-10-12 \
  --price-hours-after 48 \
  --keywords government shutdown nvidia
```

Flags of interest:
- `--news-date` / `--news-end-date` – control the inclusive range of days pulled from Google News (falls back to `defaults.news_date` / `defaults.news_end_date`)
- `--price-date` – anchor for the NVDA price pull (defaults to the news date if omitted)
- `--price-hours-after` – length of the price window in hours after the price-date market open
- `--price-trading-days` – fetch an integer number of full trading sessions instead of using hours (`0` disables this mode)
- `--interval-minutes` – resolution of the Polygon bars
- `--skip-news` – skip the Google News scrape and only pull prices
- `--news-timezone` – override the timezone used for the news date window

Outputs are written to `NVDA_<PRICE_DATE>/` (or the ticker configured in `config.yaml`).
