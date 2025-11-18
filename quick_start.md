# Getting news and certain stock price 

```
python data_fetching/data_fetching.py --config config/config.yaml
```

- The canonical intraday price history for Oct 1–20 lives in `NVDA_prices_2025-10-01_to-2025-10-20_with_weekdays.csv`. Re-run `combine_price_runs.py` after new fetches if you need to refresh it.
- Cleaned article bodies should be read from `news_data/clean_content_in.csv` before feeding downstream training or analysis steps.
