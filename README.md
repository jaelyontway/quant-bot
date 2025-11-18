## NVDA Event‑Driven Pipeline

This repo automates three pieces of the project:

1. **`data_fetching/`** – Scrape AI‑related news via Google News RSS and pull NVDA intraday bars from Polygon.
2. **`training/`** – Build and persist a Random Forest model that predicts upper/lower trading margins from sentiment.
3. **`simulation/`** – Drive the lightweight plotting/exit-signal simulation from the fetched CSV outputs.

The shared dependencies for all scripts live in `requirements.txt`.

---

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements list now includes the analysis stack (`numpy`, `matplotlib`, `scikit-learn`) plus sentiment/LLM helpers (`vaderSentiment`, `openai`) that the training and integration layers import.

---

### 2. Configure Credentials and Defaults

Copy `config/config.yaml` (or edit in place) and fill in:

- `apis.stocks.key` with your Polygon key.
- `defaults.keywords`, `defaults.news_date`, etc. to control the date windows.

The configuration is shared by both `data_fetching/data_fetching.py` and the simulation helpers.

---

### 3. Fetch News + Prices

```bash
cd data_fetching
python data_fetching.py \
  --config ../config/config.yaml \
  --news-date 2025-10-10 \
  --news-end-date 2025-10-11 \
  --price-date 2025-10-12 \
  --price-hours-after 48 \
  --keywords government shutdown nvidia
```

Common flags:
- `--news-date` / `--news-end-date` – inclusive Google News range (defaults can be set in YAML).
- `--price-date` – anchor day for NVDA prices (falls back to the news date).
- `--price-hours-after` – length of the post-event window; pair with `--price-trading-days 0` to use hours instead of sessions.
- `--price-trading-days` – fetch whole trading days (set to `0` to disable).
- `--interval-minutes` – Polygon bar resolution.
- `--skip-news` – reuse news CSVs and only re-pull prices.
- `--news-timezone` – override the timezone used for news windows.

Each run emits `NVDA_<DATE>/` with `news_*.csv`, `nvda_prices_*.csv`, and metadata JSON.

---

### 4. Predict Margins / Train ML

1. **Create training data** – derive combined sentiment/price rows or reuse existing CSVs.
2. **Train**:
   ```bash
   python training/train_ml_model.py \
     --training-data training_data.csv \
     --output models/margin_predictor.pkl \
     --test-size 0.2
   ```
3. **Use predictions** – `training/src/integrate_margins.py` loads the fetched CSVs and calls `margin_predictor.predict_margins` (rule-based by default, ML when the model exists).

---

### 5. Run the Simulation

1. Set the desired date in `simulation/targetSimDate.txt`.
2. Fetch and copy the CSV automatically:
   ```bash
   cd simulation
   python preSimulationController.py
   ```
   This temporarily overwrites the YAML date, reruns the fetch helper, and drops `NVDA_prices.csv` in the simulation folder.
3. Launch the plot/exit analysis:
   ```bash
   python simulation_program.py
   ```
   The script visualizes the upper/lower margins, marks the first exit signal, and prints the realized vs. hold returns.

---

### Quick Start

Need a refresher? `quick_start.md` contains the minimal commands to pull data for Oct 1–20 and points to the canonical CSVs.
