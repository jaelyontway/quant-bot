import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta
import matplotlib.dates as mdates

class Stock:
    def __init__(self):
        self._prices = np.array([])
        self._upper_margin = np.array([])
        self._lower_margin = np.array([])
        self._name = ""
        self._notes = ""

    # --- Getters ---
    def get_prices(self):
        return self._prices

    def get_upper_margin(self):
        return self._upper_margin

    def get_lower_margin(self):
        return self._lower_margin

    def get_name(self):
        return self._name

    def get_notes(self):
        return self._notes

    # --- Setters ---
    def set_prices(self, prices: np.ndarray):
        self._prices = prices

    def set_upper_margin(self, upper: np.ndarray):
        self._upper_margin = upper

    def set_lower_margin(self, lower: np.ndarray):
        self._lower_margin = lower

    def set_name(self, name: str):
        self._name = name

    def set_notes(self, notes: str):
        self._notes = notes


def load_prices_from_csv(filepath):
    timestamps = []
    prices = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            try:
                ts = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
                price = float(row[2])
                timestamps.append(ts)
                prices.append(price)
            except:
                continue

    return np.array(timestamps), np.array(prices)


def plot_stock(stock: Stock, timestamps, interval_minutes=30):
    prices = stock.get_prices()
    upper = stock.get_upper_margin()
    lower = stock.get_lower_margin()

    # --- x-axis as relative hours from first timestamp ---
    t = np.arange(len(prices)) * (interval_minutes / 60.0)  # hours

    # --- Detect day changes for vertical dotted lines ---
    day_change_indices = []
    for i in range(1, len(timestamps)):
        if timestamps[i].date() != timestamps[i-1].date():
            day_change_indices.append(i)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(t, prices, label="Stock Price", linewidth=2, color="blue")
    plt.plot(t, upper, 'g--', label="Upper Margin")
    plt.plot(t, lower, 'r--', label="Lower Margin")

    # Draw day boundary lines
    for idx in day_change_indices:
        plt.axvline(x=t[idx], linestyle=':', linewidth=1, color="black")

    # Determine exit signal
    hit_upper = np.where(prices >= upper)[0]
    hit_lower = np.where(prices <= lower)[0]

    hit_index = None
    signal_type = None

    if len(hit_upper) > 0 and (len(hit_lower) == 0 or hit_upper[0] < hit_lower[0]):
        hit_index = hit_upper[0]
        signal_type = "sell"
    elif len(hit_lower) > 0:
        hit_index = hit_lower[0]
        signal_type = "buy"

    # Plot exit
    if hit_index is not None:
        marker = 'gv' if signal_type == "sell" else 'r^'
        plt.plot(t[hit_index], prices[hit_index], marker, markersize=12, label="Exit Signal")
    else:
        # No margin hit → exit at last price
        marker = 'gv' if prices[-1] > prices[0] else 'r^'
        plt.plot(t[-1], prices[-1], marker, markersize=12, label="Exit Signal")
        hit_index = len(prices) - 1

    plt.title(f"{stock.get_name()} Price Simulation")
    plt.xlabel("Time (hours)")
    plt.ylabel("Price ($)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    return hit_index


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    FILE_NAME = "NVDA_prices.csv"

    s = Stock()
    s.set_name("NVDA")
    s.set_notes("Event-based price movement")

    # Load CSV
    timestamps, prices = load_prices_from_csv(FILE_NAME)
    s.set_prices(prices)

    # --- Get start and end dates from column A ---
    start_date = None
    end_date = None
    with open(FILE_NAME, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        all_rows = list(reader)
        # First row (row 2 in Excel)
        start_date_str = all_rows[0][0].split()[0]  # take first part only
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        # Last row
        end_date_str = all_rows[-1][0].split()[0]
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # Simple ±2% margins
    first_price = prices[0]
    s.set_upper_margin(first_price * 1.02 * np.ones_like(prices))
    s.set_lower_margin(first_price * 0.98 * np.ones_like(prices))

    # --- Plot with 30-min intervals ---
    exit_index = plot_stock(s, timestamps, interval_minutes=30)

    # Update title to include start and end dates
    plt.title(f"{s.get_name()} Price Simulation - {start_date.date()} → {end_date.date()}")

    plt.show()

    # --- Trading Results ---
    numShares = 1
    entryEquity = numShares * prices[0]
    exitEquity = numShares * prices[exit_index]
    totalDiff = exitEquity - entryEquity

    print(f"\nTrading Results for {s.get_name()}")
    if totalDiff >= 0:
        print(f"The bot made ${totalDiff:.2f} per share")
    else:
        print(f"The bot lost ${totalDiff:.2f} per share")

    holdDiff = numShares * (prices[-1] - prices[0])
    print(f"Holding until the end of the period would yield ${holdDiff:.2f}")
