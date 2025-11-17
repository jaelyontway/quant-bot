import numpy as np
import matplotlib.pyplot as plt

class Stock:
    def __init__(self):
        # 3 days * 24 hours/day * 2 samples/hour = 144 samples (30-min intervals)
        self._prices = np.zeros(3 * 24 * 2)
        self._upper_margin = np.zeros_like(self._prices)
        self._lower_margin = np.zeros_like(self._prices)
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


def plot_stock(stock: Stock):
    t = np.arange(len(stock.get_prices())) / 2  # in hours (30-min intervals)

    prices = stock.get_prices()
    upper = stock.get_upper_margin()
    lower = stock.get_lower_margin()

    # --- Find first time price hits a margin ---
    hit_upper = np.where(prices >= upper)[0]
    hit_lower = np.where(prices <= lower)[0]

    hit_index = None
    signal_type = None

    if len(hit_upper) > 0 and len(hit_lower) > 0:
        # whichever comes first in time
        if hit_upper[0] < hit_lower[0]:
            hit_index = hit_upper[0]
            signal_type = "sell"
        else:
            hit_index = hit_lower[0]
            signal_type = "buy"
    elif len(hit_upper) > 0:
        hit_index = hit_upper[0]
        signal_type = "sell"
    elif len(hit_lower) > 0:
        hit_index = hit_lower[0]
        signal_type = "buy"

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, prices, label="Stock Price", color="blue", linewidth=2)
    plt.plot(t, upper, 'g--', label="Upper Margin")
    plt.plot(t, lower, 'r--', label="Lower Margin")

    # plot single marker if a margin was hit
    if hit_index is not None:
        if signal_type == "sell":
            plt.plot(t[hit_index], prices[hit_index], 'gv', markersize=12, label="Good Exit")
        else:
            plt.plot(t[hit_index], prices[hit_index], 'r^', markersize=12, label="Bad Exit")

    plt.title(f"{stock.get_name()} 3-Day Simulation")
    plt.xlabel("Time (hours)")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    return hit_index

def plot_stock2(stock: Stock):
    t = np.arange(len(stock.get_prices())) / 2  # in hours (30-min intervals)

    prices = stock.get_prices()
    

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, prices, label="Stock Price", color="blue", linewidth=2)
    
    if prices[-1]>prices[0]:
        plt.plot(t[-1], prices[-1], 'gv', markersize=12, label="Good Exit")
    else:
        plt.plot(t[-1], prices[-1], 'r^', markersize=12, label="Bad Exit")
    
    breakEvenLine=np.zeros_like(prices)
    for i,val in enumerate(breakEvenLine):
        breakEvenLine[i] = prices[0]
    plt.plot(t, breakEvenLine, 'r--', label="Break Even")
    
    # plot single marker if a margin was hit
    plt.title(f"{stock.get_name()} 3-Day Simulation")
    plt.xlabel("Time (hours)")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
# Example usage:
if __name__ == "__main__":
    s = Stock()
    SnP=Stock()
    SnP.set_name("S&P500")
    SnP.set_notes("S&P placeholder")
    s.set_name("AAPL")
    s.set_notes("Example: Set upper and lower margins at x$ and y$ respectively due to market mistrust after government shutdown")

    # Simulated price data
    t = np.arange(3 * 6.5 * 2)
    prices = 150 + np.sin(t / 5) * 2 + np.random.normal(0, 0.5, len(t))
    prices2= 120 + np.cos(t / 5) * 2 + np.random.normal(0, 0.5, len(t))
    upper_margin = 152 + np.zeros_like(prices)
    lower_margin = 148 + np.zeros_like(prices)
    s.set_prices(prices)
    SnP.set_prices(prices2)
    s.set_upper_margin(upper_margin)
    s.set_lower_margin(lower_margin)
    
    '''plots'''
    exit_index=plot_stock(s)
    '''
    plot_stock2(SnP)
    '''
    plt.show()
    
    '''some results setup'''
    numShares=1
    initialStockPrice=s.get_prices()[0]
    entryEquity=numShares*initialStockPrice
    exitEquity = numShares*s.get_prices()[exit_index]
    totalDiff="{:.2f}".format(exitEquity-entryEquity)
    
    '''RESULTS'''
    if float(totalDiff)>0:
        print("The bot made ",str(totalDiff),"$",sep="")
    else:
        print("The bot lost ",str(totalDiff),"$",sep="")
    
    finalValueEquity="{:.2f}".format(numShares*(s.get_prices()[-1]-initialStockPrice))
    print("holding the stock until the end of the simulation would have yielded ",finalValueEquity,"$",sep="")

    finalSnP=SnP.get_prices()[-1]
    finalValueEquitySnP="{:.2f}".format(numShares*(SnP.get_prices()[-1]-SnP.get_prices()[0]))
    print("Buying S&P500 and selling at the end of the sim would have yielded ",finalValueEquitySnP,"$",sep="")
    print("The LLMs reasoning for its strategy are given as:")
    print(s.get_notes())