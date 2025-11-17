#!/usr/bin/env python3
"""
Simple margin calculator - shows how margins are calculated.
"""


def calculate_margins(current_price, upper_pct, lower_pct):
    """
    Calculate margin price levels from percentages.
    
    Args:
        current_price: Current stock price
        upper_pct: Expected upward move percentage (e.g., 4.53 for +4.53%)
        lower_pct: Expected downward move percentage (e.g., -5.77 for -5.77%)
    
    Returns:
        Dictionary with upper_margin and lower_margin prices
    """
    
    # Convert percentages to decimals
    upper_decimal = upper_pct / 100
    lower_decimal = lower_pct / 100
    
    # Calculate price levels
    upper_margin = current_price * (1 + upper_decimal)
    lower_margin = current_price * (1 + lower_decimal)
    
    return {
        'current_price': current_price,
        'upper_pct': upper_pct,
        'lower_pct': lower_pct,
        'upper_margin': upper_margin,
        'lower_margin': lower_margin,
        'spread': upper_margin - lower_margin,
        'spread_pct': ((upper_margin - lower_margin) / current_price) * 100
    }


def display_margins(result):
    """Display margins in a nice format."""
    
    print("\n" + "="*70)
    print("📊 MARGIN CALCULATION")
    print("="*70)
    
    print(f"\n💰 Current Price: ${result['current_price']:.2f}")
    print(f"\n📈 Upper Margin (SELL): ${result['upper_margin']:.2f} ({result['upper_pct']:+.2f}%)")
    print(f"   → If price reaches ${result['upper_margin']:.2f}, SELL (take profit)")
    
    print(f"\n📉 Lower Margin (BUY):  ${result['lower_margin']:.2f} ({result['lower_pct']:+.2f}%)")
    print(f"   → If price drops to ${result['lower_margin']:.2f}, BUY (buy the dip)")
    
    print(f"\n📏 Spread: ${result['spread']:.2f} ({result['spread_pct']:.2f}%)")
    print(f"   → Price range between buy and sell levels")
    
    print("\n" + "="*70)
    print("💡 TRADING STRATEGY")
    print("="*70)
    print(f"\n  Price ≥ ${result['upper_margin']:.2f}  →  SELL (take profit)")
    print(f"  Price ≤ ${result['lower_margin']:.2f}  →  BUY (buy the dip)")
    print(f"  ${result['lower_margin']:.2f} < Price < ${result['upper_margin']:.2f}  →  HOLD (wait)")
    print()


def calculate_profit(entry_price, exit_price, shares=100):
    """
    Calculate profit from a trade.
    
    Args:
        entry_price: Price you bought at
        exit_price: Price you sold at
        shares: Number of shares (default: 100)
    
    Returns:
        Dictionary with profit calculations
    """
    
    profit = (exit_price - entry_price) * shares
    profit_pct = ((exit_price - entry_price) / entry_price) * 100
    
    return {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'shares': shares,
        'profit': profit,
        'profit_pct': profit_pct
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate trading margins")
    parser.add_argument("--price", type=float, required=True, 
                       help="Current stock price")
    parser.add_argument("--upper", type=float, required=True,
                       help="Upper margin percentage (e.g., 4.53 for +4.53%%)")
    parser.add_argument("--lower", type=float, required=True,
                       help="Lower margin percentage (e.g., -5.77 for -5.77%%)")
    parser.add_argument("--shares", type=int, default=100,
                       help="Number of shares for profit calculation (default: 100)")
    
    args = parser.parse_args()
    
    # Calculate margins
    result = calculate_margins(args.price, args.upper, args.lower)
    
    # Display
    display_margins(result)
    
    # Show profit scenarios
    print("="*70)
    print("💵 PROFIT SCENARIOS (for {} shares)".format(args.shares))
    print("="*70)
    
    # Scenario 1: Buy at current, sell at upper margin
    profit1 = calculate_profit(args.price, result['upper_margin'], args.shares)
    print(f"\n📈 Scenario 1: Buy at ${args.price:.2f}, sell at upper margin")
    print(f"   Entry: ${profit1['entry_price']:.2f}")
    print(f"   Exit:  ${profit1['exit_price']:.2f}")
    print(f"   Profit: ${profit1['profit']:.2f} ({profit1['profit_pct']:+.2f}%)")
    
    # Scenario 2: Buy at lower margin, sell at current
    profit2 = calculate_profit(result['lower_margin'], args.price, args.shares)
    print(f"\n📉 Scenario 2: Buy at lower margin, sell at ${args.price:.2f}")
    print(f"   Entry: ${profit2['entry_price']:.2f}")
    print(f"   Exit:  ${profit2['exit_price']:.2f}")
    print(f"   Profit: ${profit2['profit']:.2f} ({profit2['profit_pct']:+.2f}%)")
    
    # Scenario 3: Buy at lower margin, sell at upper margin (best case)
    profit3 = calculate_profit(result['lower_margin'], result['upper_margin'], args.shares)
    print(f"\n🎯 Scenario 3: Buy at lower margin, sell at upper margin (BEST CASE)")
    print(f"   Entry: ${profit3['entry_price']:.2f}")
    print(f"   Exit:  ${profit3['exit_price']:.2f}")
    print(f"   Profit: ${profit3['profit']:.2f} ({profit3['profit_pct']:+.2f}%)")
    
    print("\n" + "="*70)

