# Margin Calculation Guide

## 🎯 What Are Margins?

**Margins are price thresholds for trading decisions:**

- **Upper Margin (SELL)**: Price level to **SELL** and take profit
- **Lower Margin (BUY)**: Price level to **BUY** the dip

---

## 🧮 How Margins Are Calculated

### **Formula:**

```
Upper Margin = Current Price × (1 + Upper Percentage / 100)
Lower Margin = Current Price × (1 + Lower Percentage / 100)
```

### **Example:**

```
Current Price: $140.50
Upper Percentage: +4.53%
Lower Percentage: -5.77%

Upper Margin = $140.50 × (1 + 4.53/100)
             = $140.50 × 1.0453
             = $146.87

Lower Margin = $140.50 × (1 - 5.77/100)
             = $140.50 × 0.9423
             = $132.39
```

---

## 📊 Visual Representation

```
$146.87 ← Upper Margin (SELL HERE - Take Profit)
   ↑
   | Expected upward move: +4.53%
   |
$140.50 ← Current Price (YOU ARE HERE)
   |
   | Expected downward move: -5.77%
   ↓
$132.39 ← Lower Margin (BUY HERE - Buy the Dip)
```

**Spread:** $146.87 - $132.39 = $14.48 (10.30% of current price)

---

## 💡 Trading Strategy

| Price Level | Action | Reason |
|------------|--------|--------|
| ≥ $146.87 | **SELL** | Price hit upper margin (take profit) |
| $132.39 - $146.87 | **HOLD** | Price within margins (wait) |
| ≤ $132.39 | **BUY** | Price hit lower margin (buy the dip) |

---

## 💵 Profit Scenarios (100 shares)

### **Scenario 1: Buy Now, Sell at Upper Margin**
```
Entry: $140.50 × 100 = $14,050
Exit:  $146.87 × 100 = $14,687
Profit: $637 (+4.53%)
```

### **Scenario 2: Buy at Lower Margin, Sell Now**
```
Entry: $132.39 × 100 = $13,239
Exit:  $140.50 × 100 = $14,050
Profit: $811 (+6.12%)
```

### **Scenario 3: Buy at Lower, Sell at Upper (BEST CASE)**
```
Entry: $132.39 × 100 = $13,239
Exit:  $146.87 × 100 = $14,687
Profit: $1,448 (+10.93%)
```

---

## 🚀 Quick Commands

### **1. Get Margins from News Text:**

```bash
python simple_predict.py \
    --text "NVIDIA announces breakthrough chip" \
    --price 140.50
```

**Output:**
```
📈 Upper Margin (SELL): $146.87 (+4.53%)
📉 Lower Margin (BUY):  $132.39 (-5.77%)
```

### **2. Calculate Margins from Percentages:**

```bash
python calculate_margins.py \
    --price 140.50 \
    --upper 4.53 \
    --lower -5.77 \
    --shares 100
```

Shows detailed calculations and profit scenarios.

---

## 🤖 How the ML Model Predicts Percentages

The Random Forest model learns from historical data:

1. **Input:** News text → Sentiment score (e.g., +0.74 for bullish)
2. **Features:** 6 engineered features (sentiment, volatility, etc.)
3. **Output:** Upper % and Lower % predictions
4. **Calculation:** Percentages → Price levels (margins)

**Example:**
```
News: "NVIDIA announces breakthrough chip"
  ↓ (Sentiment Analysis)
Sentiment: +0.74 (bullish)
  ↓ (Random Forest ML Model)
Upper: +4.53%, Lower: -0.14%
  ↓ (Price Calculation)
Upper Margin: $146.87, Lower Margin: $140.30
```

---

## 📈 Real-World Example

### **Day 1: Get Prediction**
```bash
python simple_predict.py \
    --text "NVIDIA beats earnings, revenue up 50%" \
    --price 140.50
```

**Output:**
```
Upper Margin: $146.87 (SELL here)
Lower Margin: $132.39 (BUY here)
```

### **Day 2-5: Monitor Price**

- **If price reaches $146.87** → SELL (take profit)
- **If price drops to $132.39** → BUY (buy the dip)
- **If price stays between** → HOLD (wait)

### **Day 6: Price hits $147.00**

✅ **SELL!** Price exceeded upper margin.

**Profit (100 shares):**
```
Entry: $140.50 × 100 = $14,050
Exit:  $147.00 × 100 = $14,700
Profit: $650 (+4.63%)
```

---

## 🎓 Key Concepts

### **Spread**
The difference between upper and lower margins.
```
Spread = Upper Margin - Lower Margin
       = $146.87 - $132.39
       = $14.48
```

Larger spread = More volatility expected

### **Confidence**
How confident the model is in its prediction (0-1 scale).
- High confidence (>0.7): Strong signal
- Medium confidence (0.4-0.7): Moderate signal
- Low confidence (<0.4): Weak signal

### **Method**
- `ml_trained`: Using trained ML model (better)
- `rule_based`: Using sentiment rules (fallback)

---

## ✅ Summary

**To calculate margins:**

1. Get percentages from ML model:
   ```bash
   python simple_predict.py --text "YOUR NEWS" --price CURRENT_PRICE
   ```

2. Margins are calculated as:
   ```
   Upper = Price × (1 + Upper%/100)
   Lower = Price × (1 + Lower%/100)
   ```

3. Use margins for trading:
   - Price ≥ Upper → SELL
   - Price ≤ Lower → BUY
   - Between → HOLD

**That's it!** 🚀

