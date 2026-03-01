# 🤖 Coinbase Trade Bot

A 24/7 autonomous crypto trading bot with technical analysis, market structure
detection, support/resistance trading, and adaptive machine learning.

---

## ✨ Features

| Module | What it does |
|---|---|
| **Technical Analysis** | Multi-timeframe OHLCV analysis with EMA, RSI, MACD, ATR, Bollinger Bands |
| **Support/Resistance** | Automatically detects and clusters S/R zones from swing highs/lows |
| **Market Structure** | Identifies BOS (Break of Structure) and CHoCH (Change of Character) |
| **Smart Entry** | Buys at support with bullish confirmation, holds when structure continues |
| **Smart Exit** | Sells at resistance on rejection candles; ignores noise if trend is intact |
| **Trailing Stop** | Activates after 1R profit, locks in gains as price moves up |
| **Adaptive ML** | Gradient Boosting model learns from every trade, blends with rule confidence |
| **Risk Management** | Fixed-fractional sizing (2% risk per trade), max 3 simultaneous positions |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get your Coinbase API key
1. Go to [Coinbase → Settings → API](https://www.coinbase.com/settings/api)
2. Create a new **Advanced Trade** API key
3. Enable **View** + **Trade** permissions
4. Copy the API Key and API Secret

### 3. Configure the bot
Open `config.py` and fill in:
```python
API_KEY    = "your_api_key"
API_SECRET = "your_api_secret"
```

Adjust risk parameters to taste:
```python
MAX_RISK_PER_TRADE = 0.02   # 2% per trade
MAX_OPEN_TRADES    = 3
STOP_LOSS_ATR_MULT = 1.5    # Stop = 1.5x ATR below support
TAKE_PROFIT_RATIO  = 2.0    # 2:1 risk/reward minimum
```

### 4. Run in DRY RUN mode first (no real orders)
```bash
python bot.py
```
The bot defaults to `DRY_RUN = True` in config.py. This lets you watch it work without risking money.

### 5. Go live when you're confident
```bash
python bot.py --live
```
Or set `DRY_RUN = False` in config.py.

---

## 📁 File Structure

```
trade_bot/
├── bot.py                # Main 24/7 loop — start here
├── config.py             # All settings (API keys, risk params)
├── coinbase_client.py    # Coinbase Advanced Trade API wrapper
├── technical_analysis.py # Indicators, zones, structure detection
├── strategy.py           # Buy/Sell/Hold decision logic
├── position_manager.py   # Manages open trades + trailing stops
├── learning.py           # ML model + trade history logger
├── requirements.txt
└── README.md
```

---

## 📊 Trading Logic

### Entry (BUY)
1. Higher timeframe is **bullish or ranging**
2. Price is **at or near a support zone** (confirmed by 2+ touches)
3. **Bullish reaction** on entry timeframe (pin bar with lower wick, or bullish engulf)
4. RSI not overbought, MACD histogram trending positive
5. Confidence score ≥ 0.55 (blended from rules + ML model)

### Hold
- No zone touch AND structure still printing **higher highs / higher lows**
- Trailing stop is moved up automatically once in 1R profit
- Time-based exit after 5 days if still in profit

### Exit (SELL)
Any of:
- **Stop loss** hit (below support zone − 1.5×ATR)
- **Take profit** reached (2× risk distance)
- **Trailing stop** triggered (highest price − 1.5×ATR)
- Price at **resistance zone** with rejection candle
- **CHoCH** (Change of Character) detected on entry timeframe
- RSI > 70 with high-volume rejection

---

## 🧠 Adaptive Learning

After **20+ closed trades**, the ML model activates:
- Features: RSI, MACD histogram, volume ratio, ATR%, zone strength, proximity, structure alignment
- Model: Gradient Boosting Classifier (scikit-learn)
- Retrains every 24 hours
- Confidence blending: starts 0% ML weight, grows to 60% as data accumulates

Trade history is stored in `trade_history.json`. The trained model is saved to `ml_model.pkl`.

---

## ⚠️ Risk Warnings

- **This is experimental software.** Crypto markets are volatile and unpredictable.
- **Never risk money you cannot afford to lose.**
- Always run in DRY RUN mode first and review logs before going live.
- Past performance of any strategy does not guarantee future results.
- Consider testing with tiny amounts (e.g. $10) before scaling up.

---

## 🔧 Customisation Tips

**Add more pairs:**
```python
TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
```

**Change timeframes:**
```python
PRIMARY_TIMEFRAME = "FOUR_HOUR"    # More conservative
ENTRY_TIMEFRAME   = "ONE_HOUR"
```

**Tighten risk:**
```python
MAX_RISK_PER_TRADE = 0.01   # 1% per trade
TAKE_PROFIT_RATIO  = 3.0    # 3:1 RR
```

**Adjust zone sensitivity:**
```python
SWING_LOOKBACK     = 15    # Larger = fewer, more significant swings
MIN_ZONE_TOUCHES   = 3     # More touches = stronger zones only
```
