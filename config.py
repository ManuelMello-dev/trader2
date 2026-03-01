"""
Configuration file for the Coinbase Trade Bot.
Fill in your API credentials and adjust parameters to your risk tolerance.
"""

# ── Coinbase Advanced Trade API ────────────────────────────────────────────────
API_KEY    = "YOUR_API_KEY_HERE"        # from Coinbase → Settings → API
API_SECRET = "YOUR_API_SECRET_HERE"     # Keep this secret, never commit to git

# ── Trading Pairs ──────────────────────────────────────────────────────────────
TRADING_PAIRS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
]

# ── Timeframes (used for multi-timeframe analysis) ─────────────────────────────
PRIMARY_TIMEFRAME   = "ONE_HOUR"     # Main trading timeframe
TREND_TIMEFRAME     = "SIX_HOUR"    # Higher timeframe for trend/structure bias
ENTRY_TIMEFRAME     = "FIFTEEN_MINUTE"  # Lower timeframe for precise entries

# Coinbase valid granularities:
# ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE,
# ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY

# ── Risk Management ────────────────────────────────────────────────────────────
MAX_RISK_PER_TRADE  = 0.02   # 2% of portfolio per trade
MAX_OPEN_TRADES     = 3      # Max simultaneous positions
STOP_LOSS_ATR_MULT  = 1.5    # Stop loss = 1.5x ATR below support
TAKE_PROFIT_RATIO   = 2.0    # Risk:Reward ratio minimum

# ── Structure Detection ────────────────────────────────────────────────────────
SWING_LOOKBACK      = 10     # Bars to look back for swing high/low detection
STRUCTURE_TOUCH_PCT = 0.003  # 0.3% tolerance for "touching" a level
ZONE_MERGE_PCT      = 0.005  # Merge zones within 0.5% of each other
MIN_ZONE_TOUCHES    = 2      # Minimum touches to confirm a S/R zone
REJECTION_CANDLES   = 2      # How many rejection candles needed at resistance

# ── Machine Learning / Adaptive Settings ──────────────────────────────────────
LEARNING_ENABLED    = True
TRADE_LOG_FILE      = "trade_history.json"
MODEL_FILE          = "ml_model.pkl"
MIN_TRADES_FOR_ML   = 20     # Minimum trades before ML adjustments kick in
RETRAIN_INTERVAL    = 24     # Hours between ML retraining

# ── Operational ────────────────────────────────────────────────────────────────
POLL_INTERVAL       = 60     # Seconds between market scans
LOG_LEVEL           = "INFO"
LOG_FILE            = "bot.log"
DRY_RUN             = True   # ⚠️  Set to False to place REAL orders
