"""
Technical Analysis Engine
Detects support/resistance zones, market structure (BOS/CHoCH),
and computes indicators needed for trade decisions.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config

logger = logging.getLogger("TradeBot.TA")


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Zone:
    """A support or resistance price zone."""
    level: float          # Mid-price of zone
    zone_type: str        # 'support' | 'resistance'
    strength: int         # Number of touches / confirming bars
    high: float           # Upper boundary
    low: float            # Lower boundary
    last_touch_idx: int   # Index of last confirmed touch
    broken: bool = False  # Zone has been breached

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def distance_pct(self, price: float) -> float:
        return abs(price - self.level) / self.level


@dataclass
class MarketStructure:
    """Snapshot of higher-timeframe market structure."""
    trend: str                      # 'bullish' | 'bearish' | 'ranging'
    last_bos: Optional[str] = None  # 'bullish_bos' | 'bearish_bos'
    last_choch: Optional[str] = None
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False


@dataclass
class Signal:
    """A generated trade signal."""
    product_id: str
    action: str           # 'buy' | 'sell' | 'hold'
    reason: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float     # 0.0 – 1.0
    zone: Optional[Zone] = None


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a candle DataFrame."""
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["ema_20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema_50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    # ── Momentum ───────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Volatility ─────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["vol_sma"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_sma"]   # > 1.5 = high-volume bar

    # ── Candle Patterns ────────────────────────────────────────────────────────
    body      = (c - df["open"]).abs()
    candle_rng = h - l
    df["body_ratio"]    = body / candle_rng.replace(0, np.nan)
    df["upper_wick"]    = h - c.clip(lower=df["open"])
    df["lower_wick"]    = c.clip(upper=df["open"]) - l
    df["is_pin_bar"]    = (
        (df["lower_wick"] > body * 2) |
        (df["upper_wick"] > body * 2)
    )
    df["is_engulfing"]  = (
        (c > df["open"]) &
        (c > df["open"].shift()) &
        (df["open"] < c.shift())
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Swing Detection
# ─────────────────────────────────────────────────────────────────────────────

def find_swing_highs(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """Return boolean Series – True where a swing high occurs."""
    n = lookback or config.SWING_LOOKBACK
    highs = df["high"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(n, len(df) - n):
        window = highs.iloc[i - n: i + n + 1]
        if highs.iloc[i] == window.max():
            is_swing.iloc[i] = True
    return is_swing


def find_swing_lows(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    n = lookback or config.SWING_LOOKBACK
    lows = df["low"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(n, len(df) - n):
        window = lows.iloc[i - n: i + n + 1]
        if lows.iloc[i] == window.min():
            is_swing.iloc[i] = True
    return is_swing


# ─────────────────────────────────────────────────────────────────────────────
# Support / Resistance Zone Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_zones(df: pd.DataFrame) -> Tuple[List[Zone], List[Zone]]:
    """
    Identify support and resistance zones using swing highs/lows.
    Returns (support_zones, resistance_zones) sorted by strength.
    """
    sh_mask = find_swing_highs(df)
    sl_mask = find_swing_lows(df)

    current_price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1] if "atr" in df.columns else current_price * 0.005

    raw_resistance: List[dict] = []
    raw_support:    List[dict] = []

    for idx in df.index[sh_mask]:
        price = df.loc[idx, "high"]
        raw_resistance.append({"price": price, "idx": idx})

    for idx in df.index[sl_mask]:
        price = df.loc[idx, "low"]
        raw_support.append({"price": price, "idx": idx})

    def cluster_levels(raw: list, zone_type: str) -> List[Zone]:
        if not raw:
            return []
        raw.sort(key=lambda x: x["price"])
        zones: List[Zone] = []
        merge_tol = config.ZONE_MERGE_PCT

        for item in raw:
            p   = item["price"]
            idx = item["idx"]
            merged = False
            for z in zones:
                if abs(p - z.level) / z.level < merge_tol:
                    # Merge into existing zone
                    z.strength      += 1
                    z.level          = (z.level * (z.strength - 1) + p) / z.strength
                    z.high           = max(z.high, p + atr * 0.25)
                    z.low            = min(z.low,  p - atr * 0.25)
                    z.last_touch_idx = max(z.last_touch_idx, idx)
                    merged = True
                    break
            if not merged:
                zones.append(Zone(
                    level         = p,
                    zone_type     = zone_type,
                    strength      = 1,
                    high          = p + atr * 0.25,
                    low           = p - atr * 0.25,
                    last_touch_idx= idx,
                ))

        # Keep only zones with enough touches
        zones = [z for z in zones if z.strength >= config.MIN_ZONE_TOUCHES]
        # Mark broken zones
        for z in zones:
            if zone_type == "resistance" and current_price > z.high:
                z.broken = True
            if zone_type == "support" and current_price < z.low:
                z.broken = True

        return sorted(zones, key=lambda z: z.strength, reverse=True)

    support_zones    = cluster_levels(raw_support,    "support")
    resistance_zones = cluster_levels(raw_resistance, "resistance")

    logger.debug(f"Found {len(support_zones)} support zones, {len(resistance_zones)} resistance zones")
    return support_zones, resistance_zones


# ─────────────────────────────────────────────────────────────────────────────
# Market Structure Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_structure(df: pd.DataFrame) -> MarketStructure:
    """
    Determine trend via swing sequence (HH/HL = bullish, LH/LL = bearish).
    Detect Break of Structure (BOS) and Change of Character (CHoCH).
    """
    sh_mask = find_swing_highs(df, lookback=5)
    sl_mask = find_swing_lows(df,  lookback=5)

    swing_h_prices = df.loc[sh_mask, "high"].values[-6:]
    swing_l_prices = df.loc[sl_mask, "low"].values[-6:]

    ms = MarketStructure(trend="ranging")
    ms.swing_highs = list(swing_h_prices)
    ms.swing_lows  = list(swing_l_prices)

    if len(swing_h_prices) >= 2 and len(swing_l_prices) >= 2:
        hh = swing_h_prices[-1] > swing_h_prices[-2]
        hl = swing_l_prices[-1] > swing_l_prices[-2]
        lh = swing_h_prices[-1] < swing_h_prices[-2]
        ll = swing_l_prices[-1] < swing_l_prices[-2]

        ms.higher_high = hh
        ms.higher_low  = hl
        ms.lower_high  = lh
        ms.lower_low   = ll

        if hh and hl:
            ms.trend = "bullish"
        elif lh and ll:
            ms.trend = "bearish"
        else:
            ms.trend = "ranging"

        # BOS: price closes beyond last significant swing
        last_close = df["close"].iloc[-1]
        if last_close > swing_h_prices[-2] and ms.trend != "bullish":
            ms.last_bos = "bullish_bos"
        elif last_close < swing_l_prices[-2] and ms.trend != "bearish":
            ms.last_bos = "bearish_bos"

        # CHoCH: opposite-direction break after trend
        if ms.trend == "bullish" and ll:
            ms.last_choch = "bearish_choch"
        elif ms.trend == "bearish" and hh:
            ms.last_choch = "bullish_choch"

    return ms


# ─────────────────────────────────────────────────────────────────────────────
# Rejection Detection (at resistance)
# ─────────────────────────────────────────────────────────────────────────────

def detect_rejection(df: pd.DataFrame, zone: Zone) -> bool:
    """
    Returns True if recent candles show rejection at a zone
    (e.g., upper wicks, engulfing reversal, or pin bar at resistance).
    """
    recent = df.iloc[-config.REJECTION_CANDLES:]
    touches_zone = any(zone.contains(row["high"]) or zone.contains(row["close"])
                       for _, row in recent.iterrows())
    if not touches_zone:
        return False

    # Check for bearish rejection signals
    last = df.iloc[-1]
    has_pin_bar   = last.get("is_pin_bar", False) and last["upper_wick"] > last["lower_wick"]
    has_engulfing = last["close"] < last["open"]  # bearish close after zone touch

    return bool(has_pin_bar or has_engulfing)


def detect_support_reaction(df: pd.DataFrame, zone: Zone) -> bool:
    """
    Returns True if recent candles show bullish reaction at a support zone
    (e.g., lower wick rejection, bullish engulf).
    """
    recent = df.iloc[-config.REJECTION_CANDLES:]
    touches_zone = any(zone.contains(row["low"]) or zone.contains(row["close"])
                       for _, row in recent.iterrows())
    if not touches_zone:
        return False

    last = df.iloc[-1]
    has_pin_bar   = last.get("is_pin_bar", False) and last["lower_wick"] > last["upper_wick"]
    has_engulfing = last["close"] > last["open"]  # bullish close after zone touch

    return bool(has_pin_bar or has_engulfing)
