"""
Trading Strategy
Integrates technical analysis into actionable buy/sell/hold signals.

Logic:
  BUY  → price at support + bullish reaction + bullish higher-TF structure
  SELL → price at resistance + rejection + or bearish structure CHoCH
  HOLD → structure is intact with no rejection; trailing stop manages risk
"""

import logging
import pandas as pd
from typing import Optional, Tuple

import config
from technical_analysis import (
    add_indicators, detect_zones, analyse_structure,
    detect_rejection, detect_support_reaction,
    Signal, Zone, MarketStructure
)

logger = logging.getLogger("TradeBot.Strategy")


class Strategy:
    def __init__(self, client):
        self.client = client

    # ── Data Loading ───────────────────────────────────────────────────────────
    def _load(self, product_id: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        df = self.client.get_candles(product_id, timeframe, limit)
        if df.empty:
            return df
        return add_indicators(df)

    # ── Entry Sizing ───────────────────────────────────────────────────────────
    def _position_size(self, entry: float, stop: float, portfolio_usd: float) -> float:
        """Kelly-inspired fixed-fractional sizing."""
        risk_usd  = portfolio_usd * config.MAX_RISK_PER_TRADE
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return 0.0
        units   = risk_usd / risk_per_unit
        usd_val = units * entry
        return round(min(usd_val, portfolio_usd * 0.20), 2)  # Cap 20% per position

    # ── Main Signal Generator ──────────────────────────────────────────────────
    def evaluate(self, product_id: str, portfolio_usd: float) -> Signal:
        """
        Run full multi-timeframe analysis and return a Signal.
        """
        logger.info(f"Evaluating {product_id}…")

        # ── Load data ──────────────────────────────────────────────────────────
        df_trend  = self._load(product_id, config.TREND_TIMEFRAME, limit=200)
        df_primary = self._load(product_id, config.PRIMARY_TIMEFRAME, limit=300)
        df_entry  = self._load(product_id, config.ENTRY_TIMEFRAME,  limit=200)

        if df_primary.empty or df_entry.empty:
            return Signal(product_id, "hold", "No data", 0, 0, 0, 0)

        current_price = df_primary["close"].iloc[-1]
        atr           = df_primary["atr"].iloc[-1]

        # ── Higher TF structure (bias) ─────────────────────────────────────────
        htf_structure = analyse_structure(df_trend) if not df_trend.empty else analyse_structure(df_primary)

        # ── Primary TF zones ───────────────────────────────────────────────────
        support_zones, resistance_zones = detect_zones(df_primary)

        # ── Entry TF confirmation ──────────────────────────────────────────────
        entry_structure = analyse_structure(df_entry)

        # ── Filter zones near current price ───────────────────────────────────
        proximity = config.STRUCTURE_TOUCH_PCT * 8   # 2.4% radius for "near"

        nearby_support = [
            z for z in support_zones
            if not z.broken and z.distance_pct(current_price) < proximity
            and current_price >= z.low
        ]
        nearby_resistance = [
            z for z in resistance_zones
            if not z.broken and z.distance_pct(current_price) < proximity
            and current_price <= z.high
        ]

        # ── RSI / MACD context ─────────────────────────────────────────────────
        rsi        = df_primary["rsi"].iloc[-1]
        macd_bull  = df_primary["macd_hist"].iloc[-1] > 0
        macd_cross = (df_primary["macd_hist"].iloc[-1] > 0 and
                      df_primary["macd_hist"].iloc[-2] <= 0)
        vol_high   = df_primary["vol_ratio"].iloc[-1] > 1.3

        # ══════════════════════════════════════════════════════════════════════
        # BUY CONDITIONS
        # ══════════════════════════════════════════════════════════════════════
        if nearby_support and htf_structure.trend in ("bullish", "ranging"):
            best_zone = max(nearby_support, key=lambda z: z.strength)
            reaction  = detect_support_reaction(df_entry, best_zone)

            confidence = _score_buy(
                reaction, htf_structure, entry_structure,
                rsi, macd_bull, macd_cross, vol_high, best_zone
            )

            if reaction and confidence >= 0.55:
                stop_loss   = best_zone.low - atr * config.STOP_LOSS_ATR_MULT
                take_profit = current_price + (current_price - stop_loss) * config.TAKE_PROFIT_RATIO
                usd_size    = self._position_size(current_price, stop_loss, portfolio_usd)

                reason = (
                    f"Price at support zone {best_zone.level:.2f} (strength={best_zone.strength}), "
                    f"bullish reaction detected, HTF={htf_structure.trend}, "
                    f"RSI={rsi:.1f}, conf={confidence:.0%}"
                )
                logger.info(f"{product_id} → BUY | {reason}")
                return Signal(product_id, "buy", reason, current_price,
                              stop_loss, take_profit, confidence, best_zone)

        # ══════════════════════════════════════════════════════════════════════
        # SELL / EXIT CONDITIONS
        # ══════════════════════════════════════════════════════════════════════
        if nearby_resistance:
            best_zone = max(nearby_resistance, key=lambda z: z.strength)
            rejection = detect_rejection(df_entry, best_zone)

            # Sell at resistance if:
            #   a) price showing rejection candles, OR
            #   b) bearish CHoCH on entry TF, OR
            #   c) HTF trend turned bearish
            should_sell = (
                rejection or
                entry_structure.last_choch == "bearish_choch" or
                htf_structure.trend == "bearish" or
                (rsi > 70 and vol_high)
            )

            if should_sell:
                confidence = _score_sell(
                    rejection, htf_structure, entry_structure,
                    rsi, vol_high, best_zone
                )
                reason = (
                    f"Price at resistance zone {best_zone.level:.2f} (strength={best_zone.strength}), "
                    f"rejection={rejection}, CHoCH={entry_structure.last_choch}, "
                    f"HTF={htf_structure.trend}, RSI={rsi:.1f}"
                )
                logger.info(f"{product_id} → SELL | {reason}")
                return Signal(product_id, "sell", reason, current_price,
                              0, 0, confidence, best_zone)

        # ══════════════════════════════════════════════════════════════════════
        # HOLD — structure continues, no rejection
        # ══════════════════════════════════════════════════════════════════════
        reason = (
            f"No actionable setup. HTF={htf_structure.trend}, "
            f"BOS={htf_structure.last_bos}, RSI={rsi:.1f}. "
            f"Support zones={len(support_zones)}, Resistance zones={len(resistance_zones)}"
        )
        logger.debug(f"{product_id} → HOLD | {reason}")
        return Signal(product_id, "hold", reason, current_price, 0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Scoring Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_buy(reaction, htf, ltf, rsi, macd_bull, macd_cross, vol_high, zone) -> float:
    score = 0.0
    score += 0.25 if reaction           else 0
    score += 0.20 if htf.trend == "bullish" else (0.10 if htf.trend == "ranging" else 0)
    score += 0.15 if ltf.trend == "bullish" else 0
    score += 0.10 if 40 <= rsi <= 60    else (0.05 if rsi < 40 else 0)
    score += 0.10 if macd_bull          else 0
    score += 0.10 if macd_cross         else 0
    score += 0.05 if vol_high           else 0
    score += min(0.05, zone.strength * 0.01)
    return round(min(score, 1.0), 3)


def _score_sell(rejection, htf, ltf, rsi, vol_high, zone) -> float:
    score = 0.0
    score += 0.30 if rejection           else 0
    score += 0.20 if htf.trend == "bearish" else (0.10 if htf.trend == "ranging" else 0)
    score += 0.15 if ltf.trend == "bearish" else 0
    score += 0.15 if rsi > 70           else (0.08 if rsi > 60 else 0)
    score += 0.10 if vol_high           else 0
    score += min(0.10, zone.strength * 0.02)
    return round(min(score, 1.0), 3)
