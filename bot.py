"""
TradeBot — Main Orchestrator
Runs 24/7, scans markets every POLL_INTERVAL seconds,
executes trades, manages positions, and learns from outcomes.

Usage:
    python bot.py

    # With a specific pair override:
    python bot.py --pairs BTC-USD ETH-USD
"""

import argparse
import logging
import sys
import time
from datetime import datetime

import config
from coinbase_client import CoinbaseClient
from strategy       import Strategy
from learning       import TradeLogger, AdaptiveLearner
from position_manager import PositionManager

# ── Logging Setup ──────────────────────────────────────────────────────────────
def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ]
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=fmt, handlers=handlers)

logger = logging.getLogger("TradeBot.Main")


# ── Bot ────────────────────────────────────────────────────────────────────────
class TradeBot:
    def __init__(self, pairs: list = None):
        self.pairs     = pairs or config.TRADING_PAIRS
        self.client    = CoinbaseClient()
        self.strategy  = Strategy(self.client)
        self.trade_log = TradeLogger()
        self.learner   = AdaptiveLearner(self.trade_log)
        self.positions = PositionManager(self.trade_log)

        mode = "🟡 DRY RUN" if config.DRY_RUN else "🟢 LIVE"
        logger.info("=" * 60)
        logger.info(f"  TradeBot starting — {mode}")
        logger.info(f"  Pairs:    {', '.join(self.pairs)}")
        logger.info(f"  Interval: {config.POLL_INTERVAL}s")
        logger.info("=" * 60)

    # ── Per-pair scan ──────────────────────────────────────────────────────────
    def _scan_pair(self, product_id: str, portfolio_usd: float):
        try:
            # ── Manage open positions first ───────────────────────────────────
            if self.positions.has_position(product_id):
                current_price = self.client.get_ticker(product_id)
                df = self.client.get_candles(product_id, config.PRIMARY_TIMEFRAME, limit=50)
                from technical_analysis import add_indicators
                df = add_indicators(df)
                atr = df["atr"].iloc[-1] if not df.empty else current_price * 0.01

                exit_reason = self.positions.check_exits(product_id, current_price, atr)
                if exit_reason:
                    pos = self.positions.positions[product_id]
                    self.client.place_market_sell(product_id, pos.base_quantity)
                    self.positions.close_position(product_id, current_price)
                    logger.info(f"EXIT {product_id} | Reason: {exit_reason} | Price: {current_price:.4f}")
                else:
                    pos = self.positions.positions[product_id]
                    pnl = pos.unrealised_pnl_pct(current_price)
                    logger.debug(f"HOLD {product_id} | Price={current_price:.4f} | PnL={pnl:+.2f}% "
                                 f"| SL={pos.stop_loss:.4f} (trailing={'ON' if pos.trailing_active else 'OFF'})")
                return

            # ── Check if we can open more positions ───────────────────────────
            if len(self.positions.positions) >= config.MAX_OPEN_TRADES:
                logger.debug(f"Max open positions ({config.MAX_OPEN_TRADES}) reached. Skipping {product_id}.")
                return

            # ── Evaluate setup ────────────────────────────────────────────────
            signal = self.strategy.evaluate(product_id, portfolio_usd)

            # ── Blend confidence with ML ──────────────────────────────────────
            features = self._extract_signal_features(product_id)
            confidence = self.learner.predict_confidence(features, signal.confidence)

            if signal.action == "buy" and confidence >= 0.55:
                usd_size = portfolio_usd * config.MAX_RISK_PER_TRADE * 10  # approx position size
                usd_size = min(usd_size, portfolio_usd * 0.20)
                if usd_size < 5:
                    logger.warning(f"Position size too small (${usd_size:.2f}). Skipping.")
                    return

                result = self.client.place_market_buy(product_id, usd_size)
                current_price = signal.entry_price or self.client.get_ticker(product_id)
                base_qty      = usd_size / current_price if current_price > 0 else 0

                trade_id = self.trade_log.log_open(
                    product_id, current_price,
                    signal.stop_loss, signal.take_profit,
                    confidence, features
                )
                self.positions.open_position(
                    product_id, trade_id,
                    current_price, signal.stop_loss, signal.take_profit,
                    base_qty, usd_size
                )
                logger.info(
                    f"✅ BUY  {product_id} | ${usd_size:.2f} | "
                    f"conf={confidence:.0%} | {signal.reason[:80]}"
                )

            elif signal.action == "hold":
                logger.debug(f"⏸  HOLD {product_id} | {signal.reason[:100]}")

        except Exception as e:
            logger.error(f"Error scanning {product_id}: {e}", exc_info=True)

    def _extract_signal_features(self, product_id: str) -> dict:
        """Pull latest indicator values for the ML feature vector."""
        try:
            from technical_analysis import add_indicators
            df  = self.client.get_candles(product_id, config.PRIMARY_TIMEFRAME, limit=60)
            df  = add_indicators(df)
            row = df.iloc[-1]
            return {
                "rsi":                    float(row.get("rsi", 50)),
                "macd_hist":              float(row.get("macd_hist", 0)),
                "vol_ratio":              float(row.get("vol_ratio", 1)),
                "atr_pct":                float(row.get("atr", 0) / row.get("close", 1)),
                "zone_strength":          1,
                "distance_to_zone_pct":   0.01,
                "htf_bullish":            0,
                "ltf_bullish":            0,
                "signal_confidence":      0.5,
            }
        except Exception:
            return {}

    # ── Main Loop ──────────────────────────────────────────────────────────────
    def run(self):
        scan_count = 0
        while True:
            try:
                scan_count += 1
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                portfolio_usd = self.client.get_usd_balance()

                logger.info(f"{'─'*50}")
                logger.info(f"Scan #{scan_count} | {now} | Portfolio: ${portfolio_usd:.2f}")

                # Periodic ML retraining
                if scan_count % (config.RETRAIN_INTERVAL * 3600 // config.POLL_INTERVAL) == 0:
                    logger.info("Triggering ML retrain…")
                    self.learner.train()

                # Performance summary every 100 scans
                if scan_count % 100 == 0:
                    summary = self.learner.get_performance_summary()
                    logger.info(f"📊 Performance: {summary}")
                    prices = {p: self.client.get_ticker(p) for p in self.pairs}
                    logger.info(f"📌 Open positions:\n{self.positions.summary(prices)}")

                # Scan all pairs
                for pair in self.pairs:
                    self._scan_pair(pair, portfolio_usd)
                    time.sleep(0.5)   # Small delay between pairs (rate limiting)

            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)

            time.sleep(config.POLL_INTERVAL)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Coinbase Trade Bot")
    parser.add_argument("--pairs", nargs="+", help="Override trading pairs, e.g. --pairs BTC-USD ETH-USD")
    parser.add_argument("--live",  action="store_true", help="Enable LIVE trading (overrides config DRY_RUN=True)")
    args = parser.parse_args()

    if args.live:
        logger.warning("⚠️  LIVE trading mode enabled! Real orders will be placed.")
        config.DRY_RUN = False

    bot = TradeBot(pairs=args.pairs)
    bot.run()
