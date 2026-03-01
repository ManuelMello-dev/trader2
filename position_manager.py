"""
Position Manager
Tracks open positions, manages trailing stops, and decides when to exit.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import config

logger = logging.getLogger("TradeBot.Positions")


@dataclass
class Position:
    product_id:    str
    trade_id:      str
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    base_quantity: float    # How much crypto we hold
    usd_invested:  float
    entry_time:    float    = field(default_factory=time.time)
    highest_price: float    = 0.0   # For trailing stop
    trailing_active: bool   = False

    def unrealised_pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price * 100

    def update_trailing(self, current_price: float, atr: float) -> float:
        """
        Move stop up as price rises (trailing stop).
        Activates once price is 1 ATR in profit.
        Returns new stop_loss level.
        """
        if current_price > self.highest_price:
            self.highest_price = current_price

        # Activate trailing once 1R in profit
        profit_r = (current_price - self.entry_price) / max(self.entry_price - self.stop_loss, 1e-8)
        if profit_r >= 1.0:
            self.trailing_active = True

        if self.trailing_active:
            trail_stop = self.highest_price - atr * config.STOP_LOSS_ATR_MULT
            self.stop_loss = max(self.stop_loss, trail_stop)

        return self.stop_loss


class PositionManager:
    def __init__(self, trade_logger):
        self.positions: Dict[str, Position] = {}   # product_id → Position
        self.trade_logger = trade_logger

    def has_position(self, product_id: str) -> bool:
        return product_id in self.positions

    def open_position(self, product_id: str, trade_id: str,
                      entry_price: float, stop_loss: float,
                      take_profit: float, base_qty: float, usd_invested: float):
        pos = Position(
            product_id    = product_id,
            trade_id      = trade_id,
            entry_price   = entry_price,
            stop_loss     = stop_loss,
            take_profit   = take_profit,
            base_quantity = base_qty,
            usd_invested  = usd_invested,
            highest_price = entry_price,
        )
        self.positions[product_id] = pos
        logger.info(f"Position opened: {product_id} @ {entry_price:.4f} | "
                    f"SL={stop_loss:.4f} | TP={take_profit:.4f}")

    def close_position(self, product_id: str, exit_price: float):
        pos = self.positions.pop(product_id, None)
        if pos:
            self.trade_logger.log_close(pos.trade_id, exit_price)
            pnl = (exit_price - pos.entry_price) / pos.entry_price * 100
            logger.info(f"Position closed: {product_id} @ {exit_price:.4f} | PnL={pnl:.2f}%")

    def check_exits(self, product_id: str, current_price: float, atr: float) -> Optional[str]:
        """
        Check if a position should be closed.
        Returns: 'stop_loss' | 'take_profit' | 'trailing_stop' | None
        """
        pos = self.positions.get(product_id)
        if not pos:
            return None

        # Update trailing stop first
        new_sl = pos.update_trailing(current_price, atr)

        if current_price <= new_sl:
            reason = "trailing_stop" if pos.trailing_active else "stop_loss"
            logger.info(f"{product_id} hit {reason} at {current_price:.4f} (SL={new_sl:.4f})")
            return reason

        if current_price >= pos.take_profit:
            logger.info(f"{product_id} hit take_profit at {current_price:.4f}")
            return "take_profit"

        # Time-based exit: if position has been open > 5 days, reconsider
        hours_open = (time.time() - pos.entry_time) / 3600
        if hours_open > 120:
            pnl_pct = pos.unrealised_pnl_pct(current_price)
            if pnl_pct > 0.5:
                logger.info(f"{product_id} time-based exit after {hours_open:.1f}h with +{pnl_pct:.2f}%")
                return "time_exit_profit"

        return None

    def summary(self, prices: dict) -> str:
        if not self.positions:
            return "No open positions."
        lines = []
        for pid, pos in self.positions.items():
            price = prices.get(pid, pos.entry_price)
            pnl   = pos.unrealised_pnl_pct(price)
            lines.append(
                f"  {pid}: entry={pos.entry_price:.2f} | "
                f"current={price:.2f} | PnL={pnl:+.2f}% | "
                f"SL={pos.stop_loss:.2f} | TP={pos.take_profit:.2f}"
            )
        return "\n".join(lines)
