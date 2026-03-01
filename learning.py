"""
Adaptive Learning Module
Tracks every trade outcome and uses a simple ML model
(Gradient Boosting) to adjust confidence thresholds and zone weights.
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("TradeBot.Learning")


# ─────────────────────────────────────────────────────────────────────────────
# Trade Log
# ─────────────────────────────────────────────────────────────────────────────

class TradeLogger:
    """Persists every trade and its outcome to JSON."""

    def __init__(self, log_file: str = None):
        self.log_file = log_file or config.TRADE_LOG_FILE
        self.trades: list = self._load()

    def _load(self) -> list:
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

    def log_open(self, product_id: str, entry_price: float,
                 stop_loss: float, take_profit: float,
                 signal_confidence: float, features: dict) -> str:
        trade_id = f"{product_id}_{int(time.time())}"
        record = {
            "trade_id":          trade_id,
            "product_id":        product_id,
            "entry_price":       entry_price,
            "stop_loss":         stop_loss,
            "take_profit":       take_profit,
            "signal_confidence": signal_confidence,
            "features":          features,
            "entry_time":        datetime.utcnow().isoformat(),
            "exit_price":        None,
            "exit_time":         None,
            "pnl_pct":           None,
            "outcome":           None,   # 'win' | 'loss' | 'break_even'
        }
        self.trades.append(record)
        self._save()
        logger.info(f"Trade opened: {trade_id}")
        return trade_id

    def log_close(self, trade_id: str, exit_price: float):
        for t in self.trades:
            if t["trade_id"] == trade_id:
                t["exit_price"] = exit_price
                t["exit_time"]  = datetime.utcnow().isoformat()
                pnl = (exit_price - t["entry_price"]) / t["entry_price"]
                t["pnl_pct"] = round(pnl * 100, 4)
                t["outcome"] = "win" if pnl > 0.001 else ("loss" if pnl < -0.001 else "break_even")
                self._save()
                logger.info(f"Trade closed: {trade_id} | PnL={t['pnl_pct']:.2f}% | {t['outcome']}")
                return
        logger.warning(f"Trade {trade_id} not found in log.")

    def get_closed_trades(self) -> list:
        return [t for t in self.trades if t["outcome"] is not None]

    def win_rate(self) -> float:
        closed = self.get_closed_trades()
        if not closed:
            return 0.5
        wins = sum(1 for t in closed if t["outcome"] == "win")
        return wins / len(closed)

    def avg_rr(self) -> float:
        """Average realised Risk:Reward."""
        closed = self.get_closed_trades()
        if not closed:
            return 0.0
        rr_list = []
        for t in closed:
            risk   = abs(t["entry_price"] - t["stop_loss"])
            reward = abs(t["exit_price"] - t["entry_price"])
            if risk > 0:
                rr_list.append(reward / risk)
        return float(np.mean(rr_list)) if rr_list else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ML Confidence Adjuster
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLearner:
    """
    Trains a Gradient Boosting classifier on historical trade features
    to predict whether a setup will be profitable.
    The predicted probability replaces or blends with the rule-based confidence.
    """

    def __init__(self, trade_logger: TradeLogger):
        self.logger_  = trade_logger
        self.model    = None
        self.trained  = False
        self.last_train_time = 0
        self._try_load_model()

    def _try_load_model(self):
        if os.path.exists(config.MODEL_FILE):
            with open(config.MODEL_FILE, "rb") as f:
                self.model   = pickle.load(f)
                self.trained = True
            logger.info("ML model loaded from disk.")

    def _save_model(self):
        with open(config.MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def _extract_features(trade: dict) -> Optional[list]:
        feat = trade.get("features")
        if not feat:
            return None
        return [
            feat.get("rsi", 50),
            feat.get("macd_hist", 0),
            feat.get("vol_ratio", 1),
            feat.get("atr_pct", 0.01),
            feat.get("zone_strength", 1),
            feat.get("distance_to_zone_pct", 0.01),
            feat.get("htf_bullish", 0),
            feat.get("ltf_bullish", 0),
            feat.get("signal_confidence", 0.5),
        ]

    def train(self, force: bool = False):
        """Train or retrain the model if enough data exists."""
        if not config.LEARNING_ENABLED:
            return

        now = time.time()
        hours_since = (now - self.last_train_time) / 3600
        if not force and self.trained and hours_since < config.RETRAIN_INTERVAL:
            return

        closed = self.logger_.get_closed_trades()
        if len(closed) < config.MIN_TRADES_FOR_ML:
            logger.info(f"Not enough trades to train ({len(closed)} / {config.MIN_TRADES_FOR_ML}). Skipping.")
            return

        X, y = [], []
        for t in closed:
            feats = self._extract_features(t)
            if feats is None:
                continue
            X.append(feats)
            y.append(1 if t["outcome"] == "win" else 0)

        if len(X) < config.MIN_TRADES_FOR_ML:
            return

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.1, random_state=42
                )),
            ])
            clf.fit(X, y)
            self.model   = clf
            self.trained = True
            self.last_train_time = now
            self._save_model()
            logger.info(f"ML model trained on {len(X)} trades. Accuracy on training set: "
                        f"{clf.score(X, y):.0%}")
        except ImportError:
            logger.warning("scikit-learn not installed. Run: pip install scikit-learn")

    def predict_confidence(self, features: dict, rule_confidence: float) -> float:
        """
        Blend rule-based confidence with ML probability.
        If model not yet trained, return rule confidence unchanged.
        """
        if not self.trained or self.model is None:
            return rule_confidence

        feat_vec = [
            features.get("rsi", 50),
            features.get("macd_hist", 0),
            features.get("vol_ratio", 1),
            features.get("atr_pct", 0.01),
            features.get("zone_strength", 1),
            features.get("distance_to_zone_pct", 0.01),
            features.get("htf_bullish", 0),
            features.get("ltf_bullish", 0),
            rule_confidence,
        ]
        try:
            ml_prob = self.model.predict_proba([feat_vec])[0][1]
            # 60/40 blend: more weight to ML as data accumulates
            n_trades = len(self.logger_.get_closed_trades())
            ml_weight = min(0.60, n_trades / 200)
            blended = ml_weight * ml_prob + (1 - ml_weight) * rule_confidence
            logger.debug(f"Confidence: rule={rule_confidence:.2f}, ML={ml_prob:.2f}, blended={blended:.2f}")
            return round(blended, 3)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return rule_confidence

    def get_performance_summary(self) -> dict:
        closed = self.logger_.get_closed_trades()
        if not closed:
            return {"message": "No closed trades yet."}
        total  = len(closed)
        wins   = sum(1 for t in closed if t["outcome"] == "win")
        losses = sum(1 for t in closed if t["outcome"] == "loss")
        pnls   = [t["pnl_pct"] for t in closed if t["pnl_pct"] is not None]
        return {
            "total_trades":  total,
            "win_rate":      f"{wins/total*100:.1f}%",
            "loss_rate":     f"{losses/total*100:.1f}%",
            "avg_pnl_pct":   f"{np.mean(pnls):.2f}%",
            "best_trade":    f"{max(pnls):.2f}%",
            "worst_trade":   f"{min(pnls):.2f}%",
            "avg_rr":        f"{self.logger_.avg_rr():.2f}",
            "model_trained": self.trained,
        }
